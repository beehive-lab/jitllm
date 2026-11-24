package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched-prefill kernels of Kimi delta attention (KDA), shared by every family with KDA layers: a
 * chunk of prompt tokens, a row per token. What a token does after the tokens before it — the
 * convolution and the delta rule — walks the chunk's real rows ({@code batchInfo[1]}) in order
 * inside one launch, and leaves the windows and matrices where the decode kernels ({@link
 * KdaKernels}) resume.
 */
public final class KdaBatchKernels {

    private KdaBatchKernels() {}

    // @formatter:off
    /**
     * The causal depthwise convolution over the chunk's real rows in order, a lane per channel,
     * carrying the channel's window from token to token; the window is the decode kernel's ({@link
     * Qwen35DeltaNetKernels#causalConv1d}), so the decode resumes where this leaves it.
     */
    // @formatter:on
    public static void causalConv1d(
            KernelContext context,
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            IntArray batchInfo,
            int channels,
            int kernel,
            int windowOffset) {
        int c = context.globalIdx;
        if (c < channels) {
            int history = kernel - 1;
            int wBase = c * kernel;
            int hBase = windowOffset + c * history;
            int rows = batchInfo.get(1);
            for (int b = 0; b < rows; b++) {
                float x = input.get(b * channels + c);
                float sum = 0.0f;
                for (int t = 0; t < history; t++) {
                    sum += weight.get(wBase + t) * window.get(hBase + t);
                }
                sum += weight.get(wBase + history) * x;
                out.set(b * channels + c, sum);
                for (int t = 0; t + 1 < history; t++) {
                    window.set(hBase + t, window.get(hBase + t + 1));
                }
                if (history > 0) {
                    window.set(hBase + history - 1, x);
                }
            }
        }
    }

    /** {@link KdaKernels#kdaDecayAndBeta} over every row: {@code decay} and {@code beta} flat. */
    public static void kdaDecayAndBeta(
            KernelContext context,
            FloatArray decay,
            FloatArray dtBias,
            FloatArray a,
            FloatArray beta,
            int width,
            int headDim,
            int heads,
            float lowerBound,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * width) {
            int c = id % width;
            float g = -a.get(c / headDim) * (decay.get(id) + dtBias.get(c));
            decay.set(id, TornadoMath.exp(lowerBound * sigmoid(g)));
        }
        if (id < batch * heads) {
            beta.set(id, sigmoid(beta.get(id)));
        }
    }

    /** The widest KDA head {@link #kdaDeltaRule} keeps in local memory. */
    public static final int MAX_KDA_HEAD = 128;

    // @formatter:off
    /**
     * The KDA delta rule over the chunk's real rows in order, as {@link KdaKernels#kdaDeltaRule}
     * does for one token and with its layout: a workgroup takes {@link KdaKernels#DELTA_COLUMNS}
     * state columns of one head, each column's rows split into {@link KdaKernels#DELTA_CHUNKS}
     * chunks, a lane per (chunk, column), the chunks' sums added in chunk order.
     *
     * <p>The workgroup's slice of the state stays in local memory for the whole chunk: read once,
     * updated row by row, written back after the last row. Each row's key, query and decay are
     * staged once for all lanes.
     *
     * <p>Grid: {@code heads * ceil(headDim / DELTA_COLUMNS)} workgroups of {@code DELTA_COLUMNS *
     * DELTA_CHUNKS} lanes; {@code headDim} at most {@link #MAX_KDA_HEAD}.
     */
    // @formatter:on
    public static void kdaDeltaRule(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            IntArray batchInfo,
            int heads,
            int headDim,
            int stateOffset) {
        final int cols = KdaKernels.DELTA_COLUMNS;
        final int chunks = KdaKernels.DELTA_CHUNKS;
        int tid = context.localIdx;
        int column = tid % cols;
        int chunk = tid / cols;
        int tiles = (headDim + cols - 1) / cols;
        int h = context.groupIdx / tiles;
        int j = (context.groupIdx - h * tiles) * cols + column;
        float[] slice = context.allocateFloatLocalArray(MAX_KDA_HEAD * KdaKernels.DELTA_COLUMNS);
        float[] keys = context.allocateFloatLocalArray(MAX_KDA_HEAD);
        float[] queries = context.allocateFloatLocalArray(MAX_KDA_HEAD);
        float[] decays = context.allocateFloatLocalArray(MAX_KDA_HEAD);
        float[] sums =
                context.allocateFloatLocalArray(KdaKernels.DELTA_COLUMNS * KdaKernels.DELTA_CHUNKS);
        int rowsPerChunk = (headDim + chunks - 1) / chunks;
        int start = chunk * rowsPerChunk;
        int end = TornadoMath.min(headDim, start + rowsPerChunk);
        int active =
                TornadoMath.min(1, TornadoMath.max(0, heads - h))
                        * TornadoMath.min(1, TornadoMath.max(0, headDim - j));
        int stateBase = stateOffset + h * headDim * headDim + j;
        int width = heads * headDim;
        if (active == 1) {
            for (int i = start; i < end; i++) {
                slice[i * cols + column] = state.get(stateBase + i * headDim);
            }
        }
        int rows = batchInfo.get(1);
        for (int b = 0; b < rows; b++) {
            int base = b * width + h * headDim;
            context.localBarrier();
            if (tid < headDim && h < heads) {
                keys[tid] = k.get(base + tid);
                queries[tid] = q.get(base + tid);
                decays[tid] = decay.get(base + tid);
            }
            context.localBarrier();
            float correction = 0.0f;
            for (int i = start; i < end; i++) {
                float s = slice[i * cols + column] * decays[i];
                slice[i * cols + column] = s;
                correction += s * keys[i];
            }
            sums[tid] = correction;
            context.localBarrier();
            float total = 0.0f;
            for (int c = 0; c < chunks; c++) {
                total += sums[c * cols + column];
            }
            context.localBarrier();
            float o = 0.0f;
            if (active == 1) {
                float delta = (v.get(base + j) - total) * beta.get(b * heads + h);
                for (int i = start; i < end; i++) {
                    float s = slice[i * cols + column] + keys[i] * delta;
                    slice[i * cols + column] = s;
                    o += s * queries[i];
                }
            }
            sums[tid] = o;
            context.localBarrier();
            int writer = active * (1 - TornadoMath.min(1, chunk));
            if (writer == 1) {
                float result = 0.0f;
                for (int c = 0; c < chunks; c++) {
                    result += sums[c * cols + column];
                }
                out.set(base + j, result);
            }
        }
        if (active == 1) {
            for (int i = start; i < end; i++) {
                state.set(stateBase + i * headDim, slice[i * cols + column]);
            }
        }
    }

    /**
     * {@link KdaKernels#kdaSoftplusDecayAndBeta} over every row: {@code decay} flat; {@code beta}
     * read from {@code betaIn}, whose rows are {@code betaStride} wide (a GEMM's padded output),
     * and written compact into {@code betaOut}, {@code heads} per row.
     */
    public static void kdaSoftplusDecayAndBeta(
            KernelContext context,
            FloatArray decay,
            FloatArray dtBias,
            FloatArray a,
            FloatArray betaIn,
            FloatArray betaOut,
            int width,
            int headDim,
            int heads,
            int betaStride,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * width) {
            int c = id % width;
            float z = decay.get(id) + dtBias.get(c);
            float linear = TornadoMath.min(1.0f, TornadoMath.max(0.0f, z - 20.0f) * 1.0e30f);
            float soft = TornadoMath.log(1.0f + TornadoMath.exp(TornadoMath.min(z, 20.0f)));
            float sp = linear * z + (1.0f - linear) * soft;
            decay.set(id, TornadoMath.exp(a.get(c / headDim) * sp));
        }
        if (id < batch * heads) {
            int b = id / heads;
            int h = id - b * heads;
            betaOut.set(id, sigmoid(betaIn.get(b * betaStride + h)));
        }
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + TornadoMath.exp(-x));
    }
}
