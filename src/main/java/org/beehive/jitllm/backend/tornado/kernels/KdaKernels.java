package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Decode kernels of Kimi delta attention (KDA), shared by every family with KDA layers
 * (GLM-5.3-Flash, Kimi-Linear): the per-channel decay in either family's form, the L2 norms of the
 * query and key, the delta rule over each head's state, and the gated output norm.
 *
 * <p>Workgroup reductions assume a power-of-two local size no larger than {@link #GROUP}.
 */
public final class KdaKernels {

    /** The widest workgroup the reductions here are written for. */
    public static final int GROUP = 256;

    /** The state columns one {@link #kdaDeltaRule} workgroup takes. */
    public static final int DELTA_COLUMNS = 32;

    /** The chunks {@link #kdaDeltaRule} splits each state column's rows into. */
    public static final int DELTA_CHUNKS = 8;

    private KdaKernels() {}

    // @formatter:off
    /**
     * The per-channel decay {@code exp(lower * sigmoid(-a[h] * (raw + dtBias)))}, in place on
     * {@code decay}, where {@code a = -exp(A_log)} as llama.cpp stores it, and {@code beta =
     * sigmoid(beta)} in place for the first {@code heads} lanes.
     */
    // @formatter:on
    public static void kdaDecayAndBeta(
            KernelContext context,
            FloatArray decay,
            FloatArray dtBias,
            FloatArray a,
            FloatArray beta,
            int width,
            int headDim,
            int heads,
            float lowerBound) {
        int c = context.globalIdx;
        if (c < width) {
            float rate = -a.get(c / headDim);
            float g = rate * (decay.get(c) + dtBias.get(c));
            decay.set(c, TornadoMath.exp(lowerBound / (1.0f + TornadoMath.exp(-g))));
        }
        if (c < heads) {
            beta.set(c, 1.0f / (1.0f + TornadoMath.exp(-beta.get(c))));
        }
    }

    /**
     * {@code x = x * scale / sqrt(sum(x²) + eps)} per head, a workgroup per head; the norm FLA
     * uses.
     */
    public static void kdaL2Norm(
            KernelContext context,
            FloatArray values,
            int heads,
            int headDim,
            float scale,
            float eps) {
        int h = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        float ss = 0.0f;
        if (h < heads) {
            for (int i = tid; i < headDim; i += local) {
                float v = values.get(h * headDim + i);
                ss += v * v;
            }
        }
        reduce[tid] = ss;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        if (h < heads) {
            float inv = scale / TornadoMath.sqrt(reduce[0] + eps);
            for (int i = tid; i < headDim; i += local) {
                values.set(h * headDim + i, values.get(h * headDim + i) * inv);
            }
        }
    }

    // @formatter:off
    /**
     * The KDA delta rule over each head's state {@code S}, the key channel {@code i} as the row:
     * {@code S[i][j] *= decay[i]}, {@code d = (v[j] - sum_i S[i][j] k[i]) * beta}, {@code S[i][j]
     * += k[i] d}, {@code out[j] = sum_i S[i][j] q[i]}.
     *
     * <p>A workgroup takes {@link #DELTA_COLUMNS} columns of one head, each column's rows split
     * into {@link #DELTA_CHUNKS} chunks, a lane per (chunk, column): adjacent lanes read adjacent
     * columns. The chunks' sums meet in local memory, added in chunk order. The second pass decays
     * the state again from what it read rather than storing it twice. The state of this layer
     * starts at {@code stateOffset}, row-major per head.
     *
     * <p>Grid: {@code heads * ceil(headDim / DELTA_COLUMNS)} workgroups of {@code DELTA_COLUMNS *
     * DELTA_CHUNKS} lanes.
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
            int heads,
            int headDim,
            int stateOffset) {
        int tid = context.localIdx;
        int column = tid % DELTA_COLUMNS;
        int chunk = tid / DELTA_COLUMNS;
        int tiles = (headDim + DELTA_COLUMNS - 1) / DELTA_COLUMNS;
        int h = context.groupIdx / tiles;
        int j = (context.groupIdx - h * tiles) * DELTA_COLUMNS + column;
        float[] sums = context.allocateFloatLocalArray(DELTA_COLUMNS * DELTA_CHUNKS);
        int rows = (headDim + DELTA_CHUNKS - 1) / DELTA_CHUNKS;
        int start = chunk * rows;
        int end = TornadoMath.min(headDim, start + rows);
        int base = h * headDim;
        int stateBase = stateOffset + h * headDim * headDim + j;
        int active =
                TornadoMath.min(1, TornadoMath.max(0, heads - h))
                        * TornadoMath.min(1, TornadoMath.max(0, headDim - j));
        float correction = 0.0f;
        if (active == 1) {
            for (int i = start; i < end; i++) {
                float s = state.get(stateBase + i * headDim) * decay.get(base + i);
                correction += s * k.get(base + i);
            }
        }
        sums[tid] = correction;
        context.localBarrier();
        float total = 0.0f;
        for (int c = 0; c < DELTA_CHUNKS; c++) {
            total += sums[c * DELTA_COLUMNS + column];
        }
        context.localBarrier();
        float o = 0.0f;
        if (active == 1) {
            float delta = (v.get(base + j) - total) * beta.get(h);
            for (int i = start; i < end; i++) {
                float s =
                        state.get(stateBase + i * headDim) * decay.get(base + i)
                                + k.get(base + i) * delta;
                state.set(stateBase + i * headDim, s);
                o += s * q.get(base + i);
            }
        }
        sums[tid] = o;
        context.localBarrier();
        int writer = active * (1 - TornadoMath.min(1, chunk));
        if (writer == 1) {
            float result = 0.0f;
            for (int c = 0; c < DELTA_CHUNKS; c++) {
                result += sums[c * DELTA_COLUMNS + column];
            }
            out.set(base + j, result);
        }
    }

    /** {@code rms_norm(x, weight) * sigmoid(gate)} per head, in place, a workgroup per head. */
    public static void kdaGatedNorm(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps) {
        int h = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        float ss = 0.0f;
        if (h < heads) {
            for (int i = tid; i < headDim; i += local) {
                float v = values.get(h * headDim + i);
                ss += v * v;
            }
        }
        reduce[tid] = ss;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        if (h < heads) {
            float inv = 1.0f / TornadoMath.sqrt(reduce[0] / headDim + eps);
            for (int i = tid; i < headDim; i += local) {
                int e = h * headDim + i;
                float g = 1.0f / (1.0f + TornadoMath.exp(-gate.get(e)));
                values.set(e, weight.get(i) * inv * values.get(e) * g);
            }
        }
    }

    // @formatter:off
    /**
     * The per-channel decay {@code exp(a[h] * softplus(raw + dtBias))}, in place on {@code decay},
     * where {@code a = -exp(A_log)} as llama.cpp stores it (Kimi-Linear), and {@code beta =
     * sigmoid(beta)} in place for the first {@code heads} lanes. {@code softplus(z)} is {@code z}
     * above 20, as ggml computes it.
     */
    // @formatter:on
    public static void kdaSoftplusDecayAndBeta(
            KernelContext context,
            FloatArray decay,
            FloatArray dtBias,
            FloatArray a,
            FloatArray beta,
            int width,
            int headDim,
            int heads) {
        int c = context.globalIdx;
        if (c < width) {
            float z = decay.get(c) + dtBias.get(c);
            float linear = TornadoMath.min(1.0f, TornadoMath.max(0.0f, z - 20.0f) * 1.0e30f);
            float soft = TornadoMath.log(1.0f + TornadoMath.exp(TornadoMath.min(z, 20.0f)));
            float sp = linear * z + (1.0f - linear) * soft;
            decay.set(c, TornadoMath.exp(a.get(c / headDim) * sp));
        }
        if (c < heads) {
            beta.set(c, 1.0f / (1.0f + TornadoMath.exp(-beta.get(c))));
        }
    }
}
