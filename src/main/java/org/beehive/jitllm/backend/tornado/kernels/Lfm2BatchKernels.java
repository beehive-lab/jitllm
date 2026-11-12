package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@code lfm2moe} kernels of a batched prefill that the shared batch kernels do not cover: the
 * short convolution over a chunk, the router and its sigmoid selection per token, and the mixture's
 * SwiGLU and weighted combine over the assignments {@code Qwen35MoeBatchKernels.groupByExpert}
 * sorted by expert.
 *
 * <p>{@code batchInfo} is the plan's chunk holder: {@code [0]} the first position, {@code [1]} the
 * chunk's real rows. Rows past it are padding and are not read or written.
 *
 * <p>Every per-token computation is the single-token kernel's arithmetic in the same order, so a
 * chunk gives the numbers the same tokens give one at a time: {@link #shortConvBatch} is {@link
 * Lfm2Kernels#shortConv} applied row by row, {@link #selectExpertsSigmoidBatch} is {@link
 * Lfm2Kernels#selectExpertsSigmoidSimdgroup} per token.
 */
// @formatter:on
public final class Lfm2BatchKernels {

    private Lfm2BatchKernels() {}

    /**
     * {@link Lfm2Kernels#shortConv} over the chunk's rows in order, one lane per channel: the
     * window carries from row to row and is left where decode continues. {@code bcx} rows are
     * {@code 3 * dim} wide, {@code out} rows {@code dim}.
     */
    public static void shortConvBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray bcx,
            FloatArray taps,
            FloatArray state,
            FloatArray out,
            int dim,
            int kernel,
            int stateOffset) {
        int c = context.globalIdx;
        if (c >= dim) {
            return;
        }
        int rows = batchInfo.get(1);
        int history = kernel - 1;
        int wBase = c * kernel;
        int hBase = stateOffset + c * history;
        for (int r = 0; r < rows; r++) {
            int in = r * 3 * dim;
            float bx = bcx.get(in + c) * bcx.get(in + 2 * dim + c);
            float sum = 0.0f;
            for (int t = 0; t < history; t++) {
                sum += taps.get(wBase + t) * state.get(hBase + t);
            }
            sum += taps.get(wBase + history) * bx;
            for (int t = 0; t + 1 < history; t++) {
                state.set(hBase + t, state.get(hBase + t + 1));
            }
            state.set(hBase + history - 1, bx);
            out.set(r * dim + c, bcx.get(in + dim + c) * sum);
        }
    }

    /** Router rows per workgroup of {@link #routerBatch}: one SIMD group each. */
    public static final int ROUTER_SIMDGROUPS = 4;

    /**
     * {@code logits[t·E + e] = router[e] · x[t]} for the chunk's rows, the F32 router as the
     * single-token {@code matrixVectorGeneric} reads it ({@code [E, dim]}); one SIMD group per
     * (row, expert), {@link #ROUTER_SIMDGROUPS} per workgroup of {@code 32 * ROUTER_SIMDGROUPS}.
     */
    public static void routerBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray x,
            FloatArray router,
            FloatArray logits,
            int dim,
            int experts) {
        int lane = context.localIdx & 31;
        int job = context.groupIdx * ROUTER_SIMDGROUPS + (context.localIdx >> 5);
        int row = job / experts;
        if (row >= batchInfo.get(1)) {
            return;
        }
        int e = job - row * experts;
        int xBase = row * dim;
        int wBase = e * dim;
        float sum = 0.0f;
        for (int i = lane; i < dim; i += 32) {
            sum += router.get(wBase + i) * x.get(xBase + i);
        }
        sum += context.simdShuffleDown(sum, 16);
        sum += context.simdShuffleDown(sum, 8);
        sum += context.simdShuffleDown(sum, 4);
        sum += context.simdShuffleDown(sum, 2);
        sum += context.simdShuffleDown(sum, 1);
        if (lane == 0) {
            logits.set(row * experts + e, sum);
        }
    }

    /** Tokens per workgroup of {@link #selectExpertsSigmoidBatch}: one SIMD group each. */
    public static final int SELECT_SIMDGROUPS = 4;

    /**
     * {@link Lfm2Kernels#selectExpertsSigmoidSimdgroup} for every row of the chunk: {@code ids[t·k
     * + j]} and {@code weights[t·k + j]}, the weights renormalized per token. One SIMD group per
     * token, {@link #SELECT_SIMDGROUPS} per workgroup; {@code experts} at most {@link
     * Lfm2Kernels#MAX_EXPERTS}.
     */
    public static void selectExpertsSigmoidBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray logits,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used) {
        int lane = context.localIdx & 31;
        int sg = context.localIdx >> 5;
        int token = context.groupIdx * SELECT_SIMDGROUPS + sg;
        float[] scores =
                context.allocateFloatLocalArray(SELECT_SIMDGROUPS * Lfm2Kernels.MAX_EXPERTS);
        float[] probs =
                context.allocateFloatLocalArray(SELECT_SIMDGROUPS * Lfm2Kernels.MAX_EXPERTS);
        int base = sg * Lfm2Kernels.MAX_EXPERTS;
        boolean active = token < batchInfo.get(1);
        if (active) {
            for (int e = lane; e < experts; e += 32) {
                float p = 1.0f / (1.0f + TornadoMath.exp(-logits.get(token * experts + e)));
                probs[base + e] = p;
                scores[base + e] = p + bias.get(e);
            }
        }
        context.localBarrier();
        float total = 0.0f;
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            float bestIndex = 1.0e9f;
            if (active) {
                for (int e = lane; e < experts; e += 32) {
                    float score = scores[base + e];
                    if (score > best) {
                        best = score;
                        bestIndex = e;
                    }
                }
            }
            float max = best;
            max = TornadoMath.max(max, context.simdShuffleDown(max, 16));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 8));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 4));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 2));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 1));
            max = context.simdBroadcastFirst(max);
            float index = best == max ? bestIndex : 1.0e9f;
            index = TornadoMath.min(index, context.simdShuffleDown(index, 16));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 8));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 4));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 2));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 1));
            int chosen = (int) context.simdBroadcastFirst(index);
            if (active) {
                float w = probs[base + chosen];
                total += w;
                if (lane == 0) {
                    ids.set(token * used + k, chosen);
                    weights.set(token * used + k, w);
                }
            }
            context.localBarrier();
            if (active && lane == chosen % 32) {
                scores[base + chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        if (active && lane == 0) {
            if (total < 6.103515625e-5f) {
                total = 6.103515625e-5f;
            }
            for (int j = 0; j < used; j++) {
                weights.set(token * used + j, weights.get(token * used + j) / total);
            }
        }
    }

    /**
     * {@code out[i] = 0} over the chunk's {@code rows · width} elements: the store a tiled batch
     * GEMM that only accumulates is turned into, where no overwriting form exists.
     */
    public static void zeroRows(
            KernelContext context, IntArray batchInfo, FloatArray out, int width) {
        int i = context.globalIdx;
        if (i >= batchInfo.get(1) * width) {
            return;
        }
        out.set(i, 0.0f);
    }

    /**
     * {@code gate[a·H + i] = silu(gate[a·H + i]) · up[a·H + i]} over the chunk's {@code rows ·
     * used} assignments, in sorted order: a lane per element.
     */
    public static void siluMultiplyAssignments(
            KernelContext context,
            IntArray batchInfo,
            FloatArray gate,
            FloatArray up,
            int hidden,
            int used) {
        int i = context.globalIdx;
        if (i >= batchInfo.get(1) * used * hidden) {
            return;
        }
        float g = gate.get(i);
        gate.set(i, g / (1.0f + TornadoMath.exp(-g)) * up.get(i));
    }

    /**
     * {@code x[t·dim + c] += Σ_s weights[t·k + s] · out[position[t·k + s]·dim + c]}, the slots
     * summed in selection order as the single-token down kernels sum them: a lane per (row,
     * column).
     */
    public static void combineBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray out,
            IntArray position,
            FloatArray weights,
            FloatArray x,
            int dim,
            int used) {
        int i = context.globalIdx;
        if (i >= batchInfo.get(1) * dim) {
            return;
        }
        int t = i / dim;
        int c = i - t * dim;
        float sum = 0.0f;
        for (int s = 0; s < used; s++) {
            int a = t * used + s;
            sum += weights.get(a) * out.get(position.get(a) * dim + c);
        }
        x.set(i, x.get(i) + sum);
    }
}
