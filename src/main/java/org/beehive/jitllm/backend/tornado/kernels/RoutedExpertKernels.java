package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Decode kernels of a mixture of experts with sigmoid routing and shared experts over Q8_0 or Q4_0
 * routed experts (GLM-5.3-Flash, Kimi-Linear): the top-k routing with a selection-only bias, the
 * selected and shared experts' (optionally clamped) SwiGLU, and their weighted down projection.
 *
 * <p>Workgroup reductions assume a power-of-two local size no larger than {@link #GROUP}.
 */
public final class RoutedExpertKernels {

    /** The widest workgroup the reductions here are written for. */
    public static final int GROUP = 256;

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;
    private static final int Q4_BLOCK_BYTES = 18;

    /** The most experts {@link #routerTopKSigmoid} routes over. */
    public static final int MAX_EXPERTS = 512;

    private RoutedExpertKernels() {}

    // @formatter:off
    /**
     * Sigmoid routing for up to {@link #MAX_EXPERTS} experts, in one workgroup: the top {@code
     * used} of {@code sigmoid(logit) + bias}, weighted by their sigmoids, normalized to sum to one
     * when {@code normalize} is 1 and scaled. Ties go to the lower expert.
     */
    // @formatter:on
    public static void routerTopKSigmoid(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used,
            int normalize,
            float scale) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] select = context.allocateFloatLocalArray(MAX_EXPERTS);
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        for (int e = tid; e < experts; e += local) {
            float p = 1.0f / (1.0f + TornadoMath.exp(-logits.get(e)));
            select[e] = p + bias.get(e);
        }
        context.localBarrier();
        for (int k = 0; k < used; k++) {
            float b = Float.NEGATIVE_INFINITY;
            int index = experts;
            for (int e = tid; e < experts; e += local) {
                float s = select[e];
                int better = s > b ? 1 : 0;
                b = better == 1 ? s : b;
                index = better == 1 ? e : index;
            }
            best[tid] = b;
            bestIndex[tid] = index;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float other = best[tid + stride];
                    int otherIndex = bestIndex[tid + stride];
                    int take =
                            other > best[tid] || (other == best[tid] && otherIndex < bestIndex[tid])
                                    ? 1
                                    : 0;
                    best[tid] = take == 1 ? other : best[tid];
                    bestIndex[tid] = take == 1 ? otherIndex : bestIndex[tid];
                }
                context.localBarrier();
            }
            if (tid == 0) {
                int chosen = bestIndex[0];
                ids.set(k, chosen);
                weights.set(k, 1.0f / (1.0f + TornadoMath.exp(-logits.get(chosen))));
                select[chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        if (tid == 0) {
            float total = 0.0f;
            for (int k = 0; k < used; k++) {
                total += weights.get(k);
            }
            // A blend rather than a branch: a value merged from an if/else is miscompiled by the
            // CUDA backend.
            float n = normalize;
            float norm = n * (total + 1e-20f) + (1.0f - n);
            for (int k = 0; k < used; k++) {
                weights.set(k, weights.get(k) / norm * scale);
            }
        }
    }

    // @formatter:off
    /**
     * The selected experts' and the shared expert's clamped SwiGLU over the quantized input, a warp
     * per hidden row: {@code hidden[slot * hiddenDim + r]} for each slot, then the shared expert's
     * rows. The routed experts are held in two stacks, experts {@code [0, split)} in the first; a
     * model with one stack passes it twice.
     *
     * <p>Each branch reduces and stores its own row: a value merged from an if/else is miscompiled
     * by the CUDA backend. Worker: {@code (used * hiddenDim + sharedHiddenDim) * 32} lanes.
     */
    // @formatter:on
    public static void expertsGateUpClamped(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray ids,
            ByteArray gate0,
            ByteArray up0,
            ByteArray gate1,
            ByteArray up1,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split,
            float limit) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int routedRows = used * hiddenDim;
        int blocks = dim / QK;
        if (warp < routedRows) {
            int slot = warp / hiddenDim;
            int r = warp - slot * hiddenDim;
            int e = ids.get(slot);
            if (e < split) {
                int row = e * hiddenDim + r;
                float g = warpSum(context, rowPartial(gate0, row, blocks, xQuants, xScales, lane));
                float u = warpSum(context, rowPartial(up0, row, blocks, xQuants, xScales, lane));
                if (lane == 0) {
                    hidden.set(warp, clampedSilu(g, u, limit));
                }
            } else {
                int row = (e - split) * hiddenDim + r;
                float g = warpSum(context, rowPartial(gate1, row, blocks, xQuants, xScales, lane));
                float u = warpSum(context, rowPartial(up1, row, blocks, xQuants, xScales, lane));
                if (lane == 0) {
                    hidden.set(warp, clampedSilu(g, u, limit));
                }
            }
        }
        if (warp >= routedRows && warp < routedRows + sharedHiddenDim) {
            int row = warp - routedRows;
            float g = warpSum(context, rowPartial(sharedGate, row, blocks, xQuants, xScales, lane));
            float u = warpSum(context, rowPartial(sharedUp, row, blocks, xQuants, xScales, lane));
            if (lane == 0) {
                hidden.set(warp, clampedSilu(g, u, limit));
            }
        }
    }

    // @formatter:off
    /**
     * {@code out[r] = sum_slot weight[slot] * (down_e[r] . hidden[slot]) + sharedDown[r] .
     * hidden[used]}, a warp per output row, over the hidden activations quantized per block.
     *
     * <p>Only the stack that holds a slot's expert is read. Each branch adds to the lane's sum in
     * local memory rather than to a value merged after the branch, which the CUDA backend
     * miscompiles, and both stacks' expert indices are clamped into range, since the backend may
     * form either address. A warp's 32 blocks of one step lie in one slot when {@code hiddenDim /
     * 32} is a multiple of 32, so the branch is uniform across the warp. Worker: {@code dim * 32}
     * lanes, local at most {@link #GROUP}.
     */
    // @formatter:on
    public static void expertsDown(
            KernelContext context,
            IntArray hQuants,
            FloatArray hScales,
            IntArray ids,
            FloatArray weights,
            ByteArray down0,
            ByteArray down1,
            ByteArray sharedDown,
            FloatArray out,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int tid = context.localIdx;
        float[] sums = context.allocateFloatLocalArray(GROUP);
        sums[tid] = 0.0f;
        if (row < dim) {
            int perSlot = hiddenDim / QK;
            int routedPairs = used * perSlot;
            for (int p = lane; p < routedPairs; p += 32) {
                int slot = p / perSlot;
                int block = p - slot * perSlot;
                int e = ids.get(slot);
                float w = weights.get(slot) * hScales.get(p);
                int e0 = TornadoMath.min(e, split - 1);
                int e1 = TornadoMath.max(0, e - split);
                if (e < split) {
                    int offset = ((e0 * dim + row) * perSlot + block) * BLOCK_BYTES;
                    sums[tid] +=
                            w
                                    * down0.getHalfFloat(offset).getFloat32()
                                    * blockDot(down0, offset, hQuants, p * (QK / 4));
                }
                if (e >= split) {
                    int offset = ((e1 * dim + row) * perSlot + block) * BLOCK_BYTES;
                    sums[tid] +=
                            w
                                    * down1.getHalfFloat(offset).getFloat32()
                                    * blockDot(down1, offset, hQuants, p * (QK / 4));
                }
            }
            int sharedBlocks = sharedHiddenDim / QK;
            float shared = 0.0f;
            for (int b = lane; b < sharedBlocks; b += 32) {
                int offset = (row * sharedBlocks + b) * BLOCK_BYTES;
                int p = routedPairs + b;
                shared +=
                        sharedDown.getHalfFloat(offset).getFloat32()
                                * hScales.get(p)
                                * blockDot(sharedDown, offset, hQuants, p * (QK / 4));
            }
            float sum = warpSum(context, sums[tid] + shared);
            if (lane == 0) {
                out.set(row, sum);
            }
        }
    }

    /**
     * {@link #expertsGateUpClamped} with the routed experts in Q4_0 and the shared expert in Q8_0.
     */
    public static void expertsGateUpClampedQ4_0(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray ids,
            ByteArray gate0,
            ByteArray up0,
            ByteArray gate1,
            ByteArray up1,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split,
            float limit) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int routedRows = used * hiddenDim;
        int blocks = dim / QK;
        if (warp < routedRows) {
            int slot = warp / hiddenDim;
            int r = warp - slot * hiddenDim;
            int e = ids.get(slot);
            if (e < split) {
                int row = e * hiddenDim + r;
                float g =
                        warpSum(
                                context,
                                rowPartialQ4_0(gate0, row, blocks, xQuants, xScales, lane));
                float u =
                        warpSum(context, rowPartialQ4_0(up0, row, blocks, xQuants, xScales, lane));
                if (lane == 0) {
                    hidden.set(warp, clampedSilu(g, u, limit));
                }
            } else {
                int row = (e - split) * hiddenDim + r;
                float g =
                        warpSum(
                                context,
                                rowPartialQ4_0(gate1, row, blocks, xQuants, xScales, lane));
                float u =
                        warpSum(context, rowPartialQ4_0(up1, row, blocks, xQuants, xScales, lane));
                if (lane == 0) {
                    hidden.set(warp, clampedSilu(g, u, limit));
                }
            }
        }
        if (warp >= routedRows && warp < routedRows + sharedHiddenDim) {
            int row = warp - routedRows;
            float g = warpSum(context, rowPartial(sharedGate, row, blocks, xQuants, xScales, lane));
            float u = warpSum(context, rowPartial(sharedUp, row, blocks, xQuants, xScales, lane));
            if (lane == 0) {
                hidden.set(warp, clampedSilu(g, u, limit));
            }
        }
    }

    /** {@link #expertsDown} with the routed experts in Q4_0 and the shared expert in Q8_0. */
    public static void expertsDownQ4_0(
            KernelContext context,
            IntArray hQuants,
            FloatArray hScales,
            IntArray ids,
            FloatArray weights,
            ByteArray down0,
            ByteArray down1,
            ByteArray sharedDown,
            FloatArray out,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int tid = context.localIdx;
        float[] sums = context.allocateFloatLocalArray(GROUP);
        sums[tid] = 0.0f;
        if (row < dim) {
            int perSlot = hiddenDim / QK;
            int routedPairs = used * perSlot;
            for (int p = lane; p < routedPairs; p += 32) {
                int slot = p / perSlot;
                int block = p - slot * perSlot;
                int e = ids.get(slot);
                float w = weights.get(slot) * hScales.get(p);
                int e0 = TornadoMath.min(e, split - 1);
                int e1 = TornadoMath.max(0, e - split);
                if (e < split) {
                    int offset = ((e0 * dim + row) * perSlot + block) * Q4_BLOCK_BYTES;
                    sums[tid] +=
                            w
                                    * down0.getHalfFloat(offset).getFloat32()
                                    * blockDotQ4_0(down0, offset, hQuants, p * (QK / 4));
                }
                if (e >= split) {
                    int offset = ((e1 * dim + row) * perSlot + block) * Q4_BLOCK_BYTES;
                    sums[tid] +=
                            w
                                    * down1.getHalfFloat(offset).getFloat32()
                                    * blockDotQ4_0(down1, offset, hQuants, p * (QK / 4));
                }
            }
            int sharedBlocks = sharedHiddenDim / QK;
            float shared = 0.0f;
            for (int b = lane; b < sharedBlocks; b += 32) {
                int offset = (row * sharedBlocks + b) * BLOCK_BYTES;
                int p = routedPairs + b;
                shared +=
                        sharedDown.getHalfFloat(offset).getFloat32()
                                * hScales.get(p)
                                * blockDot(sharedDown, offset, hQuants, p * (QK / 4));
            }
            float sum = warpSum(context, sums[tid] + shared);
            if (lane == 0) {
                out.set(row, sum);
            }
        }
    }

    /** {@link #rowPartial} over a Q4_0 weight. */
    private static float rowPartialQ4_0(
            ByteArray w, int row, int blocks, IntArray xQuants, FloatArray xScales, int lane) {
        float partial = 0.0f;
        for (int b = lane; b < blocks; b += 32) {
            int offset = (row * blocks + b) * Q4_BLOCK_BYTES;
            partial +=
                    w.getHalfFloat(offset).getFloat32()
                            * xScales.get(b)
                            * blockDotQ4_0(w, offset, xQuants, b * (QK / 4));
        }
        return partial;
    }

    /**
     * The int dot product of the Q4_0 block at byte {@code offset} with eight words of int8
     * activations: byte {@code j} holds weight {@code j} in its low nibble and weight {@code 16 +
     * j} in its high nibble, both stored plus 8, so the sum of the activations times 8 comes off.
     */
    static int blockDotQ4_0(ByteArray w, int offset, IntArray xQuants, int quantBase) {
        int dot = 0;
        int sum = 0;
        for (int g = 0; g < 4; g++) {
            int o = offset + 2 + g * 4;
            int packed =
                    (w.getHalfFloat(o).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(o + 2).getHalfFloatValue() & 0xFFFF) << 16);
            int lo = xQuants.get(quantBase + g);
            int hi = xQuants.get(quantBase + 4 + g);
            dot = QuantizationUtils.dp4a_packed(packed & 0x0F0F0F0F, lo, dot);
            dot = QuantizationUtils.dp4a_packed((packed >>> 4) & 0x0F0F0F0F, hi, dot);
            sum = QuantizationUtils.dp4a_packed(0x01010101, lo, sum);
            sum = QuantizationUtils.dp4a_packed(0x01010101, hi, sum);
        }
        return dot - 8 * sum;
    }

    private static float clampedSilu(float gate, float up, float limit) {
        float g = TornadoMath.min(gate, limit);
        float u = TornadoMath.max(-limit, TornadoMath.min(up, limit));
        return g / (1.0f + TornadoMath.exp(-g)) * u;
    }

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    private static int blockDot(ByteArray w, int offset, IntArray xQuants, int quantBase) {
        int dot = 0;
        for (int g = 0; g < QK / 4; g++) {
            int o = offset + 2 + g * 4;
            int packed =
                    (w.getHalfFloat(o).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(o + 2).getHalfFloatValue() & 0xFFFF) << 16);
            dot = QuantizationUtils.dp4a_packed(packed, xQuants.get(quantBase + g), dot);
        }
        return dot;
    }

    private static float rowPartial(
            ByteArray w, int row, int blocks, IntArray xQuants, FloatArray xScales, int lane) {
        float partial = 0.0f;
        for (int b = lane; b < blocks; b += 32) {
            int offset = (row * blocks + b) * BLOCK_BYTES;
            partial +=
                    w.getHalfFloat(offset).getFloat32()
                            * xScales.get(b)
                            * blockDot(w, offset, xQuants, b * (QK / 4));
        }
        return partial;
    }

    /** {@code x[i] += delta[i]} for the first {@code n} values: the experts into the residual. */
    public static void addInto(KernelContext context, FloatArray x, FloatArray delta, int n) {
        int i = context.globalIdx;
        if (i < n) {
            x.set(i, x.get(i) + delta.get(i));
        }
    }
}
