package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

// @formatter:off
/**
 * The mixture-of-experts feed-forward of {@code qwen35moe}, one token at a time: four kernels per
 * layer.
 *
 * <ol>
 *   <li>{@link #routerAndSharedGate}: a warp per expert scores the normalized activation against
 *       its router row; one more warp computes the shared expert's gate, already through its
 *       sigmoid.
 *   <li>{@link #routerTopK}: one warp keeps the {@code used} best scores, highest first (a tie goes
 *       to the lower expert index), and their weights: a softmax over the kept scores, which is the
 *       full softmax renormalized over them — its denominator cancels.
 *   <li>{@link #expertsGateUpQ8_0DP4A}: a warp per hidden row of every selected expert and of the
 *       shared expert, {@code silu(gate . x) * (up . x)} as packed integer dot products against the
 *       quantized activation. The shared expert is slot {@code used}.
 *   <li>{@link #expertsDownResidualQ8_0}: a warp per output row adds every slot's down projection,
 *       weighted by its routing weight (the shared expert by its gate), into the residual stream.
 * </ol>
 *
 * <p>The routed experts are read in place from the stacked tensors the file stores: expert {@code
 * e}'s gate or up row {@code r} is row {@code e * hidden + r} of a {@code [experts * hidden][dim]}
 * Q8_0 matrix, and its down row {@code r} is row {@code e * dim + r} of {@code [experts *
 * dim][hidden]}. Every kernel's worker is a whole number of 128-lane blocks of four warps, and a
 * warp past the last row returns.
 */
// @formatter:on
public final class Qwen35MoeKernels {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    /** Lanes of every worker block here: four warps. */
    public static final int LOCAL = 128;

    /** Experts the top-k kernel can hold: eight scores a lane. */
    public static final int MAX_EXPERTS = 256;

    private Qwen35MoeKernels() {}

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    /**
     * {@code logits[e] = router[e] . x} for every expert, and {@code sharedGate[0] = sigmoid(gate .
     * x)}. Worker: {@code (experts + 1) * 32} lanes rounded up to blocks of {@link #LOCAL}.
     */
    public static void routerAndSharedGate(
            KernelContext context,
            FloatArray x,
            FloatArray router,
            FloatArray sharedGateInput,
            FloatArray logits,
            FloatArray sharedGate,
            int dim,
            int experts) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < experts) {
            int base = warp * dim;
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += router.get(base + i) * x.get(i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                logits.set(warp, sum);
            }
        } else if (warp == experts) {
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += sharedGateInput.get(i) * x.get(i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                sharedGate.set(0, 1.0f / (1.0f + TornadoMath.exp(-sum)));
            }
        }
    }

    // @formatter:off
    /**
     * The {@code used} highest of {@code experts} scores and their weights, by one warp.
     *
     * <p>The scores sit in shared memory, lane {@code l} owning {@code l, l + 32, ...}. Each round
     * every lane finds its best remaining score, a shuffle reduction takes the warp's maximum and a
     * second the lowest expert index holding it (the index travels as a float, exact below 2^24),
     * and the owner retires it. The weights are {@code exp(s - s_max)} normalized over the kept
     * scores. Worker: one block of 32 lanes.
     */
    // @formatter:on
    public static void routerTopK(
            KernelContext context,
            FloatArray logits,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used) {
        int lane = context.localIdx;
        float[] scores = context.allocateFloatLocalArray(MAX_EXPERTS);
        if (lane < 32) {
            for (int e = lane; e < experts; e += 32) {
                scores[e] = logits.get(e);
            }
        }
        context.localBarrier();
        float top = 0.0f;
        float total = 0.0f;
        float mine = 0.0f;
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            float bestIndex = 1.0e9f;
            for (int e = lane; e < experts; e += 32) {
                float s = scores[e];
                if (s > best) {
                    best = s;
                    bestIndex = e;
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
            if (k == 0) {
                top = max;
            }
            float w = TornadoMath.exp(max - top);
            total += w;
            if (lane == k) {
                mine = w;
            }
            if ((chosen & 31) == lane) {
                scores[chosen] = Float.NEGATIVE_INFINITY;
            }
            if (lane == 0) {
                ids.set(k, chosen);
            }
        }
        if (lane < used) {
            weights.set(lane, mine / total);
        }
    }

    /** One Q8_0 block's integer dot product with the matching quantized activation block. */
    private static int blockDot(ByteArray w, int blockByteOffset, IntArray xQuants, int quantBase) {
        int dot = 0;
        for (int g = 0; g < QK / 4; g++) {
            int offset = blockByteOffset + 2 + g * 4;
            int packed =
                    (w.getHalfFloat(offset).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(offset + 2).getHalfFloatValue() & 0xFFFF) << 16);
            dot = QuantizationUtils.dp4a_packed(packed, xQuants.get(quantBase + g), dot);
        }
        return dot;
    }

    /** This lane's share of row {@code row}'s dot product with the quantized activation. */
    private static float rowPartial(
            ByteArray w,
            int row,
            int blocksPerRow,
            IntArray xQuants,
            FloatArray xScales,
            int lane) {
        float partial = 0.0f;
        for (int block = lane; block < blocksPerRow; block += 32) {
            int offset = (row * blocksPerRow + block) * BLOCK_BYTES;
            partial +=
                    w.getHalfFloat(offset).getFloat32()
                            * xScales.get(block)
                            * blockDot(w, offset, xQuants, block * (QK / 4));
        }
        return partial;
    }

    /**
     * {@code hidden[slot * hiddenDim + r] = silu(gate_e[r] . x) * (up_e[r] . x)} for each selected
     * expert {@code e = ids[slot]}, and slot {@code used} for the shared expert's {@code
     * sharedHiddenDim} rows. Worker: {@code (used * hiddenDim + sharedHiddenDim) * 32} lanes in
     * blocks of {@link #LOCAL}.
     */
    public static void expertsGateUpQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray ids,
            ByteArray gateExperts,
            ByteArray upExperts,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int routedRows = used * hiddenDim;
        int blocksPerRow = dim / QK;
        // Each branch reduces and stores its own row: a value merged from an if/else is
        // miscompiled by the CUDA backend (the merge re-assigns the else branch's value).
        if (warp < routedRows) {
            int slot = warp / hiddenDim;
            int row = ids.get(slot) * hiddenDim + (warp - slot * hiddenDim);
            float gate =
                    warpSum(
                            context,
                            rowPartial(gateExperts, row, blocksPerRow, xQuants, xScales, lane));
            float up =
                    warpSum(
                            context,
                            rowPartial(upExperts, row, blocksPerRow, xQuants, xScales, lane));
            if (lane == 0) {
                hidden.set(warp, gate / (1.0f + TornadoMath.exp(-gate)) * up);
            }
        }
        if (warp >= routedRows && warp < routedRows + sharedHiddenDim) {
            int row = warp - routedRows;
            float gate =
                    warpSum(
                            context,
                            rowPartial(sharedGate, row, blocksPerRow, xQuants, xScales, lane));
            float up =
                    warpSum(
                            context,
                            rowPartial(sharedUp, row, blocksPerRow, xQuants, xScales, lane));
            if (lane == 0) {
                hidden.set(warp, gate / (1.0f + TornadoMath.exp(-gate)) * up);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #expertsDownResidualQ8_0} over the quantized hidden activations: every slot's hidden
     * rows quantized per 32-block, and each (slot, block) of an output row a packed integer dot
     * product. Lane {@code l} takes pairs {@code l, l + 32, ...}: the routed slots' first, then the
     * shared expert's, in two loops so no value is merged across a branch. Worker: {@code dim * 32}
     * lanes in blocks of {@link #LOCAL}.
     */
    // @formatter:on
    public static void expertsDownResidualQ8_0DP4A(
            KernelContext context,
            IntArray hQuants,
            FloatArray hScales,
            IntArray ids,
            FloatArray weights,
            FloatArray sharedGate,
            ByteArray downExperts,
            ByteArray sharedDown,
            FloatArray residual,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (row < dim) {
            int perSlot = hiddenDim / QK;
            int routedPairs = used * perSlot;
            float sum = 0.0f;
            for (int p = lane; p < routedPairs; p += 32) {
                int slot = p / perSlot;
                int block = p - slot * perSlot;
                int offset = ((ids.get(slot) * dim + row) * perSlot + block) * BLOCK_BYTES;
                sum +=
                        weights.get(slot)
                                * downExperts.getHalfFloat(offset).getFloat32()
                                * hScales.get(p)
                                * blockDot(downExperts, offset, hQuants, p * (QK / 4));
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
            sum = warpSum(context, sum + sharedGate.get(0) * shared);
            if (lane == 0) {
                residual.set(row, residual.get(row) + sum);
            }
        }
    }

    /** This lane's share of one down row's dot product with a slot's hidden activation. */
    private static float downPartial(
            ByteArray w, int row, int hiddenDim, FloatArray hidden, int hiddenBase, int lane) {
        int blocksPerRow = hiddenDim / QK;
        float partial = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int offset = (row * blocksPerRow + block) * BLOCK_BYTES;
            float scale = w.getHalfFloat(offset).getFloat32();
            partial +=
                    scale * w.get(offset + 2 + lane) * hidden.get(hiddenBase + block * QK + lane);
        }
        return partial;
    }

    /**
     * {@code residual[r] += sum over slots of weight[slot] * (down_e[r] . hidden[slot]) +
     * sharedGate * (sharedDown[r] . hidden[used])}. Worker: {@code dim * 32} lanes in blocks of
     * {@link #LOCAL}.
     */
    public static void expertsDownResidualQ8_0(
            KernelContext context,
            FloatArray hidden,
            IntArray ids,
            FloatArray weights,
            FloatArray sharedGate,
            ByteArray downExperts,
            ByteArray sharedDown,
            FloatArray residual,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (row < dim) {
            float sum = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                int expertRow = ids.get(slot) * dim + row;
                sum +=
                        weights.get(slot)
                                * downPartial(
                                        downExperts,
                                        expertRow,
                                        hiddenDim,
                                        hidden,
                                        slot * hiddenDim,
                                        lane);
            }
            if (sharedHiddenDim > 0) {
                sum +=
                        sharedGate.get(0)
                                * downPartial(
                                        sharedDown,
                                        row,
                                        sharedHiddenDim,
                                        hidden,
                                        used * hiddenDim,
                                        lane);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                residual.set(row, residual.get(row) + sum);
            }
        }
    }
}
