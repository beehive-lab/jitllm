package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Device kernels for the {@code lfm2moe} architecture's single-token decode.
 *
 * <p>Three things no other family's kernels do:
 *
 * <ul>
 *   <li><b>The gated short convolution</b>: {@code y = C ⊙ conv(B ⊙ x)} per channel, with the
 *       window advanced in place. One lane per channel, the same arithmetic as the host's {@code
 *       CpuOperations.causalConv1d} with the two products around it.
 *   <li><b>The sigmoid router's selection</b>: top-k on score plus selection bias, weighted by the
 *       unbiased score, renormalized — on the device, so a token's experts are chosen without a
 *       round trip to the host.
 *   <li><b>Expert projections indexed by that selection.</b> The stacked expert tensors hold every
 *       expert; a workgroup reads the expert id the router wrote and addresses that expert's rows,
 *       so only the selected experts' weights are read. One kernel per representation, each
 *       decoding its blocks with that representation's own decoder — the same helpers the dense
 *       kernels use — and none switching on a type inside its loop.
 * </ul>
 *
 * <p>The expert outputs are accumulated into the residual stream in one reduction per output row,
 * each lane weighting its own partial sums; the host sums expert by expert, so the two differ by
 * rounding only.
 */
// @formatter:on
public final class Lfm2Kernels {

    private Lfm2Kernels() {}

    /**
     * {@code out[c] = C[c] * (Σ_t taps[c,t] · window ‖ B[c]·x[c])}, advancing the window.
     *
     * <p>{@code bcx} is {@code in_proj}'s output, {@code B ‖ C ‖ x}, each {@code dim} wide. The
     * window holds, per channel, the last {@code kernel - 1} values of {@code B ⊙ x}, oldest first,
     * at {@code stateOffset} in the shared state array. One lane per channel: channels never mix.
     */
    public static void shortConv(
            KernelContext context,
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
        int history = kernel - 1;
        float bx = bcx.get(c) * bcx.get(2 * dim + c);
        int wBase = c * kernel;
        int hBase = stateOffset + c * history;
        float sum = 0.0f;
        for (int t = 0; t < history; t++) {
            sum += taps.get(wBase + t) * state.get(hBase + t);
        }
        sum += taps.get(wBase + history) * bx;
        for (int t = 0; t + 1 < history; t++) {
            state.set(hBase + t, state.get(hBase + t + 1));
        }
        state.set(hBase + history - 1, bx);
        out.set(c, bcx.get(dim + c) * sum);
    }

    /**
     * Top-{@code used} selection over sigmoid scores, on one lane.
     *
     * <p>{@code logits} holds the router's raw projection. Selection is on {@code sigmoid(logit) +
     * bias}, ascending scan with a strict {@code >} so a tie takes the lowest index; the weights
     * are the unbiased sigmoids, renormalized to sum to one with the sum floored at the smallest
     * FP16 normal, as llama.cpp floors it. Chosen experts are marked by overwriting their logit
     * with negative infinity, which is why the logits are scratch.
     */
    public static void selectExpertsSigmoid(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray selected,
            FloatArray weights,
            int experts,
            int used) {
        if (context.globalIdx != 0) {
            return;
        }
        float total = 0.0f;
        for (int j = 0; j < used; j++) {
            int best = -1;
            float bestScore = 0.0f;
            float bestWeight = 0.0f;
            for (int e = 0; e < experts; e++) {
                float logit = logits.get(e);
                if (logit != Float.NEGATIVE_INFINITY) {
                    float p = 1.0f / (1.0f + TornadoMath.exp(-logit));
                    float score = p + bias.get(e);
                    if (best < 0 || score > bestScore) {
                        best = e;
                        bestScore = score;
                        bestWeight = p;
                    }
                }
            }
            selected.set(j, best);
            weights.set(j, bestWeight);
            logits.set(best, Float.NEGATIVE_INFINITY);
            total += bestWeight;
        }
        if (total < 6.103515625e-5f) {
            total = 6.103515625e-5f;
        }
        for (int j = 0; j < used; j++) {
            weights.set(j, weights.get(j) / total);
        }
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────────

    /** Lane-partial dot of global row {@code row} of a Q4_K matrix with {@code x[xOffset..]}. */
    private static float partialDotQ4_K(
            KernelContext context,
            int localSize,
            ByteArray w,
            int row,
            FloatArray x,
            int xOffset,
            int n) {
        int blocksPerRow = n / 256;
        int rowBlockOffset = row * blocksPerRow;
        float partial = 0.0f;
        for (int j = context.localIdx; j < n; j += localSize) {
            int blockIdx = j / 256;
            int withinBlock = j - blockIdx * 256;
            int blockByteOffset = (rowBlockOffset + blockIdx) * 144;
            partial +=
                    (TransformerComputeKernelsQ4_K.decode(w, blockByteOffset, withinBlock))
                            * x.get(xOffset + j);
        }
        return partial;
    }

    /**
     * {@code hb[s·H + r] = silu(gate_e[r]·x) · (up_e[r]·x)} for every selected slot {@code s} and
     * expert row {@code r}, {@code e = selected[s]}. One workgroup per (slot, row).
     */
    public static void expertGateUpSiLUQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray gate,
            ByteArray up,
            IntArray selected,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int group = context.groupIdx;
        if (group >= used * expertHidden) {
            return;
        }
        int slot = group / expertHidden;
        int r = group - slot * expertHidden;
        int row = selected.get(slot) * expertHidden + r;
        int localId = context.localIdx;
        float[] gateSums = context.allocateFloatLocalArray(localSize);
        float[] upSums = context.allocateFloatLocalArray(localSize);
        gateSums[localId] = partialDotQ4_K(context, localSize, gate, row, x, 0, dim);
        upSums[localId] = partialDotQ4_K(context, localSize, up, row, x, 0, dim);
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                gateSums[localId] += gateSums[localId + stride];
                upSums[localId] += upSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            float g = gateSums[0];
            hb.set(group, g / (1.0f + TornadoMath.exp(-g)) * upSums[0]);
        }
    }

    /**
     * {@code x[r] += Σ_s weights[s] · (down_e[r] · hb[s·H ..])}, {@code e = selected[s]}, summed in
     * selection order. One workgroup per output row.
     */
    public static void expertDownResidualQ4_K(
            KernelContext context,
            FloatArray hb,
            FloatArray x,
            ByteArray down,
            IntArray selected,
            FloatArray weights,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int r = context.groupIdx;
        if (r >= dim) {
            return;
        }
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);
        // Each lane weights its own partials, so one reduction covers every slot. The products are
        // summed in a different order than slot by slot, which is a rounding difference only.
        float partial = 0.0f;
        for (int s = 0; s < used; s++) {
            int row = selected.get(s) * dim + r;
            partial +=
                    weights.get(s)
                            * partialDotQ4_K(
                                    context,
                                    localSize,
                                    down,
                                    row,
                                    hb,
                                    s * expertHidden,
                                    expertHidden);
        }
        localSums[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            x.set(r, x.get(r) + localSums[0]);
        }
    }

    // ── Q5_K ─────────────────────────────────────────────────────────────────────

    /** Lane-partial dot of global row {@code row} of a Q5_K matrix with {@code x[xOffset..]}. */
    private static float partialDotQ5_K(
            KernelContext context,
            int localSize,
            ByteArray w,
            int row,
            FloatArray x,
            int xOffset,
            int n) {
        int blocksPerRow = n / 256;
        int rowBlockOffset = row * blocksPerRow;
        float partial = 0.0f;
        for (int j = context.localIdx; j < n; j += localSize) {
            int blockIdx = j / 256;
            int withinBlock = j - blockIdx * 256;
            int blockByteOffset = (rowBlockOffset + blockIdx) * 176;
            partial +=
                    (TransformerComputeKernelsQ5_K.decode(w, blockByteOffset, withinBlock))
                            * x.get(xOffset + j);
        }
        return partial;
    }

    /**
     * {@code hb[s·H + r] = silu(gate_e[r]·x) · (up_e[r]·x)} for every selected slot {@code s} and
     * expert row {@code r}, {@code e = selected[s]}. One workgroup per (slot, row).
     */
    public static void expertGateUpSiLUQ5_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray gate,
            ByteArray up,
            IntArray selected,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int group = context.groupIdx;
        if (group >= used * expertHidden) {
            return;
        }
        int slot = group / expertHidden;
        int r = group - slot * expertHidden;
        int row = selected.get(slot) * expertHidden + r;
        int localId = context.localIdx;
        float[] gateSums = context.allocateFloatLocalArray(localSize);
        float[] upSums = context.allocateFloatLocalArray(localSize);
        gateSums[localId] = partialDotQ5_K(context, localSize, gate, row, x, 0, dim);
        upSums[localId] = partialDotQ5_K(context, localSize, up, row, x, 0, dim);
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                gateSums[localId] += gateSums[localId + stride];
                upSums[localId] += upSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            float g = gateSums[0];
            hb.set(group, g / (1.0f + TornadoMath.exp(-g)) * upSums[0]);
        }
    }

    /**
     * {@code x[r] += Σ_s weights[s] · (down_e[r] · hb[s·H ..])}, {@code e = selected[s]}, summed in
     * selection order. One workgroup per output row.
     */
    public static void expertDownResidualQ5_K(
            KernelContext context,
            FloatArray hb,
            FloatArray x,
            ByteArray down,
            IntArray selected,
            FloatArray weights,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int r = context.groupIdx;
        if (r >= dim) {
            return;
        }
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);
        // Each lane weights its own partials, so one reduction covers every slot. The products are
        // summed in a different order than slot by slot, which is a rounding difference only.
        float partial = 0.0f;
        for (int s = 0; s < used; s++) {
            int row = selected.get(s) * dim + r;
            partial +=
                    weights.get(s)
                            * partialDotQ5_K(
                                    context,
                                    localSize,
                                    down,
                                    row,
                                    hb,
                                    s * expertHidden,
                                    expertHidden);
        }
        localSums[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            x.set(r, x.get(r) + localSums[0]);
        }
    }

    // ── Q6_K ─────────────────────────────────────────────────────────────────────

    /** Lane-partial dot of global row {@code row} of a Q6_K matrix with {@code x[xOffset..]}. */
    private static float partialDotQ6_K(
            KernelContext context,
            int localSize,
            ByteArray w,
            int row,
            FloatArray x,
            int xOffset,
            int n) {
        int blocksPerRow = n / 256;
        int rowBlockOffset = row * blocksPerRow;
        float partial = 0.0f;
        for (int j = context.localIdx; j < n; j += localSize) {
            int blockIdx = j / 256;
            int withinBlock = j - blockIdx * 256;
            int blockByteOffset = (rowBlockOffset + blockIdx) * 210;
            partial +=
                    (TransformerComputeKernelsQ6_K.decode(w, blockByteOffset, withinBlock))
                            * x.get(xOffset + j);
        }
        return partial;
    }

    /**
     * {@code hb[s·H + r] = silu(gate_e[r]·x) · (up_e[r]·x)} for every selected slot {@code s} and
     * expert row {@code r}, {@code e = selected[s]}. One workgroup per (slot, row).
     */
    public static void expertGateUpSiLUQ6_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray gate,
            ByteArray up,
            IntArray selected,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int group = context.groupIdx;
        if (group >= used * expertHidden) {
            return;
        }
        int slot = group / expertHidden;
        int r = group - slot * expertHidden;
        int row = selected.get(slot) * expertHidden + r;
        int localId = context.localIdx;
        float[] gateSums = context.allocateFloatLocalArray(localSize);
        float[] upSums = context.allocateFloatLocalArray(localSize);
        gateSums[localId] = partialDotQ6_K(context, localSize, gate, row, x, 0, dim);
        upSums[localId] = partialDotQ6_K(context, localSize, up, row, x, 0, dim);
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                gateSums[localId] += gateSums[localId + stride];
                upSums[localId] += upSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            float g = gateSums[0];
            hb.set(group, g / (1.0f + TornadoMath.exp(-g)) * upSums[0]);
        }
    }

    /**
     * {@code x[r] += Σ_s weights[s] · (down_e[r] · hb[s·H ..])}, {@code e = selected[s]}, summed in
     * selection order. One workgroup per output row.
     */
    public static void expertDownResidualQ6_K(
            KernelContext context,
            FloatArray hb,
            FloatArray x,
            ByteArray down,
            IntArray selected,
            FloatArray weights,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int r = context.groupIdx;
        if (r >= dim) {
            return;
        }
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);
        // Each lane weights its own partials, so one reduction covers every slot. The products are
        // summed in a different order than slot by slot, which is a rounding difference only.
        float partial = 0.0f;
        for (int s = 0; s < used; s++) {
            int row = selected.get(s) * dim + r;
            partial +=
                    weights.get(s)
                            * partialDotQ6_K(
                                    context,
                                    localSize,
                                    down,
                                    row,
                                    hb,
                                    s * expertHidden,
                                    expertHidden);
        }
        localSums[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            x.set(r, x.get(r) + localSums[0]);
        }
    }

    // ── Q8_0 ─────────────────────────────────────────────────────────────────────

    /** Lane-partial dot of global row {@code row} of a Q8_0 matrix with {@code x[xOffset..]}. */
    private static float partialDotQ8_0(
            KernelContext context,
            int localSize,
            ByteArray w,
            int row,
            FloatArray x,
            int xOffset,
            int n) {
        int blocksPerRow = n / 32;
        int rowBlockOffset = row * blocksPerRow;
        float partial = 0.0f;
        for (int j = context.localIdx; j < n; j += localSize) {
            int blockIdx = j / 32;
            int withinBlock = j - blockIdx * 32;
            int blockByteOffset = (rowBlockOffset + blockIdx) * 34;
            partial +=
                    (w.getHalfFloat(blockByteOffset).getFloat32()
                                    * w.get(blockByteOffset + 2 + withinBlock))
                            * x.get(xOffset + j);
        }
        return partial;
    }

    /**
     * {@code hb[s·H + r] = silu(gate_e[r]·x) · (up_e[r]·x)} for every selected slot {@code s} and
     * expert row {@code r}, {@code e = selected[s]}. One workgroup per (slot, row).
     */
    public static void expertGateUpSiLUQ8_0(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray gate,
            ByteArray up,
            IntArray selected,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int group = context.groupIdx;
        if (group >= used * expertHidden) {
            return;
        }
        int slot = group / expertHidden;
        int r = group - slot * expertHidden;
        int row = selected.get(slot) * expertHidden + r;
        int localId = context.localIdx;
        float[] gateSums = context.allocateFloatLocalArray(localSize);
        float[] upSums = context.allocateFloatLocalArray(localSize);
        gateSums[localId] = partialDotQ8_0(context, localSize, gate, row, x, 0, dim);
        upSums[localId] = partialDotQ8_0(context, localSize, up, row, x, 0, dim);
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                gateSums[localId] += gateSums[localId + stride];
                upSums[localId] += upSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            float g = gateSums[0];
            hb.set(group, g / (1.0f + TornadoMath.exp(-g)) * upSums[0]);
        }
    }

    /**
     * {@code x[r] += Σ_s weights[s] · (down_e[r] · hb[s·H ..])}, {@code e = selected[s]}, summed in
     * selection order. One workgroup per output row.
     */
    public static void expertDownResidualQ8_0(
            KernelContext context,
            FloatArray hb,
            FloatArray x,
            ByteArray down,
            IntArray selected,
            FloatArray weights,
            int dim,
            int expertHidden,
            int used,
            int localSize) {
        int r = context.groupIdx;
        if (r >= dim) {
            return;
        }
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);
        // Each lane weights its own partials, so one reduction covers every slot. The products are
        // summed in a different order than slot by slot, which is a rounding difference only.
        float partial = 0.0f;
        for (int s = 0; s < used; s++) {
            int row = selected.get(s) * dim + r;
            partial +=
                    weights.get(s)
                            * partialDotQ8_0(
                                    context,
                                    localSize,
                                    down,
                                    row,
                                    hb,
                                    s * expertHidden,
                                    expertHidden);
        }
        localSums[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            x.set(r, x.get(r) + localSums[0]);
        }
    }
}
