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

    /** Experts {@link #selectExpertsSigmoidSimdgroup} holds in threadgroup memory. */
    public static final int MAX_EXPERTS = 256;

    /** Router rows per workgroup of {@link #routerLogitsWarps}: four warps each. */
    public static final int ROUTER_ROWS = 2;

    /**
     * {@code logits[e] = w[e] · x} over the F32 router, four warps per row: warp {@code q} sums
     * quarter {@code q} of the row, lane-strided, and the quarters are added in order. Example:
     * LFM2.5-24B-A2B's 64 experts over 2048, 32 workgroups. Launch {@code rows / ROUTER_ROWS}
     * workgroups of 256 threads; {@code n % 128 == 0}.
     */
    public static void routerLogitsWarps(
            KernelContext context, FloatArray x, FloatArray logits, FloatArray w, int n, int rows) {
        float[] partial = context.allocateFloatLocalArray(8);
        int warp = context.localIdx >> 5;
        int lane = context.localIdx & 31;
        int row = context.groupIdx * ROUTER_ROWS + (warp >> 2);
        int quarter = n >> 2;
        int start = (warp & 3) * quarter;
        float sum = 0.0f;
        if (row < rows) {
            int base = row * n;
            for (int i = start + lane; i < start + quarter; i += 32) {
                sum += w.get(base + i) * x.get(i);
            }
        }
        sum += context.simdShuffleDown(sum, 16);
        sum += context.simdShuffleDown(sum, 8);
        sum += context.simdShuffleDown(sum, 4);
        sum += context.simdShuffleDown(sum, 2);
        sum += context.simdShuffleDown(sum, 1);
        if (lane == 0) {
            partial[warp] = sum;
        }
        context.localBarrier();
        int t = context.localIdx;
        int r = context.groupIdx * ROUTER_ROWS + t;
        if (t < ROUTER_ROWS && r < rows) {
            logits.set(
                    r,
                    partial[4 * t] + partial[4 * t + 1] + partial[4 * t + 2] + partial[4 * t + 3]);
        }
    }

    /** Largest {@code n} {@link #routerLogitsRmsApply} holds in threadgroup memory. */
    public static final int ROUTER_NORM_MAX_DIM = 4096;

    /**
     * The FFN norm's finalize and apply inside the F32 router, two dispatches fewer: every
     * workgroup adds the norm's partial sums of squares ({@code partials} from index 1) into the
     * scale as the finalize kernel does, then {@code logits[e] = w[e] · (weight ⊙ scale·x)}.
     * Workgroup 0 also writes the normalized vector to {@code xb} for the expert kernels, after its
     * dot product: a device store inside the loop keeps the loads from overlapping. Example:
     * LFM2-8B-A1B, 32 experts over 2048, 32 workgroups of 128. Launch {@code rows} workgroups of
     * 128.
     */
    public static void routerLogitsRmsApply(
            KernelContext context,
            FloatArray x,
            FloatArray xb,
            FloatArray normWeight,
            FloatArray partials,
            FloatArray logits,
            FloatArray w,
            int n,
            int rows,
            float eps) {
        float[] localSum = context.allocateFloatLocalArray(128);
        float[] normedLocal = context.allocateFloatLocalArray(ROUTER_NORM_MAX_DIM);
        int row = context.groupIdx;
        int t = context.localIdx;
        float ss = 0.0f;
        for (int i = 1; i < partials.getSize(); i++) {
            ss += partials.get(i);
        }
        ss /= n;
        ss += eps;
        ss = 1.0f / TornadoMath.sqrt(ss);

        float partialSum = 0.0f;
        if (row < rows) {
            int rowOffset = row * n;
            for (int j = t; j < n; j += 128) {
                float normed = normWeight.get(j) * (ss * x.get(j));
                normedLocal[j] = normed;
                partialSum += w.get(rowOffset + j) * normed;
            }
        }
        if (row == 0) {
            for (int j = t; j < n; j += 128) {
                xb.set(j, normedLocal[j]);
            }
        }
        localSum[t] = partialSum;
        context.localBarrier();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (t < stride) {
                localSum[t] += localSum[t + stride];
            }
            context.localBarrier();
        }
        if (t == 0 && row < rows) {
            logits.set(row, localSum[0]);
        }
    }

    /** Experts {@link #selectExpertSimdgroup} holds in registers, two per lane. */
    public static final int REGISTER_EXPERTS = 64;

    /**
     * {@link #selectExpertsSigmoidSimdgroup} with the scores in registers, for a consumer's SIMD
     * group: runs the rounds up to {@code slot} and returns its expert; with {@code write} it runs
     * all {@code used} rounds and stores {@code selected} and the renormalized {@code weights}.
     * Example: LFM2.5-24B-A2B, 64 experts, lane {@code k} holds experts {@code k} and {@code k+32}.
     */
    public static int selectExpertSimdgroup(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray selected,
            FloatArray weights,
            int experts,
            int used,
            int slot,
            boolean write) {
        int lane = context.localIdx & 31;
        float p0 = 0.0f;
        float p1 = 0.0f;
        float s0 = Float.NEGATIVE_INFINITY;
        float s1 = Float.NEGATIVE_INFINITY;
        if (lane < experts) {
            p0 = 1.0f / (1.0f + TornadoMath.exp(-logits.get(lane)));
            s0 = p0 + bias.get(lane);
        }
        if (lane + 32 < experts) {
            p1 = 1.0f / (1.0f + TornadoMath.exp(-logits.get(lane + 32)));
            s1 = p1 + bias.get(lane + 32);
        }
        int rounds = write ? used : slot + 1;
        int mine = 0;
        float total = 0.0f;
        for (int k = 0; k < rounds; k++) {
            float best = Float.NEGATIVE_INFINITY;
            float bestIndex = 1.0e9f;
            if (lane < experts && s0 > best) {
                best = s0;
                bestIndex = lane;
            }
            if (lane + 32 < experts && s1 > best) {
                best = s1;
                bestIndex = lane + 32;
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
            float w = context.simdShuffle(chosen < 32 ? p0 : p1, chosen & 31);
            total += w;
            if (k == slot) {
                mine = chosen;
            }
            if (write && lane == 0) {
                selected.set(k, chosen);
                weights.set(k, w);
            }
            if (lane == chosen) {
                s0 = Float.NEGATIVE_INFINITY;
            }
            if (lane + 32 == chosen) {
                s1 = Float.NEGATIVE_INFINITY;
            }
        }
        if (write && lane == 0) {
            if (total < 6.103515625e-5f) {
                total = 6.103515625e-5f;
            }
            for (int j = 0; j < used; j++) {
                weights.set(j, weights.get(j) / total);
            }
        }
        return mine;
    }

    /**
     * {@link #selectExpertsSigmoid} on one SIMD group: each round takes the largest score across
     * the lanes, then the lowest index among the lanes that hold it, so the same tie rule. Example:
     * LFM2.5-24B-A2B, 64 experts, two per lane, four rounds. Launch one workgroup of 32 threads.
     */
    public static void selectExpertsSigmoidSimdgroup(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray selected,
            FloatArray weights,
            int experts,
            int used) {
        int lane = context.localIdx;
        float[] scores = context.allocateFloatLocalArray(MAX_EXPERTS);
        float[] probs = context.allocateFloatLocalArray(MAX_EXPERTS);
        for (int e = lane; e < experts; e += 32) {
            float p = 1.0f / (1.0f + TornadoMath.exp(-logits.get(e)));
            probs[e] = p;
            scores[e] = p + bias.get(e);
        }
        context.localBarrier();
        float total = 0.0f;
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            float bestIndex = 1.0e9f;
            for (int e = lane; e < experts; e += 32) {
                float score = scores[e];
                if (score > best) {
                    best = score;
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
            float w = probs[chosen];
            total += w;
            if (lane == 0) {
                selected.set(k, chosen);
                weights.set(k, w);
            }
            context.localBarrier();
            if (lane == chosen % 32) {
                scores[chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        if (lane == 0) {
            if (total < 6.103515625e-5f) {
                total = 6.103515625e-5f;
            }
            for (int j = 0; j < used; j++) {
                weights.set(j, weights.get(j) / total);
            }
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
