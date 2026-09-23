package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@code qwen35} mixer kernels over a chunk of prompt tokens.
 *
 * <h2>Two kinds of kernel, and the difference is the whole design</h2>
 *
 * <p>Most of what a layer does is per token: a projection, a norm, a rotation, a gate. Those become
 * batched by adding a row index, and the rows are independent.
 *
 * <p>Two are not. A recurrent layer's convolution keeps a rolling window per channel and its
 * delta-net keeps a matrix per value head, and token {@code t} reads what token {@code t-1} wrote.
 * Those two kernels <b>scan</b>: a lane walks the chunk in order, in one loop, updating the state
 * it alone owns. That is exact rather than approximate — a convolution channel's window is private
 * to that channel and a delta-net value column's state is private to that column, so the sequential
 * dependency lives entirely inside one lane and needs no barrier and no ordering between lanes.
 *
 * <p>It is also why the result cannot depend on the chunk size: the arithmetic per token is the
 * same expression in the same order as the single-token kernel performs, and the batch width only
 * decides how many iterations a lane runs before returning.
 *
 * <h2>Padding</h2>
 *
 * <p>Every kernel launches a fixed number of rows and is told how many are active. An inactive row
 * computes nothing, writes no key/value entry, and — in the scans — is never reached, because the
 * loop bound is the active count rather than the launch width.
 *
 * <p>Bodies are lifted into lane methods wherever the arithmetic is worth checking on the host, for
 * the reason {@code Qwen35DeltaNetKernels} gives: a method taking a {@link KernelContext} can only
 * be exercised by running it on a device.
 */
// @formatter:on
public final class Qwen35BatchKernels {

    /** Positions whose weights the workgroup computes together in the value pass. */
    private static final int ATTENTION_TILE = 16;

    /**
     * Output elements one lane carries in the value pass.
     *
     * <p>{@code headSize / localSize}, rounded up, for this family's 256-wide head against a
     * 128-lane workgroup. A constant because a private array's extent has to be one.
     */
    private static final int ATTENTION_SLOTS = 4;

    /** Head dimensions one staged key tile of the first pass holds. */
    private static final int ATTENTION_STAGE_DIMS = 32;

    /**
     * Positions the wide value pass takes at a time: one per lane of the 128-lane workgroup, so
     * every lane computes one weight and the two barriers are paid per 128 positions. The
     * candidate's own constant; {@link #ATTENTION_TILE} stays the reference kernels'.
     */
    private static final int ATTENTION_VALUE_TILE_WIDE = 128;

    /**
     * Lanes the staged first pass is written for: its load mapping puts one position's 32
     * dimensions on one warp and covers a 128-position tile with 128 lanes.
     */
    public static final int ATTENTION_STAGE_LANES = 128;

    /**
     * Row pitch of the transposed key tile, in floats: the 128 positions of a tile plus one, so
     * that the 32 dimensions of one position — stored by one warp in one instruction — land in 32
     * distinct banks.
     */
    private static final int ATTENTION_STAGE_LD = 129;

    /** Warps of the warp first pass: one per lane-quad of the 128-lane workgroup. */
    private static final int ATTENTION_WARPS = 4;

    /** Head width the warp first pass is written for: eight dimensions a lane over 32 lanes. */
    private static final int ATTENTION_WARP_HEAD = 256;

    private Qwen35BatchKernels() {}

    // ---- the recurrent scans -------------------------------------------------

    /**
     * One channel of the depthwise causal convolution, walked across the chunk in token order.
     *
     * <p>The single-token kernel's body, in a loop: for each token it convolves the window with
     * this channel's taps, writes that token's output, and shifts the window. A lane owns one
     * channel's window slice, so nothing here races and nothing needs a barrier.
     *
     * @param channel the lane
     * @param activeRows how many of the chunk's rows carry a real token
     */
    static void causalConv1dScanLane(
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray outBatch,
            int channels,
            int kernel,
            int windowOffset,
            int activeRows,
            int channel) {
        int history = kernel - 1;
        int wBase = channel * kernel;
        int hBase = windowOffset + channel * history;

        for (int row = 0; row < activeRows; row++) {
            float x = inputBatch.get(row * channels + channel);

            float sum = 0.0f;
            for (int t = 0; t < history; t++) {
                sum += weight.get(wBase + t) * window.get(hBase + t);
            }
            sum += weight.get(wBase + history) * x;
            outBatch.set(row * channels + channel, sum);

            for (int t = 0; t + 1 < history; t++) {
                window.set(hBase + t, window.get(hBase + t + 1));
            }
            if (history > 0) {
                window.set(hBase + history - 1, x);
            }
        }
    }

    /** One lane per channel; each walks the whole chunk. */
    public static void causalConv1dScan(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray outBatch,
            int channels,
            int kernel,
            int windowOffset,
            IntArray batchInfo) {
        int channel = context.globalIdx;
        if (channel >= channels) {
            return;
        }
        causalConv1dScanLane(
                inputBatch,
                weight,
                window,
                outBatch,
                channels,
                kernel,
                windowOffset,
                batchInfo.get(1),
                channel);
    }

    // @formatter:off
    /**
     * The causal convolution with a lane per (row, channel), the SiLU and the three-way split
     * folded into its store. Row {@code r} of channel {@code c} is {@code w[0] x[r-3] + w[1] x[r-2]
     * + w[2] x[r-1] + w[3] x[r]} (kernel 4), {@code x[i]} for {@code i < 0} being the window the
     * chunk started with, so the rows are independent given that window; the taps are summed in
     * that order. The output {@code y} is written as {@code y / (1 + exp(-y))} — the expression of
     * {@link #siluInPlaceBatch} — into the query, key or value row its channel belongs to, as
     * {@link #splitThreeWayBatch} copies. The window is not touched: {@link
     * #causalConv1dWindowUpdate} writes the chunk's final window afterwards, as a task of its own,
     * after every row has read the initial one. Bit-equal to {@link #causalConv1dScan} followed by
     * the SiLU and the split.
     *
     * <p>Worker: {@code rows * channels} lanes; {@code channels = dimA + dimB + dimC}.
     */
    // @formatter:on
    public static void causalConv1dSiluSplitBatch(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC,
            int kernel,
            int windowOffset,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        int channels = dimA + dimB + dimC;
        if (lane >= channels * batchInfo.get(1)) {
            return;
        }
        int row = lane / channels;
        int channel = lane - row * channels;
        int history = kernel - 1;
        int wBase = channel * kernel;
        int hBase = windowOffset + channel * history;

        float sum = 0.0f;
        for (int t = 0; t < history; t++) {
            int source = row - history + t;
            float h;
            if (source < 0) {
                h = window.get(hBase + source + history);
            } else {
                h = inputBatch.get(source * channels + channel);
            }
            sum += weight.get(wBase + t) * h;
        }
        sum += weight.get(wBase + history) * inputBatch.get(lane);
        float value = sum / (1.0f + TornadoMath.exp(-sum));
        if (channel < dimA) {
            a.set(row * dimA + channel, value);
        } else if (channel < dimA + dimB) {
            b.set(row * dimB + (channel - dimA), value);
        } else {
            c.set(row * dimC + (channel - dimA - dimB), value);
        }
    }

    /**
     * The window {@link #causalConv1dScan} leaves after a chunk: the chunk's last {@code kernel -
     * 1} inputs of each channel, taken from the initial window where the chunk is shorter than
     * that. One lane per channel reads its old window before writing any of it. Runs after {@link
     * #causalConv1dSiluSplitBatch}, which reads the initial window. Written for a four-tap kernel.
     * Worker: {@code channels} lanes.
     */
    public static void causalConv1dWindowUpdate(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray window,
            int channels,
            int kernel,
            int windowOffset,
            IntArray batchInfo) {
        int channel = context.globalIdx;
        if (channel >= channels) {
            return;
        }
        int activeRows = batchInfo.get(1);
        int history = kernel - 1;
        int hBase = windowOffset + channel * history;
        float h0 = window.get(hBase);
        float h1 = window.get(hBase + 1);
        float h2 = window.get(hBase + 2);
        for (int t = 0; t < history; t++) {
            int source = activeRows - history + t;
            float value;
            if (source < 0) {
                int old = source + history;
                value = old == 0 ? h0 : old == 1 ? h1 : h2;
            } else {
                value = inputBatch.get(source * channels + channel);
            }
            window.set(hBase + t, value);
        }
    }

    /**
     * One value column of one head, walked across the chunk in token order.
     *
     * <p>Per token: decay this column, predict, correct, accumulate the rank-one update, read out.
     * The same two passes the single-token kernel makes, and the same modulo mapping from a value
     * head to its key head — the reference repeats key heads by tiling, and dividing instead pairs
     * every value head with the wrong key.
     *
     * @param lane {@code head * stateDim + column}
     */
    static void deltaRuleScanLane(
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            int activeRows,
            int lane) {
        int head = lane / stateDim;
        int column = lane - head * stateDim;

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;

            float prediction = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = stateBase + i * stateDim + column;
                float decayed = state.get(index) * g;
                state.set(index, decayed);
                prediction += decayed * kBatch.get(keyRow + i);
            }

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            float readout = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = stateBase + i * stateDim + column;
                float updated = state.get(index) + kBatch.get(keyRow + i) * correction;
                state.set(index, updated);
                readout += updated * qBatch.get(keyRow + i);
            }
            outBatch.set(valueRow + column, readout);
        }
    }

    /** One lane per (value head, value column); each walks the whole chunk. */
    public static void deltaRuleScan(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * stateDim) {
            return;
        }
        deltaRuleScanLane(
                qBatch,
                kBatch,
                vBatch,
                decayBatch,
                betaBatch,
                state,
                outBatch,
                valueHeads,
                keyHeads,
                stateDim,
                stateOffset,
                batchInfo.get(1),
                lane);
    }

    /** Columns one workgroup of {@link #deltaRuleScanShared} owns: one lane each. */
    public static final int DELTA_SHARED_COLUMNS = 32;

    /** State rows the shared tile is sized for: this family's 128-wide state. */
    public static final int DELTA_SHARED_STATE_DIM = 128;

    /**
     * Whether the state geometry can run the shared-state scan: the tile is sized for a 128-wide
     * state, whose columns fall into whole groups of 32.
     */
    public static boolean deltaSharedEligible(int stateDim) {
        return stateDim == DELTA_SHARED_STATE_DIM;
    }

    // @formatter:off
    /**
     * {@link #deltaRuleScan} with the state column held in shared memory across the chunk.
     *
     * <p>One 32-lane workgroup owns 32 adjacent value columns of one head; lane {@code l} owns
     * column {@code col0 + l}. The 128 x 32 state slice is staged once from the persistent state —
     * for each state row, the 32 lanes read 32 consecutive floats — into {@code tile[i * 32 +
     * lane]}, so a lane's column lives in one bank and no lane ever touches another lane's
     * elements; the scan then walks the active tokens in order doing exactly the per-token decay,
     * prediction, correction, update and readout of the reference lane, in the reference order,
     * reading and writing the column through the tile, and writes the column back once at the end.
     * No consumer reads the persistent state between tokens of a chunk: the scan is the only writer
     * in the batched graph and the next reader is the next chunk's scan (or decode's), so the
     * intermediate values are private to the lane either way.
     *
     * <p>Worker: {@code valueHeads * (stateDim / 32)} groups of 32 lanes. Requires {@code stateDim
     * % 32 == 0}, which {@link #deltaSharedEligible(int)} decides.
     */
    // @formatter:on
    public static void deltaRuleScanShared(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            IntArray batchInfo) {
        int lane = context.localIdx;
        int group = context.groupIdx;
        int columnGroups = stateDim / DELTA_SHARED_COLUMNS;
        int head = group / columnGroups;
        int column = (group - head * columnGroups) * DELTA_SHARED_COLUMNS + lane;
        int activeRows = batchInfo.get(1);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        // The state column: tile[i * 32 + lane] is element i of this lane's column.
        float[] tile =
                context.allocateFloatLocalArray(DELTA_SHARED_STATE_DIM * DELTA_SHARED_COLUMNS);
        for (int i = 0; i < stateDim; i++) {
            tile[i * DELTA_SHARED_COLUMNS + lane] = state.get(stateBase + i * stateDim + column);
        }

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;

            float prediction = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = i * DELTA_SHARED_COLUMNS + lane;
                float decayed = tile[index] * g;
                tile[index] = decayed;
                prediction += decayed * kBatch.get(keyRow + i);
            }

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            float readout = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = i * DELTA_SHARED_COLUMNS + lane;
                float updated = tile[index] + kBatch.get(keyRow + i) * correction;
                tile[index] = updated;
                readout += updated * qBatch.get(keyRow + i);
            }
            outBatch.set(valueRow + column, readout);
        }

        for (int i = 0; i < stateDim; i++) {
            state.set(stateBase + i * stateDim + column, tile[i * DELTA_SHARED_COLUMNS + lane]);
        }
    }

    /** Warps per workgroup of {@link #deltaRuleScanWarp}: each owns one value column. */
    public static final int DELTA_WARP_COLUMNS_PER_GROUP = 4;

    /** Lanes of {@link #deltaRuleScanWarp}'s workgroup. */
    public static final int DELTA_WARP_LOCAL = DELTA_WARP_COLUMNS_PER_GROUP * 32;

    /**
     * Whether the state geometry can run the warp-per-column scan: a 128-wide state, four rows a
     * lane.
     */
    public static boolean deltaWarpEligible(int stateDim) {
        return stateDim == DELTA_SHARED_STATE_DIM;
    }

    /**
     * The sum of {@code value} over the 32 lanes of the warp, folded with five shuffle-down steps
     * (lane {@code l} adds lane {@code l + 16}, then {@code l + 8}, ... ), and broadcast from lane
     * zero so every lane holds it. A fixed association: {@code ((v0 + v16) + (v8 + v24)) + ...},
     * the same on every call.
     */
    private static float warpSumBroadcast(KernelContext context, float value) {
        float sum = value;
        sum += context.simdShuffleDown(sum, 16);
        sum += context.simdShuffleDown(sum, 8);
        sum += context.simdShuffleDown(sum, 4);
        sum += context.simdShuffleDown(sum, 2);
        sum += context.simdShuffleDown(sum, 1);
        return context.simdBroadcastFirst(sum);
    }

    // @formatter:off
    /**
     * The batched delta-rule scan with one warp per value column and the column's 128 state
     * elements spread over the warp's lanes in registers, four a lane.
     *
     * <p>Lane {@code l} holds state rows {@code l, l + 32, l + 64, l + 96} of its warp's column,
     * loaded once from the persistent {@code [head][row][column]} layout (a strided gather, since a
     * lane's rows are a state width apart) and written back once. Per token, in order: every lane
     * decays each of its four elements, multiplies each by that row's key and folds the four
     * products; the warp sums the 32 partials for the prediction; every lane computes the same
     * correction from the broadcast prediction, adds {@code key * correction} to each element and
     * folds each updated element times that row's query; the warp sums the partials for the
     * readout, which lane zero stores. The head-to-key-head mapping, the token order, the query
     * scaling and the decay input are the per-lane scan's.
     *
     * <p><b>Not bit-preserving.</b> The per-lane scan sums a column's 128 products in row order in
     * one accumulator; here each lane sums four and the warp folds the 32 partials as a tree, so
     * the prediction and the readout are reassociated. The decay is applied to each element before
     * its product, as in the per-lane scan, and not factored out of the sum.
     *
     * <p>Worker: {@code valueHeads * stateDim} lanes in groups of {@link #DELTA_WARP_LOCAL}: four
     * warps a group, one column each.
     */
    // @formatter:on
    public static void deltaRuleScanWarp(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            IntArray batchInfo) {
        int local = context.localIdx;
        int lane = local & 31;
        int warp = local >> 5;
        int column0 = context.groupIdx * DELTA_WARP_COLUMNS_PER_GROUP + warp;
        int head = column0 / stateDim;
        int column = column0 - head * stateDim;
        int activeRows = batchInfo.get(1);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        // This lane's four state rows: lane, lane + 32, lane + 64, lane + 96.
        int row0 = lane;
        int row1 = lane + 32;
        int row2 = lane + 64;
        int row3 = lane + 96;
        float s0 = state.get(stateBase + row0 * stateDim + column);
        float s1 = state.get(stateBase + row1 * stateDim + column);
        float s2 = state.get(stateBase + row2 * stateDim + column);
        float s3 = state.get(stateBase + row3 * stateDim + column);

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;
            float k0 = kBatch.get(keyRow + row0);
            float k1 = kBatch.get(keyRow + row1);
            float k2 = kBatch.get(keyRow + row2);
            float k3 = kBatch.get(keyRow + row3);

            // Decay each element, then its product with the key; fold the four in row order.
            s0 = s0 * g;
            s1 = s1 * g;
            s2 = s2 * g;
            s3 = s3 * g;
            float partial = s0 * k0;
            partial += s1 * k1;
            partial += s2 * k2;
            partial += s3 * k3;
            float prediction = warpSumBroadcast(context, partial);

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            s0 = s0 + k0 * correction;
            s1 = s1 + k1 * correction;
            s2 = s2 + k2 * correction;
            s3 = s3 + k3 * correction;
            float readoutPartial = s0 * qBatch.get(keyRow + row0);
            readoutPartial += s1 * qBatch.get(keyRow + row1);
            readoutPartial += s2 * qBatch.get(keyRow + row2);
            readoutPartial += s3 * qBatch.get(keyRow + row3);
            float readout = warpSumBroadcast(context, readoutPartial);
            if (lane == 0) {
                outBatch.set(valueRow + column, readout);
            }
        }

        state.set(stateBase + row0 * stateDim + column, s0);
        state.set(stateBase + row1 * stateDim + column, s1);
        state.set(stateBase + row2 * stateDim + column, s2);
        state.set(stateBase + row3 * stateDim + column, s3);
    }

    // ---- per-token kernels, with a row index ---------------------------------

    /** SiLU over a chunk, in place. One lane per element of the chunk. */
    public static void siluInPlaceBatch(
            KernelContext context, FloatArray values, int count, IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        float v = values.get(lane);
        values.set(lane, v / (1.0f + TornadoMath.exp(-v)));
    }

    /** {@code x *= scale} over a chunk, in place. */
    public static void scaleInPlaceBatch(
            KernelContext context, FloatArray values, float scale, int count, IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        values.set(lane, values.get(lane) * scale);
    }

    /**
     * The fused {@code q ‖ k ‖ v} split over a chunk.
     *
     * <p>Row-major throughout: the source row is {@code dimA + dimB + dimC} wide and each
     * destination row is its own width, so the split is a change of stride as well as of offset.
     */
    static void splitThreeWayBatchLane(
            FloatArray fusedBatch,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC,
            int lane) {
        int fusedWidth = dimA + dimB + dimC;
        int row = lane / fusedWidth;
        int element = lane - row * fusedWidth;
        float value = fusedBatch.get(lane);
        if (element < dimA) {
            a.set(row * dimA + element, value);
        } else if (element < dimA + dimB) {
            b.set(row * dimB + (element - dimA), value);
        } else {
            c.set(row * dimC + (element - dimA - dimB), value);
        }
    }

    /** One lane per element of the chunk's fused rows. */
    public static void splitThreeWayBatch(
            KernelContext context,
            FloatArray fusedBatch,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= (dimA + dimB + dimC) * batchInfo.get(1)) {
            return;
        }
        splitThreeWayBatchLane(fusedBatch, a, b, c, dimA, dimB, dimC, lane);
    }

    /** One head of one row scaled to unit length. One lane per (row, head). */
    public static void l2NormPerHeadBatch(
            KernelContext context,
            FloatArray values,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= heads * batchInfo.get(1)) {
            return;
        }
        int base = lane * headDim;
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.max(TornadoMath.sqrt(ss), eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, values.get(base + i) * inv);
        }
    }

    /** One row's decay and write strength. One lane per (row, value head). */
    public static void decayAndBetaBatch(
            KernelContext context,
            FloatArray alphaBatch,
            FloatArray betaBatch,
            FloatArray dtBias,
            FloatArray a,
            int valueHeads,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * batchInfo.get(1)) {
            return;
        }
        int head = lane % valueHeads;

        float raw = betaBatch.get(lane);
        betaBatch.set(lane, 1.0f / (1.0f + TornadoMath.exp(-raw)));

        float biased = alphaBatch.get(lane) + dtBias.get(head);
        float softplus = biased > 20.0f ? biased : TornadoMath.log(1.0f + TornadoMath.exp(biased));
        alphaBatch.set(lane, TornadoMath.exp(a.get(head) * softplus));
    }

    /** {@code rms_norm(values, weight) * silu(gate)} per head. One lane per (row, head). */
    public static void gatedNormPerHeadBatch(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= heads * batchInfo.get(1)) {
            return;
        }
        int base = lane * headDim;

        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);

        for (int i = 0; i < headDim; i++) {
            float z = gate.get(base + i);
            float silu = z / (1.0f + TornadoMath.exp(-z));
            values.set(base + i, weight.get(i) * (inv * values.get(base + i)) * silu);
        }
    }

    // ---- per-head norms, a workgroup per (row, head) ---------------------------------------

    // @formatter:off
    /**
     * {@link #gatedNormPerHeadBatch} with a workgroup per (row, head) and a lane per element, the
     * arithmetic and its order unchanged.
     *
     * <p>The one-lane form launches {@code rows * heads} single-thread workgroups, each walking its
     * head twice from global memory in series with the reciprocal square root sunk into the apply
     * loop. Here the workgroup loads the head once, coalesced, into shared memory; lane zero then
     * computes the sum of squares as the same left fold in element order over the staged values and
     * publishes {@code inv}; every lane applies its own element. The fold's order, the expression
     * {@code weight[i] * (inv * v) * silu(z)} and the epsilon are those of the one-lane form, so
     * the results are bit-equal to it (asserted by the test); what changes is only who does the
     * work.
     *
     * <p>Worker: {@code rows * heads} groups of {@code headDim} lanes.
     */
    // @formatter:on
    public static void gatedNormPerHeadBatchGroup(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int group = context.groupIdx;
        if (group >= heads * batchInfo.get(1)) {
            return;
        }
        int lane = context.localIdx;
        int index = group * headDim + lane;
        float[] shared = context.allocateFloatLocalArray(headDim + 1);
        float v = values.get(index);
        shared[lane] = v;
        context.localBarrier();
        if (lane == 0) {
            float ss = 0.0f;
            for (int i = 0; i < headDim; i++) {
                float x = shared[i];
                ss += x * x;
            }
            shared[headDim] = 1.0f / TornadoMath.sqrt(ss / headDim + eps);
        }
        context.localBarrier();
        float inv = shared[headDim];
        float z = gate.get(index);
        float silu = z / (1.0f + TornadoMath.exp(-z));
        values.set(index, weight.get(lane) * (inv * v) * silu);
    }

    /**
     * {@link #l2NormPerHeadBatch} with a workgroup per (row, head) and a lane per element; lane
     * zero keeps the left fold over the staged head, so the results are bit-equal to the one-lane
     * form. Worker: {@code rows * heads} groups of {@code headDim} lanes.
     */
    public static void l2NormPerHeadBatchGroup(
            KernelContext context,
            FloatArray values,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int group = context.groupIdx;
        if (group >= heads * batchInfo.get(1)) {
            return;
        }
        int lane = context.localIdx;
        int index = group * headDim + lane;
        float[] shared = context.allocateFloatLocalArray(headDim + 1);
        float v = values.get(index);
        shared[lane] = v;
        context.localBarrier();
        if (lane == 0) {
            float ss = 0.0f;
            for (int i = 0; i < headDim; i++) {
                float x = shared[i];
                ss += x * x;
            }
            shared[headDim] = 1.0f / TornadoMath.max(TornadoMath.sqrt(ss), eps);
        }
        context.localBarrier();
        values.set(index, v * shared[headDim]);
    }

    /**
     * {@link #fusedQKRmsNormBatch} with a workgroup per (row, head) and a lane per element; lane
     * zero keeps the left fold over the staged head, so the results are bit-equal to the one-lane
     * form. Groups past the query heads of a row address its key heads. Worker: {@code rows *
     * (heads + keyValueHeads)} groups of {@code headDim} lanes.
     */
    public static void fusedQKRmsNormBatchGroup(
            KernelContext context,
            FloatArray queryBatch,
            FloatArray keyBatch,
            FloatArray queryWeights,
            FloatArray keyWeights,
            int heads,
            int keyValueHeads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int group = context.groupIdx;
        int perRow = heads + keyValueHeads;
        if (group >= perRow * batchInfo.get(1)) {
            return;
        }
        int lane = context.localIdx;
        int row = group / perRow;
        int head = group - row * perRow;
        float[] shared = context.allocateFloatLocalArray(headDim + 1);
        if (head < heads) {
            int index = row * heads * headDim + head * headDim + lane;
            float v = queryBatch.get(index);
            shared[lane] = v;
            context.localBarrier();
            if (lane == 0) {
                float ss = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float x = shared[i];
                    ss += x * x;
                }
                shared[headDim] = 1.0f / TornadoMath.sqrt(ss / headDim + eps);
            }
            context.localBarrier();
            queryBatch.set(index, queryWeights.get(lane) * (shared[headDim] * v));
        } else {
            int index = row * keyValueHeads * headDim + (head - heads) * headDim + lane;
            float v = keyBatch.get(index);
            shared[lane] = v;
            context.localBarrier();
            if (lane == 0) {
                float ss = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float x = shared[i];
                    ss += x * x;
                }
                shared[headDim] = 1.0f / TornadoMath.sqrt(ss / headDim + eps);
            }
            context.localBarrier();
            keyBatch.set(index, keyWeights.get(lane) * (shared[headDim] * v));
        }
    }

    /** Lanes of one row-norm workgroup. */
    public static final int RMS_GROUP_LOCAL = 256;

    /**
     * {@code TransformerBatchPrefillKernels.batchedRmsReduce} with a workgroup per row: the row is
     * staged into shared memory by all lanes, coalesced, and lane zero keeps the left fold in
     * element order over the staged values, so the scale is bit-equal to the one-lane form's.
     * Worker: {@code rows} groups of {@link #RMS_GROUP_LOCAL} lanes; {@code dim} a multiple of it.
     */
    public static void rmsReduceBatchGroup(
            KernelContext context, FloatArray x, FloatArray scale, int dim, float eps) {
        int row = context.groupIdx;
        int lane = context.localIdx;
        int base = row * dim;
        float[] shared = context.allocateFloatLocalArray(dim);
        for (int i = lane; i < dim; i += RMS_GROUP_LOCAL) {
            shared[i] = x.get(base + i);
        }
        context.localBarrier();
        if (lane == 0) {
            float ss = 0.0f;
            for (int i = 0; i < dim; i++) {
                float v = shared[i];
                ss += v * v;
            }
            ss /= dim;
            ss += eps;
            scale.set(row, 1.0f / TornadoMath.sqrt(ss));
        }
    }

    // ---- attention-layer kernels ---------------------------------------------

    /** The interleaved query/gate projection, separated per row. */
    public static void splitQueryGateBatch(
            KernelContext context,
            FloatArray fusedBatch,
            FloatArray queryBatch,
            FloatArray gateBatch,
            int heads,
            int headDim,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        int width = heads * headDim;
        if (lane >= width * batchInfo.get(1)) {
            return;
        }
        int row = lane / width;
        int within = lane - row * width;
        int head = within / headDim;
        int element = within - head * headDim;
        int fusedBase = row * 2 * width + head * 2 * headDim;
        queryBatch.set(lane, fusedBatch.get(fusedBase + element));
        gateBatch.set(lane, fusedBatch.get(fusedBase + headDim + element));
    }

    /**
     * Partial NeoX rotation over a chunk, at each row's own position.
     *
     * <p>{@code batchInfo[0]} is the position of row 0, so row {@code r} rotates at {@code
     * batchInfo[0] + r}. A single scalar position would rotate the whole chunk as though it were
     * one token, which is the defect this signature exists to prevent.
     */
    public static void ropeNeoxPartialBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            FloatArray keyBatch,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int heads,
            int keyValueHeads,
            int headDim,
            int rotaryDim) {
        int lane = context.globalIdx;
        int half = rotaryDim / 2;
        int perRow = heads * half;
        if (lane >= perRow * batchInfo.get(1)) {
            return;
        }
        int row = lane / perRow;
        int within = lane - row * perRow;
        int head = within / half;
        int ic = within - head * half;
        int position = batchInfo.get(0) + row;

        float fcr = freqCisReal.get(position * half + ic);
        float fci = freqCisImag.get(position * half + ic);

        int queryBase = row * heads * headDim + head * headDim;
        float q0 = queryBatch.get(queryBase + ic);
        float q1 = queryBatch.get(queryBase + ic + half);
        queryBatch.set(queryBase + ic, q0 * fcr - q1 * fci);
        queryBatch.set(queryBase + ic + half, q0 * fci + q1 * fcr);

        if (head < keyValueHeads) {
            int keyBase = row * keyValueHeads * headDim + head * headDim;
            float k0 = keyBatch.get(keyBase + ic);
            float k1 = keyBatch.get(keyBase + ic + half);
            keyBatch.set(keyBase + ic, k0 * fcr - k1 * fci);
            keyBatch.set(keyBase + ic + half, k0 * fci + k1 * fcr);
        }
    }

    /**
     * Each row's key and value written into the paged store, at that row's own position.
     *
     * <p>Exactly one write per (row, element): the store is left in the state sequential ingestion
     * would leave it in, which is what decode then reads.
     */
    public static void appendKeyValueBatchPaged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray keyBatch,
            FloatArray valueBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            IntArray blockTable,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride) {
        int lane = context.globalIdx;
        if (lane >= kvDim * batchInfo.get(1)) {
            return;
        }
        int row = lane / kvDim;
        int element = lane - row * kvDim;
        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);

        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        slot,
                        position,
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);
        keyCache.set(cacheOffset + element, keyBatch.get(lane));
        valueCache.set(cacheOffset + element, valueBatch.get(lane));
    }

    /**
     * Causal attention for one (row, head), over the paged store.
     *
     * <p>One workgroup per (row, head), with the query staged and the tiles held in local memory
     * sized from the head width — this family's head is 256 wide, and the shared split-KV kernel's
     * arrays are fixed at 128.
     *
     * <p>The mask is the loop bound. A row reads positions {@code 0..startPos + row} inclusive and
     * cannot reach a later row, whose key/value entries the append above has already written.
     */
    public static void attentionBatchPaged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize) {
        int tid = context.localIdx;
        // The workgroup width as a parameter, not as context.localGroupSizeX: a local array's
        // extent has to be a compile-time constant on CUDA, and a value read from the context is
        // not one ("expression must have a constant value" from nvrtc, on the __shared__ decl).
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int queryBase = row * heads * headSize + head * headSize;
        for (int i = tid; i < headSize; i += localSize) {
            qShared[i] = queryBatch.get(queryBase + i);
        }
        context.localBarrier();

        // Pass 1: this lane's slice of the causal range, tracking a running maximum and sum.
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d);
            }
            score *= invSqrt;
            maxScore = TornadoMath.max(maxScore, score);
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d);
            }
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, over the same range.
        //
        // A position's score does not depend on the output element, so it is computed once per
        // position and shared, rather than once per (position, output element). Writing the loops
        // the other way round — an output element outside, a position inside, a dot product
        // innermost — costs `headSize` times the arithmetic of pass 1 for the same answer, which
        // is what this kernel used to do and what made attention the second-largest item in the
        // prefill profile.
        //
        // Positions are taken a tile at a time: the workgroup computes the tile's weights
        // cooperatively, then every lane sweeps its own output elements over that tile.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += qShared[i] * keyCache.get(base + i);
                }
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d);
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /** {@link #appendKeyValueBatchPaged} into a half-precision store. */
    public static void appendKeyValueBatchFP16Paged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray keyBatch,
            FloatArray valueBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray blockTable,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride) {
        int lane = context.globalIdx;
        if (lane >= kvDim * batchInfo.get(1)) {
            return;
        }
        int row = lane / kvDim;
        int element = lane - row * kvDim;
        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);

        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        slot,
                        position,
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);
        keyCache.set(cacheOffset + element, new HalfFloat(keyBatch.get(lane)));
        valueCache.set(cacheOffset + element, new HalfFloat(valueBatch.get(lane)));
    }

    /**
     * {@link #attentionBatchPaged} over a half-precision store.
     *
     * <p>Entries are widened as they are read and every accumulation stays FP32, so the half
     * precision is in the store and nowhere else.
     */
    public static void attentionBatchFP16Paged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize) {
        int tid = context.localIdx;
        // The workgroup width as a parameter, not as context.localGroupSizeX: a local array's
        // extent has to be a compile-time constant on CUDA, and a value read from the context is
        // not one ("expression must have a constant value" from nvrtc, on the __shared__ decl).
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int queryBase = row * heads * headSize + head * headSize;
        for (int i = tid; i < headSize; i += localSize) {
            qShared[i] = queryBatch.get(queryBase + i);
        }
        context.localBarrier();

        // Pass 1: this lane's slice of the causal range, tracking a running maximum and sum.
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d).getFloat32();
            }
            score *= invSqrt;
            maxScore = TornadoMath.max(maxScore, score);
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d).getFloat32();
            }
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, over the same range.
        //
        // A position's score does not depend on the output element, so it is computed once per
        // position and shared, rather than once per (position, output element). Writing the loops
        // the other way round — an output element outside, a position inside, a dot product
        // innermost — costs `headSize` times the arithmetic of pass 1 for the same answer, which
        // is what this kernel used to do and what made attention the second-largest item in the
        // prefill profile.
        //
        // Positions are taken a tile at a time: the workgroup computes the tile's weights
        // cooperatively, then every lane sweeps its own output elements over that tile.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += qShared[i] * keyCache.get(base + i).getFloat32();
                }
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /**
     * {@link #attentionBatchFP16PagedScoredStaged} with the value pass taking positions 128 at a
     * time instead of 16.
     *
     * <p>Only the third pass changes: its weights array holds 128 exponentials, all 128 lanes
     * compute one each, and a lane sweeps its output elements over the 128 positions of the tile —
     * in increasing position order, the same order the 16-position tiles produce end to end,
     * including the partial last tile — before the next tile is prepared. The two barriers around
     * the shared weights are kept, so they are paid once per 128 positions rather than once per 16.
     * The staged first pass, the stored unscaled scores, the maximum, the denominator, the
     * exponential expression, the causal range, the paged addressing, the head mapping and the grid
     * are the staged kernel's.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScoredStagedWide(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
        int tid = context.localIdx;
        // The workgroup width as a parameter, not as context.localGroupSizeX: a local array's
        // extent has to be a compile-time constant on CUDA, and a value read from the context is
        // not one ("expression must have a constant value" from nvrtc, on the __shared__ decl).
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int queryBase = row * heads * headSize + head * headSize;
        for (int i = tid; i < headSize; i += localSize) {
            qShared[i] = queryBatch.get(queryBase + i);
        }
        context.localBarrier();

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 through the staged tiles: this lane's positions, one per 128-position tile.
        float[] keyTile =
                context.allocateFloatLocalArray(ATTENTION_STAGE_DIMS * ATTENTION_STAGE_LD);
        float maxScore = Float.NEGATIVE_INFINITY;
        int loadPos0 = tid >> 5;
        int loadDim = tid & 31;
        for (int tileStart = 0; tileStart <= position; tileStart += localSize) {
            int p = tileStart + tid;
            float score = 0.0f;
            for (int dimStart = 0; dimStart < headSize; dimStart += ATTENTION_STAGE_DIMS) {
                // Stage: lane tid loads dimension (dimStart + tid % 32) of positions
                // tileStart + tid / 32 + 4i. A warp's 32 lanes read one position's 32
                // consecutive halves.
                for (int i = 0; i < localSize / 4; i++) {
                    int posInTile = loadPos0 + 4 * i;
                    int loadPos = tileStart + posInTile;
                    float value = 0.0f;
                    if (loadPos <= position) {
                        int base =
                                KvBlockAddress.offset(
                                                blockTable,
                                                slot,
                                                loadPos,
                                                layerOff,
                                                kvDim,
                                                blockCfg,
                                                blockStride)
                                        + kvHead * headSize;
                        value = keyCache.get(base + dimStart + loadDim).getFloat32();
                    }
                    keyTile[loadDim * ATTENTION_STAGE_LD + posInTile] = value;
                }
                context.localBarrier();
                if (p <= position) {
                    for (int d = 0; d < ATTENTION_STAGE_DIMS; d++) {
                        score += qShared[dimStart + d] * keyTile[d * ATTENTION_STAGE_LD + tid];
                    }
                }
                context.localBarrier();
            }
            if (p <= position) {
                scores.set(scoreBase + p, score);
                score *= invSqrt;
                maxScore = TornadoMath.max(maxScore, score);
            }
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, a tile of positions at a time, as in the reference.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_VALUE_TILE_WIDE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_VALUE_TILE_WIDE) {
            int tileEnd = tileStart + ATTENTION_VALUE_TILE_WIDE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    // @formatter:off
    /**
     * {@link #attentionBatchFP16PagedScoredStagedWide} with the first pass computed by warps: each
     * causal query-key dot product by one 32-lane warp, no shared key tile, no staging barriers.
     *
     * <p><b>Mapping.</b> One workgroup of 128 lanes per (row, head), as before. Warp {@code w}
     * (lanes {@code 32w..32w+31}) takes the causal positions {@code w, w + 4, w + 8, ...} up to the
     * row's position — a loop bound every lane of the warp shares, so no lane leaves it early. Lane
     * {@code l} owns head dimensions {@code l, l + 32, ..., l + 224}: its eight query elements are
     * read once from the FP32 query and kept in registers; per position it reads the eight FP16
     * keys at those dimensions (a warp reads 64 contiguous bytes per step) and widens each exactly.
     *
     * <p><b>Arithmetic.</b> A lane accumulates its eight products in dimension order in FP32; the
     * warp then folds the 32 partials with five shuffle-down steps (lane {@code l} adds lane {@code
     * l + 16}, then {@code l + 8}, ... ) and lane zero stores the unscaled sum to {@code scores}
     * and folds {@code sum * invSqrt} into its running maximum. This is the reference's sum of the
     * same 256 products in a different order — 32 partials of 8 in place of one running sum — so
     * the score is an FP32 reassociation of the reference's, not bit-equal to it. The other 31
     * lanes of a warp contribute nothing to the maximum (they enter the workgroup reduction at
     * negative infinity); the workgroup maximum, the denominator, the exponential expression, the
     * 128-position value pass, the causal range, the paged addressing, the head mapping, the scores
     * scratch and the output are the wide kernel's, unchanged.
     *
     * <p>Requires a 256-wide head and 128-lane workgroups, and the backend whose warp shuffle is
     * verified (CUDA); the dispatch keeps the wide kernel for everything else.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScoredWarp(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
        int tid = context.localIdx;
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 by warps: this lane's eight query elements, then this warp's positions.
        int warp = tid >> 5;
        int lane = tid & 31;
        int queryBase = row * heads * headSize + head * headSize;
        float q0 = queryBatch.get(queryBase + lane);
        float q1 = queryBatch.get(queryBase + lane + 32);
        float q2 = queryBatch.get(queryBase + lane + 64);
        float q3 = queryBatch.get(queryBase + lane + 96);
        float q4 = queryBatch.get(queryBase + lane + 128);
        float q5 = queryBatch.get(queryBase + lane + 160);
        float q6 = queryBatch.get(queryBase + lane + 192);
        float q7 = queryBatch.get(queryBase + lane + 224);
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = warp; p <= position; p += ATTENTION_WARPS) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize
                            + lane;
            float partial = q0 * keyCache.get(base).getFloat32();
            partial += q1 * keyCache.get(base + 32).getFloat32();
            partial += q2 * keyCache.get(base + 64).getFloat32();
            partial += q3 * keyCache.get(base + 96).getFloat32();
            partial += q4 * keyCache.get(base + 128).getFloat32();
            partial += q5 * keyCache.get(base + 160).getFloat32();
            partial += q6 * keyCache.get(base + 192).getFloat32();
            partial += q7 * keyCache.get(base + 224).getFloat32();
            float score = warpSumBroadcast(context, partial);
            if (lane == 0) {
                scores.set(scoreBase + p, score);
                maxScore = TornadoMath.max(maxScore, score * invSqrt);
            }
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: the weighted value sum, 128 positions at a time, as in the wide kernel.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_VALUE_TILE_WIDE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_VALUE_TILE_WIDE) {
            int tileEnd = tileStart + ATTENTION_VALUE_TILE_WIDE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /**
     * Whether the warp first pass fits the geometry: a 256-wide head (eight dimensions a lane over
     * 32 lanes) and 128-lane workgroups (four warps).
     */
    public static boolean attentionWarpEligible(int headSize, int localSize) {
        return headSize == ATTENTION_WARP_HEAD && localSize == ATTENTION_WARPS * 32;
    }

    // ---- tensor-core attention -----------------------------------------------------------------

    /** Queries one workgroup of the tensor-core attention owns: one MMA row tile. */
    private static final int TC_QUERIES = 16;

    /**
     * Keys one staged tile holds: 32, so that the K, V, Q and P tiles fit the 48 KiB static limit.
     */
    private static final int TC_KEYS = 32;

    /** Head width the kernel is written for. */
    private static final int TC_HEAD = 256;

    /** Lanes of the 16-query kernel: four warps. */
    private static final int TC_LANES = 128;

    /** Halves of the FP16 staging scratch one workgroup owns: its Q tile and its P tile. */
    public static final int TC_STAGE_HALVES = TC_QUERIES * TC_HEAD + TC_QUERIES * TC_KEYS;

    /** Keys a transposed score region holds: the capacity rounded up to whole key tiles. */
    public static int tcScoreKeys(int capacity) {
        return (capacity + TC_KEYS - 1) / TC_KEYS * TC_KEYS;
    }

    /** Ints of one key or value tile: 32 keys by 256 halves. */
    private static final int TC_KV_TILE_INTS = TC_KEYS * TC_HEAD / 2;

    /**
     * Issues the {@code cp.async} copies of key tile {@code tileStart} into {@code kvTile} at
     * {@code buf}, as two A blocks of 16 keys by 16 dim steps; keys past {@code lastKey} come from
     * {@code lastKey}'s row. Not committed.
     */
    private static void stageKeyTile(
            KernelContext context,
            int[] kvTile,
            int buf,
            HalfFloatArray keyCache,
            IntArray blockTable,
            int slot,
            int layerOff,
            int kvDim,
            int blockCfg,
            int blockStride,
            int kvHead,
            int headSize,
            int tileStart,
            int lastKey,
            int tid) {
        for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC_LANES; i++) {
            int e = i * TC_LANES + tid;
            int key = e >> 7;
            int d = (e & 127) << 1;
            int p = tileStart + key;
            if (p > lastKey) {
                p = lastKey;
            }
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            int dst =
                    buf
                            + (((key >> 4) << 4) + (d >> 4)) * 128
                            + ((key & 15) << 3)
                            + ((d & 15) >> 1);
            context.asyncCopyToLocal(kvTile, dst, keyCache, base + d);
        }
    }

    /** Issues the copies of value tile {@code tileStart} into {@code kvTile} at {@code buf}. */
    private static void stageValueTile(
            KernelContext context,
            int[] kvTile,
            int buf,
            HalfFloatArray valueCache,
            IntArray blockTable,
            int slot,
            int layerOff,
            int kvDim,
            int blockCfg,
            int blockStride,
            int kvHead,
            int headSize,
            int tileStart,
            int lastKey,
            int tid) {
        for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC_LANES; i++) {
            int e = i * TC_LANES + tid;
            int key = e >> 7;
            int d = (e & 127) << 1;
            int p = tileStart + key;
            if (p > lastKey) {
                p = lastKey;
            }
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            int dst =
                    buf + (((key >> 4) << 5) + (d >> 3)) * 64 + ((key & 15) << 2) + ((d & 7) >> 1);
            context.asyncCopyToLocal(kvTile, dst, valueCache, base + d);
        }
    }

    // @formatter:off
    /**
     * Batched FP16-KV attention on the tensor cores, 16 queries per four-warp workgroup: {@code S^T
     * = K Q^T} and {@code O = P V} as {@code m16n8k16} FP16 MMAs with FP32 accumulators, in three
     * passes over the causal range — scores, softmax statistics, then {@code P V} against the
     * settled maximum. The 32-query form {@link #attentionBatchFP16PagedTensorCoreT32} is
     * dispatched where the width divides into 32; this one serves widths that divide only into 16.
     *
     * <p><b>Equations.</b> {@code m = max_k s_k / sqrt(d)}, {@code l = sum_k exp(s_k / sqrt(d) -
     * m)}, {@code O = (sum_k p_k v_k) / l}. Queries are rounded to FP16 <i>unscaled</i>, the {@code
     * 1/sqrt(256)} applied in FP32 to the accumulated score; keys and values are the stored FP16;
     * scores, maxima, sums, exponentials and the output accumulators are FP32; each probability is
     * rounded to FP16 for the {@code P V} MMA while the denominator sums the FP32 probabilities. No
     * online softmax: an MMA accumulator cannot be rescaled in registers, so the scores go through
     * the score scratch.
     *
     * <p><b>Score region.</b> Workgroup {@code (queryTile, head)} owns the region {@code (queryTile
     * * heads + head) * tcScoreKeys(capacity) * 16} of the score scratch: {@code
     * tcScoreKeys(capacity)} rows of 16 queries, row {@code p} holding the unscaled FP32 scores of
     * key {@code p} against the tile's 16 queries. The MMA store writes 16 keys by 8 queries per
     * warp, so the region is padded to whole key tiles; the scratch is sized {@code rows * heads *
     * tcScoreKeys(capacity)} floats. Nothing outside the kernel reads the region.
     *
     * <p><b>Pass 1.</b> Per 32-key tile: {@code kvTile} (int[4096], the values' tile in pass 3)
     * takes the 32 key rows as two A blocks of 16 keys, 16 dims a step, by {@code cp.async},
     * double-buffered so the next tile's copy overlaps this tile's MMAs; warp {@code w} computes
     * keys {@code 16 (w / 2) ..} against queries {@code 8 (w % 2) ..} over the 16 dimension steps
     * and stores the 16 x 8 tile to the region. {@code qTile} (HalfFloat[4096]): Q^T as the
     * swizzled B operand, sub-tile {@code 2 t + g} = dims {@code 16 t ..} by queries {@code 8 g
     * ..}, staged once from the FP32 queries.
     *
     * <p><b>Pass 2.</b> Lane {@code tid}: query {@code tid % 16}, part {@code tid / 16} of eight;
     * the part walks keys {@code part, part + 8, ...} of its query (sixteen lanes read sixteen
     * consecutive floats of a key row), then the eight partials of a query are folded {@code ((p0 +
     * p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))}.
     *
     * <p><b>Pass 3.</b> Per 32-key tile: the value tile by {@code cp.async} (double-buffered), the
     * probabilities computed from the region into the FP16 staging scratch as {@code P[query][key]}
     * (lane {@code i}: key {@code i / 16}, query {@code i % 16}), copied back as the A operand,
     * then two 16-key steps of eight 16 x 8 MMAs per warp (warp {@code w}: output dimensions {@code
     * 64 w ..}); the outputs are stored and divided by the denominator.
     *
     * <p><b>Masking and partial tiles.</b> Key {@code k} is valid for query row {@code q} iff
     * {@code k <= position(q) = startPos + 16 (g / heads) + q}; an invalid probability is zero. The
     * key-tile loops run to the tile's largest position, a workgroup-uniform bound; rows at or past
     * the chunk's active count compute at their nominal position and land in padding rows nothing
     * consumes. Keys past the last valid key are staged from that key's row and always masked. Head
     * {@code h} reads key/value head {@code h / kvMul}; each key row is located by {@code
     * KvBlockAddress.offset}, so pages need not be contiguous.
     *
     * <p>Requires the chunk width a multiple of 16, a 256-wide head, 128-lane workgroups, and a
     * staging scratch of {@code (rows / 16) * heads * TC_STAGE_HALVES} halves. Worker: {@code (rows
     * / 16) * heads * 128} lanes, local 128; {@code scoreStride} is the capacity.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedTensorCoreT(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride,
            HalfFloatArray stage) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int queryTile = group / heads;
        int head = group - queryTile * heads;
        int rowBase = queryTile * TC_QUERIES;
        if (rowBase >= batchInfo.get(1)) {
            return;
        }
        int startPos = batchInfo.get(0);
        int slot = batchInfo.get(2);
        int capacity = scoreStride;
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        HalfFloat[] qTile = context.allocateHalfFloatLocalArray(TC_QUERIES * TC_HEAD);
        // Two buffers, holding key tiles in pass 1 and value tiles in pass 3 (never both): the
        // next tile's copy lands in the buffer this tile is not reading.
        int[] kvTile = context.allocateIntLocalArray(2 * TC_KV_TILE_INTS);
        int[] pTile = context.allocateIntLocalArray(TC_QUERIES * TC_KEYS / 2);
        float[] rowStat = context.allocateFloatLocalArray(TC_QUERIES * 2 + TC_LANES);

        int stageBase = group * TC_STAGE_HALVES;
        int qBase = (rowBase * heads + head) * headSize;
        // Q^T as the swizzled B operand: sub-tile (dims 16 t .., queries 8 g ..) at 2 t + g.
        for (int i = tid; i < TC_QUERIES * TC_HEAD; i += TC_LANES) {
            int row = i >> 8;
            int d = i & 255;
            context.mmaStoreBSwizzled(
                    qTile,
                    d & 15,
                    row & 7,
                    8,
                    new HalfFloat(queryBatch.get(qBase + row * heads * headSize + d)),
                    (((d >> 4) << 1) + (row >> 3)) * 256);
        }
        context.localBarrier();

        int maxPos = startPos + rowBase + TC_QUERIES - 1;
        int lastKey = capacity - 1;
        if (maxPos < lastKey) {
            lastKey = maxPos;
        }
        // The region: its first row, in rows of 16 floats.
        int regionRow = (queryTile * heads + head) * tcScoreKeys(capacity);
        int regionBase = regionRow * TC_QUERIES;
        int keyBlock = warp >> 1;
        int qBlock = warp & 1;

        // Pass 1: S^T = K Q^T, 32 keys a tile, stored unscaled to the region. The tile after this
        // one is copied while this one is multiplied.
        stageKeyTile(
                context,
                kvTile,
                0,
                keyCache,
                blockTable,
                slot,
                layerOff,
                kvDim,
                blockCfg,
                blockStride,
                kvHead,
                headSize,
                0,
                lastKey,
                tid);
        context.asyncCopyCommit();
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
            int bufThis = ((tileStart / TC_KEYS) & 1) * TC_KV_TILE_INTS;
            int bufNext = TC_KV_TILE_INTS - bufThis;
            if (tileStart + TC_KEYS <= lastKey) {
                stageKeyTile(
                        context,
                        kvTile,
                        bufNext,
                        keyCache,
                        blockTable,
                        slot,
                        layerOff,
                        kvDim,
                        blockCfg,
                        blockStride,
                        kvHead,
                        headSize,
                        tileStart + TC_KEYS,
                        lastKey,
                        tid);
                context.asyncCopyCommit();
                context.asyncCopyWaitGroup(1);
            } else {
                context.asyncCopyWaitGroup(0);
            }
            context.localBarrier();
            float[] s0 = context.mmaFragment(0.0f);
            for (int t = 0; t < TC_HEAD / 16; t++) {
                HalfFloat[] a =
                        context.mmaLoadA(
                                kvTile, 16, (bufThis >> 7) * 512 + ((keyBlock << 4) + t) * 512);
                HalfFloat[] b0 = context.mmaLoadBSwizzled(qTile, 16, ((t << 1) + qBlock) * 256);
                s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
            }
            context.mmaStore(
                    s0, scores, regionRow + tileStart + (keyBlock << 4), qBlock << 3, TC_QUERIES);
            context.localBarrier();
        }

        // Pass 2: per query, the maximum and the denominator of the scaled scores; lane = (query
        // tid % 16, part tid / 16), the parts folded in the shuffle tree's order.
        int statRow = tid & 15;
        int statPart = tid >> 4;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statPart; p <= statPos; p += 8) {
            rowMax = TornadoMath.max(rowMax, scores.get(regionBase + (p << 4) + statRow) * invSqrt);
        }
        rowStat[TC_QUERIES * 2 + tid] = rowMax;
        context.localBarrier();
        if (tid < TC_QUERIES) {
            int b = TC_QUERIES * 2 + tid;
            float m04 = TornadoMath.max(rowStat[b], rowStat[b + 64]);
            float m26 = TornadoMath.max(rowStat[b + 32], rowStat[b + 96]);
            float m15 = TornadoMath.max(rowStat[b + 16], rowStat[b + 80]);
            float m37 = TornadoMath.max(rowStat[b + 48], rowStat[b + 112]);
            rowStat[tid] = TornadoMath.max(TornadoMath.max(m04, m26), TornadoMath.max(m15, m37));
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statPart; p <= statPos; p += 8) {
            rowSum += TornadoMath.exp(scores.get(regionBase + (p << 4) + statRow) * invSqrt - m);
        }
        context.localBarrier();
        rowStat[TC_QUERIES * 2 + tid] = rowSum;
        context.localBarrier();
        if (tid < TC_QUERIES) {
            int b = TC_QUERIES * 2 + tid;
            float s04 = rowStat[b] + rowStat[b + 64];
            float s26 = rowStat[b + 32] + rowStat[b + 96];
            float s15 = rowStat[b + 16] + rowStat[b + 80];
            float s37 = rowStat[b + 48] + rowStat[b + 112];
            rowStat[TC_QUERIES + tid] = (s04 + s26) + (s15 + s37);
        }
        context.localBarrier();

        // Pass 3: O = P V, 32 keys a tile, against the settled maximum.
        float[] o0 = context.mmaFragment(0.0f);
        float[] o1 = context.mmaFragment(0.0f);
        float[] o2 = context.mmaFragment(0.0f);
        float[] o3 = context.mmaFragment(0.0f);
        float[] o4 = context.mmaFragment(0.0f);
        float[] o5 = context.mmaFragment(0.0f);
        float[] o6 = context.mmaFragment(0.0f);
        float[] o7 = context.mmaFragment(0.0f);
        int pBase = stageBase + TC_QUERIES * TC_HEAD;
        stageValueTile(
                context,
                kvTile,
                0,
                valueCache,
                blockTable,
                slot,
                layerOff,
                kvDim,
                blockCfg,
                blockStride,
                kvHead,
                headSize,
                0,
                lastKey,
                tid);
        context.asyncCopyCommit();
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
            int bufThis = ((tileStart / TC_KEYS) & 1) * TC_KV_TILE_INTS;
            int bufNext = TC_KV_TILE_INTS - bufThis;
            boolean hasNext = tileStart + TC_KEYS <= lastKey;
            // P for this tile: lane covers (key = i / 16, query = i % 16), 4 per lane, as halves
            // into P[query][key]; copied back as one group, then the next tile's values as the
            // group after it, so waiting for all but one group leaves only that copy in flight.
            for (int i = tid; i < TC_QUERIES * TC_KEYS; i += TC_LANES) {
                int key = i >> 4;
                int row = i & 15;
                int p = tileStart + key;
                float prob = 0.0f;
                if (p <= startPos + rowBase + row) {
                    prob =
                            TornadoMath.exp(
                                    scores.get(regionBase + (p << 4) + row) * invSqrt
                                            - rowStat[row]);
                }
                stage.set(pBase + (row << 5) + key, new HalfFloat(prob));
            }
            context.localBarrier();
            for (int i = tid; i < TC_QUERIES * TC_KEYS / 2; i += TC_LANES) {
                int t = i >> 7;
                int q = (i >> 3) & 15;
                int kk = (t << 4) + ((i & 7) << 1);
                context.asyncCopyToLocal(pTile, i, stage, pBase + (q << 5) + kk);
            }
            context.asyncCopyCommit();
            if (hasNext) {
                stageValueTile(
                        context,
                        kvTile,
                        bufNext,
                        valueCache,
                        blockTable,
                        slot,
                        layerOff,
                        kvDim,
                        blockCfg,
                        blockStride,
                        kvHead,
                        headSize,
                        tileStart + TC_KEYS,
                        lastKey,
                        tid);
                context.asyncCopyCommit();
                context.asyncCopyWaitGroup(1);
            } else {
                context.asyncCopyWaitGroup(0);
            }
            context.localBarrier();
            int vOff = bufThis >> 6;
            for (int t = 0; t < TC_KEYS / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(pTile, 16, t * 512);
                int vBase = vOff + (t << 5) + (warp << 3);
                o0 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 0) * 256),
                                o0,
                                MMAShape.M16N8K16);
                o1 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 1) * 256),
                                o1,
                                MMAShape.M16N8K16);
                o2 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 2) * 256),
                                o2,
                                MMAShape.M16N8K16);
                o3 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 3) * 256),
                                o3,
                                MMAShape.M16N8K16);
                o4 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 4) * 256),
                                o4,
                                MMAShape.M16N8K16);
                o5 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 5) * 256),
                                o5,
                                MMAShape.M16N8K16);
                o6 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 6) * 256),
                                o6,
                                MMAShape.M16N8K16);
                o7 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 7) * 256),
                                o7,
                                MMAShape.M16N8K16);
            }
            context.localBarrier();
        }

        int ld = heads * headSize;
        int colBase = head * headSize + (warp << 6);
        context.mmaStore(o0, outBatch, rowBase, colBase, ld);
        context.mmaStore(o1, outBatch, rowBase, colBase + 8, ld);
        context.mmaStore(o2, outBatch, rowBase, colBase + 16, ld);
        context.mmaStore(o3, outBatch, rowBase, colBase + 24, ld);
        context.mmaStore(o4, outBatch, rowBase, colBase + 32, ld);
        context.mmaStore(o5, outBatch, rowBase, colBase + 40, ld);
        context.mmaStore(o6, outBatch, rowBase, colBase + 48, ld);
        context.mmaStore(o7, outBatch, rowBase, colBase + 56, ld);
        context.localBarrier();
        // Divide by the denominator: lane covers (row = i / 256, dim = i % 256).
        for (int i = tid; i < TC_QUERIES * TC_HEAD; i += TC_LANES) {
            int row = i >> 8;
            int idx = (rowBase + row) * ld + head * headSize + (i & 255);
            outBatch.set(idx, outBatch.get(idx) / rowStat[TC_QUERIES + row]);
        }
    }

    /** Queries one workgroup of the 32-query tensor-core attention owns: two MMA row tiles. */
    public static final int TC32_QUERIES = 32;

    /** Lanes of the 32-query kernel: eight warps. */
    public static final int TC32_LANES = 256;

    /** Halves of the FP16 staging scratch one 32-query workgroup owns: its P tile. */
    public static final int TC32_STAGE_HALVES = TC32_QUERIES * TC_KEYS;

    // @formatter:off
    /**
     * {@link #attentionBatchFP16PagedTensorCoreT} with 32 queries per workgroup and eight warps
     * (the dispatched form wherever the width divides into 32): every staged key or value tile
     * serves twice the queries, so the copies per multiply-add halve, and two workgroups an SM hold
     * sixteen warps instead of eight.
     *
     * <p>Workgroup {@code g}: query tile {@code g / heads} (rows {@code 32 (g / heads) .. + 31}),
     * head {@code g % heads}. Pass 1: warp {@code w} computes keys {@code 16 (w / 4) ..} against
     * queries {@code 8 (w % 4) ..} per 32-key tile. Pass 2: lane {@code tid} is query {@code tid %
     * 32}, part {@code tid / 32} of eight, folded in the eight-lane shuffle tree's order. Pass 3:
     * warp {@code w} accumulates query rows {@code 16 (w / 4) ..} by dimensions {@code 64 (w % 4)
     * ..}. The score region is {@code (queryTile * heads + head) * tcScoreKeys(capacity) * 32}
     * floats, 32 queries a key row; the scratch is sized {@code rows * heads *
     * tcScoreKeys(capacity)} as before. The queries are rounded to FP16 unscaled and the
     * probabilities to FP16, as before; each output is the 16-query kernel's bit for bit (the same
     * products in the same order, the same folds).
     *
     * <p>Shared: Q^T (16 KiB, swizzled B sub-tiles {@code 4 t + g}), one key/value tile (16 KiB), P
     * (2 KiB) and the statistics. Requires the chunk width a multiple of 32, a 256-wide head,
     * 256-lane workgroups and a staging scratch of {@code (rows / 32) * heads * TC32_STAGE_HALVES}
     * halves. Worker: {@code (rows / 32) * heads * 256} lanes, local 256.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedTensorCoreT32(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride,
            HalfFloatArray stage) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int group = context.groupIdx;
        int queryTile = group / heads;
        int head = group - queryTile * heads;
        int rowBase = queryTile * TC32_QUERIES;
        if (rowBase >= batchInfo.get(1)) {
            return;
        }
        int startPos = batchInfo.get(0);
        int slot = batchInfo.get(2);
        int capacity = scoreStride;
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        HalfFloat[] qTile = context.allocateHalfFloatLocalArray(TC32_QUERIES * TC_HEAD);
        int[] kvTile = context.allocateIntLocalArray(TC_KV_TILE_INTS);
        int[] pTile = context.allocateIntLocalArray(TC32_QUERIES * TC_KEYS / 2);
        float[] rowStat = context.allocateFloatLocalArray(TC32_QUERIES * 2 + TC32_LANES);

        int pBase = group * TC32_STAGE_HALVES;
        int qBase = (rowBase * heads + head) * headSize;
        // Q^T as the swizzled B operand: sub-tile (dims 16 t .., queries 8 g ..) at 4 t + g.
        for (int i = tid; i < TC32_QUERIES * TC_HEAD; i += TC32_LANES) {
            int row = i >> 8;
            int d = i & 255;
            context.mmaStoreBSwizzled(
                    qTile,
                    d & 15,
                    row & 7,
                    8,
                    new HalfFloat(queryBatch.get(qBase + row * heads * headSize + d)),
                    (((d >> 4) << 2) + (row >> 3)) * 256);
        }
        context.localBarrier();

        int maxPos = startPos + rowBase + TC32_QUERIES - 1;
        int lastKey = capacity - 1;
        if (maxPos < lastKey) {
            lastKey = maxPos;
        }
        int regionRow = (queryTile * heads + head) * tcScoreKeys(capacity);
        int regionBase = regionRow * TC32_QUERIES;
        int keyBlock = warp >> 2;
        int qBlock = warp & 3;

        // Pass 1: S^T = K Q^T, 32 keys a tile, stored unscaled to the region.
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
            for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC32_LANES; i++) {
                int e = i * TC32_LANES + tid;
                int key = e >> 7;
                int d = (e & 127) << 1;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                int dst =
                        (((key >> 4) << 4) + (d >> 4)) * 128 + ((key & 15) << 3) + ((d & 15) >> 1);
                context.asyncCopyToLocal(kvTile, dst, keyCache, base + d);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(0);
            context.localBarrier();
            float[] s0 = context.mmaFragment(0.0f);
            for (int t = 0; t < TC_HEAD / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(kvTile, 16, ((keyBlock << 4) + t) * 512);
                HalfFloat[] b0 = context.mmaLoadBSwizzled(qTile, 16, ((t << 2) + qBlock) * 256);
                s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
            }
            context.mmaStore(
                    s0, scores, regionRow + tileStart + (keyBlock << 4), qBlock << 3, TC32_QUERIES);
            context.localBarrier();
        }

        // Pass 2: per query, the maximum and the denominator; lane = (query tid % 32, part
        // tid / 32), the parts folded in the shuffle tree's order.
        int statRow = tid & 31;
        int statPart = tid >> 5;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statPart; p <= statPos; p += 8) {
            rowMax = TornadoMath.max(rowMax, scores.get(regionBase + (p << 5) + statRow) * invSqrt);
        }
        rowStat[TC32_QUERIES * 2 + tid] = rowMax;
        context.localBarrier();
        if (tid < TC32_QUERIES) {
            int b = TC32_QUERIES * 2 + tid;
            float m04 = TornadoMath.max(rowStat[b], rowStat[b + 128]);
            float m26 = TornadoMath.max(rowStat[b + 64], rowStat[b + 192]);
            float m15 = TornadoMath.max(rowStat[b + 32], rowStat[b + 160]);
            float m37 = TornadoMath.max(rowStat[b + 96], rowStat[b + 224]);
            rowStat[tid] = TornadoMath.max(TornadoMath.max(m04, m26), TornadoMath.max(m15, m37));
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statPart; p <= statPos; p += 8) {
            rowSum += TornadoMath.exp(scores.get(regionBase + (p << 5) + statRow) * invSqrt - m);
        }
        context.localBarrier();
        rowStat[TC32_QUERIES * 2 + tid] = rowSum;
        context.localBarrier();
        if (tid < TC32_QUERIES) {
            int b = TC32_QUERIES * 2 + tid;
            float s04 = rowStat[b] + rowStat[b + 128];
            float s26 = rowStat[b + 64] + rowStat[b + 192];
            float s15 = rowStat[b + 32] + rowStat[b + 160];
            float s37 = rowStat[b + 96] + rowStat[b + 224];
            rowStat[TC32_QUERIES + tid] = (s04 + s26) + (s15 + s37);
        }
        context.localBarrier();

        // Pass 3: O = P V, 32 keys a tile; warp w: query rows 16 (w / 4) .., dims 64 (w % 4) ..
        float[] o0 = context.mmaFragment(0.0f);
        float[] o1 = context.mmaFragment(0.0f);
        float[] o2 = context.mmaFragment(0.0f);
        float[] o3 = context.mmaFragment(0.0f);
        float[] o4 = context.mmaFragment(0.0f);
        float[] o5 = context.mmaFragment(0.0f);
        float[] o6 = context.mmaFragment(0.0f);
        float[] o7 = context.mmaFragment(0.0f);
        int rowTile = warp >> 2;
        int dimBlock = warp & 3;
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
            for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC32_LANES; i++) {
                int e = i * TC32_LANES + tid;
                int key = e >> 7;
                int d = (e & 127) << 1;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                int dst = (((key >> 4) << 5) + (d >> 3)) * 64 + ((key & 15) << 2) + ((d & 7) >> 1);
                context.asyncCopyToLocal(kvTile, dst, valueCache, base + d);
            }
            context.asyncCopyCommit();
            // P for this tile: lane covers (key = i / 32, query = i % 32), 4 per lane, as halves
            // into P[query][key].
            for (int i = tid; i < TC32_QUERIES * TC_KEYS; i += TC32_LANES) {
                int key = i >> 5;
                int row = i & 31;
                int p = tileStart + key;
                float prob = 0.0f;
                if (p <= startPos + rowBase + row) {
                    prob =
                            TornadoMath.exp(
                                    scores.get(regionBase + (p << 5) + row) * invSqrt
                                            - rowStat[row]);
                }
                stage.set(pBase + (row << 5) + key, new HalfFloat(prob));
            }
            context.localBarrier();
            // P as two A row tiles of [16 queries][32 keys]: block (rowTile r, key step t) at
            // (2 r + t) * 128 ints, 8 ints a row.
            for (int i = tid; i < TC32_QUERIES * TC_KEYS / 2; i += TC32_LANES) {
                int blk = i >> 7;
                int q = ((blk >> 1) << 4) + ((i >> 3) & 15);
                int kk = ((blk & 1) << 4) + ((i & 7) << 1);
                context.asyncCopyToLocal(pTile, i, stage, pBase + (q << 5) + kk);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(0);
            context.localBarrier();
            for (int t = 0; t < TC_KEYS / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(pTile, 16, ((rowTile << 1) + t) * 512);
                int vBase = (t << 5) + (dimBlock << 3);
                o0 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 0) * 256),
                                o0,
                                MMAShape.M16N8K16);
                o1 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 1) * 256),
                                o1,
                                MMAShape.M16N8K16);
                o2 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 2) * 256),
                                o2,
                                MMAShape.M16N8K16);
                o3 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 3) * 256),
                                o3,
                                MMAShape.M16N8K16);
                o4 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 4) * 256),
                                o4,
                                MMAShape.M16N8K16);
                o5 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 5) * 256),
                                o5,
                                MMAShape.M16N8K16);
                o6 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 6) * 256),
                                o6,
                                MMAShape.M16N8K16);
                o7 =
                        context.mma(
                                a,
                                context.mmaLoadB(kvTile, 16, (vBase + 7) * 256),
                                o7,
                                MMAShape.M16N8K16);
            }
            context.localBarrier();
        }

        int ld = heads * headSize;
        int outRow = rowBase + (rowTile << 4);
        int colBase = head * headSize + (dimBlock << 6);
        context.mmaStore(o0, outBatch, outRow, colBase, ld);
        context.mmaStore(o1, outBatch, outRow, colBase + 8, ld);
        context.mmaStore(o2, outBatch, outRow, colBase + 16, ld);
        context.mmaStore(o3, outBatch, outRow, colBase + 24, ld);
        context.mmaStore(o4, outBatch, outRow, colBase + 32, ld);
        context.mmaStore(o5, outBatch, outRow, colBase + 40, ld);
        context.mmaStore(o6, outBatch, outRow, colBase + 48, ld);
        context.mmaStore(o7, outBatch, outRow, colBase + 56, ld);
        context.localBarrier();
        // Divide by the denominator: lane covers (row = i / 256, dim = i % 256).
        for (int i = tid; i < TC32_QUERIES * TC_HEAD; i += TC32_LANES) {
            int row = i >> 8;
            int idx = (rowBase + row) * ld + head * headSize + (i & 255);
            outBatch.set(idx, outBatch.get(idx) / rowStat[TC32_QUERIES + row]);
        }
    }

    /**
     * Whether the tensor-core attention fits the geometry: 256-wide head, 128 lanes, a chunk of
     * whole 16-query tiles, a context capacity of whole 8-key score sub-tiles.
     */
    public static boolean attentionTensorCoreEligible(
            int headSize, int localSize, int rows, int capacity) {
        return headSize == TC_HEAD
                && localSize == TC_LANES
                && rows % TC_QUERIES == 0
                && capacity % 8 == 0;
    }

    /** The attention result gated by the logistic of its gate, over a chunk. */
    public static void applyOutputGateBatch(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            int count,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        float g = gate.get(lane);
        values.set(lane, values.get(lane) * (1.0f / (1.0f + TornadoMath.exp(-g))));
    }

    /**
     * The per-head query and key RMS norms over a chunk, one lane per (row, head).
     *
     * <p>A lane rather than a workgroup with a reduction: the head is 256 wide and there are 28 of
     * them per row, so the serial sum costs less than the barriers would. The learned scales are
     * one head wide and shared by every head and every row.
     *
     * @param heads query heads; lanes past them address key heads
     */
    public static void fusedQKRmsNormBatch(
            KernelContext context,
            FloatArray queryBatch,
            FloatArray keyBatch,
            FloatArray queryWeights,
            FloatArray keyWeights,
            int heads,
            int keyValueHeads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        int perRow = heads + keyValueHeads;
        if (lane >= perRow * batchInfo.get(1)) {
            return;
        }
        int row = lane / perRow;
        int head = lane - row * perRow;

        if (head < heads) {
            normalizeHead(
                    queryBatch, queryWeights, row * heads * headDim + head * headDim, headDim, eps);
        } else {
            int keyHead = head - heads;
            normalizeHead(
                    keyBatch,
                    keyWeights,
                    row * keyValueHeads * headDim + keyHead * headDim,
                    headDim,
                    eps);
        }
    }

    private static void normalizeHead(
            FloatArray values, FloatArray weights, int base, int headDim, float eps) {
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, weights.get(i) * (inv * values.get(base + i)));
        }
    }
}
