package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

// @formatter:off
/**
 * Reference forms of the Qwen3.8 batched-prefill kernels, kept for the tests only.
 *
 * <p>Each is the plainer kernel a production kernel was proved raw-bit equal to when it replaced
 * it, with its arithmetic in the simplest order: one lane per element or per output, row-major
 * scratch, no staging tricks. They are not dispatched by the engine and are not tuned; they exist
 * so that a production kernel is never its own oracle. Constants and helpers are copies of the
 * production ones they were written against.
 *
 * <ul>
 *   <li>{@link #projectionMMAQ4_0} / {@link #projectionMMAQ5_K}: the direct tensor-core
 *       projections, one warp per 16 x 8 tile, weights decoded into shared per round.
 *   <li>{@link #dequantizeQ4_0ToFP16} / {@link #dequantizeQ4_1ToFP16} / {@link
 *       #dequantizeQ5_KToFP16}: row-major decoders, a lane per element; {@link
 *       #dequantizeQ4_0ToFP16Tiled}: the tiled layout with a lane per element.
 *   <li>{@link #causalConv1dBatch}: the convolution without the SiLU and split folded in.
 *   <li>{@link #processHeadsFlashAttentionSplitKVFP16PagedWideHead}: the split-KV decode attention
 *       with a lane per position and a shared accumulator row per lane.
 *   <li>{@link #attentionBatchFP16PagedScoredStaged}: the scored attention with a staged key tile,
 *       128 lanes; {@link #attentionBatchFP16PagedTensorCore}: the first tensor-core attention, 16
 *       queries, K^T restaged per tile, scores in the [row][head][key] scratch.
 * </ul>
 */
// @formatter:on
public final class Qwen35ReferenceKernels {

    private Qwen35ReferenceKernels() {}

    // ---- constants of the projection kernels ---------------------------------------------------

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 18;
    private static final int WARP_SIZE = 32;
    private static final int PANEL = 8;
    static final int BM = Qwen35MMAKernels.BM;
    static final int BN = Qwen35MMAKernels.BN;
    private static final int BK = 16;
    private static final int B_SUBTILE_BYTES = 256;
    private static final int A_SUBTILE_BYTES = BM * BK * 2;
    private static final int QK_K = 256;
    private static final int K_BLOCK_BYTES = 176;
    private static final int K_SCALES_OFFSET = 4;
    private static final int K_QH_OFFSET = 16;
    private static final int K_QS_OFFSET = 48;
    private static final int BLOCK_BYTES_Q4_1 = 20;
    private static final int GEMM_BN = 128;
    private static final int GEMM_BK = 16;
    private static final int GEMM_B_TILE_INTS = GEMM_BK * GEMM_BN / 2;

    // ---- constants of the attention kernels ----------------------------------------------------

    private static final int ATTENTION_TILE = 16;
    private static final int ATTENTION_SLOTS = 4;
    private static final int ATTENTION_STAGE_DIMS = 32;
    private static final int ATTENTION_STAGE_LANES = Qwen35BatchKernels.ATTENTION_STAGE_LANES;
    private static final int ATTENTION_STAGE_LD = 129;
    private static final int TC_QUERIES = 16;
    private static final int TC_KEYS = 32;
    private static final int TC_HEAD = 256;
    private static final int TC_LANES = 128;
    public static final int TC_STAGE_HALVES = Qwen35BatchKernels.TC_STAGE_HALVES;

    /**
     * The fp16 at {@code index}, from two byte loads. See {@code TransformerComputeKernelsQ5_K}.
     */
    private static float halfFromBytes(ByteArray w, int index) {
        int lo = w.get(index) & 0xFF;
        int hi = w.get(index + 1) & 0xFF;
        int h = (hi << 8) | lo;
        int mantissa = h & 0x3FF;
        int exponent = (h >>> 10) & 0x1F;
        float magnitude;
        if (exponent == 0) {
            magnitude = mantissa * 5.9604645E-8f;
        } else {
            int e = exponent - 15;
            int magnitudeOfE = e;
            if (e < 0) {
                magnitudeOfE = -e;
            }
            float scale = 1.0f;
            if ((magnitudeOfE & 1) != 0) {
                scale *= 2.0f;
            }
            if ((magnitudeOfE & 2) != 0) {
                scale *= 4.0f;
            }
            if ((magnitudeOfE & 4) != 0) {
                scale *= 16.0f;
            }
            if ((magnitudeOfE & 8) != 0) {
                scale *= 256.0f;
            }
            if (e < 0) {
                scale = 1.0f / scale;
            }
            magnitude = (1.0f + mantissa * (1.0f / 1024.0f)) * scale;
        }
        if ((h & 0x8000) != 0) {
            return -magnitude;
        }
        return magnitude;
    }

    /**
     * A sub-block's 6-bit scale and minimum, packed as {@code (scale << 8) | min}.
     *
     * <p>Its own method for the reason {@code TransformerComputeKernelsQ5_K} gives: inlined, the
     * decode grew large enough that TornadoVM's CUDA backend emitted a kernel referring to an
     * undeclared {@code context}.
     */
    private static int scaleAndMin(ByteArray w, int scalesBase, int subBlock) {
        if (subBlock < 4) {
            return ((w.get(scalesBase + subBlock) & 63) << 8)
                    | (w.get(scalesBase + subBlock + 4) & 63);
        }
        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
        int sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
        int m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }

    // @formatter:off
    /**
     * {@code out[M,N] = A[M,K] x W[N,K]}, with {@code W} held as {@code Q4_0} and decoded into a
     * shared tile as it is staged.
     *
     * <p>One warp per output tile of sixteen rows by eight columns — the {@code m16n8k16} shape
     * itself. The B tile is staged through {@code swizzleStoreFp16Stride32} and read with {@code
     * mmaLoadBSwizzled}, which is the route that avoids ever reading a half's bits in the kernel;
     * two implementations that did read them were tried, one refusing to compile and one crashing
     * the compiler.
     *
     * @param aFP16 activations, {@code [M padded to 16][K]}, FP16
     * @param w the weight matrix, as the file stores it, {@code [N][K]} in {@code Q4_0}
     * @param out {@code [M][N]}, FP32
     */
    // @formatter:on
    public static void projectionMMAQ4_0(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w,
            FloatArray out,
            int m,
            int n,
            int k) {
        int lane = ctx.localIdx;
        int colTiles = n / BN;
        int group = ctx.groupIdx;
        int rowTile = group / colTiles;
        int colTile = group - rowTile * colTiles;
        int blockRow = rowTile * BM;
        int blockCol = colTile * BN;
        int blocksPerRow = k / QK;

        // One allocation for both A panels, as for B: the first at byte offset zero, the second at
        // A_SUBTILE_BYTES. The A load applies no swizzle — its per-lane address is
        // (row << 5) + col with row < 16 and col in {0, 16}, so a panel reaches at most byte 496
        // and stays inside its own 512 — and the offset-aware load adds the base afterwards, so
        // each panel sees exactly the layout it had as its own array.
        int[] aTile = ctx.allocateIntLocalArray(2 * BM * BK / 2);
        // One allocation for both B panels: the low half at byte offset zero, the high half at
        // B_SUBTILE_BYTES. The offset-aware store and load apply the swizzle to the in-panel
        // address first and add the offset afterwards, so each panel keeps exactly the layout it
        // had as its own array, and the two cannot overlap -- a panel's swizzled address stays
        // inside its own 256 bytes.
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(2 * PANEL * BK);

        float[] acc = ctx.mmaFragment(0.0f);

        // One staging round per Q4_0 block, not per MMA step. A block is 32 elements and the MMA
        // step is 16, so a round stages two tiles and issues two MMAs: the block scale is read
        // once for all 32 of its weights instead of once per weight, both nibble halves of each
        // packed byte are used, and the two barriers are paid per 32 elements rather than per 16.
        //
        // A lane owns eight consecutive elements of one column. Which half of the block those
        // eight fall in is fixed by the lane, so the choice of destination tile is loop-invariant
        // rather than a branch per element.
        int stageCol = lane >> 2;
        int stageQuarter = lane & 3;
        int stageFirst = stageQuarter * 8;
        int stageByte = stageFirst & 15;
        boolean stageHighNibble = stageFirst >= 16;
        boolean stageHighHalf = stageFirst >= 16;
        // Which panel this lane stages, as a byte offset rather than a choice of array.
        int stageOffset = 0;
        if (stageHighHalf) {
            stageOffset = B_SUBTILE_BYTES;
        }
        int stageK = stageFirst & 15;

        int numBlocks = k / QK;
        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            int kBase = blockIndex * QK;

            // A: 256 ints over 32 lanes, eight each — two MMA steps' worth. Int i holds row i/8 at
            // element pair (i%8)*2 for the first step, and the same for the second.
            for (int slot = 0; slot < 8; slot++) {
                int i = lane + slot * WARP_SIZE;
                int half = i >>> 7;
                int j = i & 127;
                int row = j >>> 3;
                int kk = (j & 7) << 1;
                int base = (blockRow + row) * k + kBase + half * BK + kk;
                // The same two adjacent halves, the same destination slot, packed the same way —
                // src[base] | src[base + 1] << 16 — but copied global-to-shared without the
                // register round-trip. `base` is even (k is a whole number of Q4_0 blocks, and
                // kBase, half * BK and kk are all even), so the source byte address is
                // header + 2 * base and four-byte aligned, which is what cp.async requires.
                ctx.asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base);
            }

            // B: this lane's column, one scale, eight contiguous packed bytes.
            int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES;
            // Read through the array's own half accessor rather than assembling the half from two
            // bytes: the block stride is 18, so every block scale is two-byte aligned, and this
            // lowers to one hardware conversion where halfFromBytes lowers to a ten-branch
            // software expansion. Same bytes, same interpretation, same value. The other kernels
            // in this file keep halfFromBytes.
            float scale = w.getHalfFloat(base).getFloat32();
            for (int t = 0; t < 8; t++) {
                int packed = w.get(base + 2 + stageByte + t) & 0xFF;
                int q = packed & 0xF;
                if (stageHighNibble) {
                    q = (packed >> 4) & 0xF;
                }
                HalfFloat value = new HalfFloat(scale * (q - 8));
                // (k index, column, columns per row) — the order the swizzled load expects.
                ctx.mmaStoreBSwizzled(bTile, stageK + t, stageCol, PANEL, value, stageOffset);
            }
            // Commit and wait before the barrier that publishes both tiles: every lane issues
            // its own eight copies -- i = lane + slot * 32 covers 0..255 exactly once across the
            // warp -- and every lane waits, so no MMA reads a slot whose copy is still in flight.
            // The trailing barrier below keeps the next round's copies out of a tile this round is
            // still reading.
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, 0),
                            ctx.mmaLoadBSwizzled(bTile, BK, 0),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, A_SUBTILE_BYTES),
                            ctx.mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }

    // ---- Q4_0 dequantization to FP16 (experiment) --------------------------------

    /**
     * {@code out[row][e] = fp16(scale * (q - 8))} for a whole {@code Q4_0} matrix, one lane per
     * element, decoded with the expression the tensor-core kernels stage into their shared tiles so
     * the halves carry the same bits those kernels multiply.
     *
     * <p>The nibble is chosen by a branch on a lane-dependent value, as in the projection kernels:
     * with the choice foldable at compile time the high nibble's recentring is stamped unsigned and
     * every value below eight decodes to infinity (see
     * everyNibbleDecodesWithTheSignItsScaleGivesIt).
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ4_0ToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int blocksPerRow = k / QK;
        int row = lane / k;
        int element = lane - row * k;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set(lane, new HalfFloat(scale * (q - 8)));
    }

    /**
     * {@code out[row][e] = fp16(scale * (q5 + high * 16) - minimum)} for a whole {@code Q5_K}
     * matrix, one lane per element: the sub-block's scale and minimum through {@link #scaleAndMin},
     * the low nibble chosen by a branch on the sub-block's parity, the high bit from the {@code qh}
     * plane — the expression {@link #projectionMMAQ5_KPaired} stages, so the halves carry the bits
     * it multiplies.
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ5_KToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int superBlocksPerRow = k / QK_K;
        int row = lane / k;
        int element = lane - row * k;
        int superBlock = element >> 8;
        int inSuper = element & 255;
        int subBlock = inSuper >> 5;
        int posInSub = inSuper & 31;
        int base = (row * superBlocksPerRow + superBlock) * K_BLOCK_BYTES;
        float d = w.getHalfFloat(base).getFloat32();
        float dmin = w.getHalfFloat(base + 2).getFloat32();
        int packedScale = scaleAndMin(w, base + K_SCALES_OFFSET, subBlock);
        float scale = d * (packedScale >> 8);
        float minimum = dmin * (packedScale & 0xFF);
        int pairIndex = subBlock >> 1;
        int highNibble = subBlock & 1;
        int qsByte = w.get(base + K_QS_OFFSET + pairIndex * 32 + posInSub) & 0xFF;
        int low = qsByte & 0xF;
        if (highNibble == 1) {
            low = (qsByte >> 4) & 0xF;
        }
        int qhByte = w.get(base + K_QH_OFFSET + posInSub) & 0xFF;
        int high = (qhByte >> (pairIndex * 2 + highNibble)) & 1;
        out.set(lane, new HalfFloat(scale * (low + high * 16) - minimum));
    }

    /**
     * {@code out[row][e] = fp16(scale * q + minimum)} for a whole {@code Q4_1} matrix, one lane per
     * element: the block's two halves, the unsigned nibble chosen by a branch on a lane-dependent
     * value (the branchless form is stamped unsigned and decodes wrongly, as the Q4_0 decoder
     * records), the expression {@link #projectionMMAQ4_1} stages, so the halves carry the bits it
     * multiplies.
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ4_1ToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int blocksPerRow = k / QK;
        int row = lane / k;
        int element = lane - row * k;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES_Q4_1;
        float scale = w.getHalfFloat(base).getFloat32();
        float minimum = w.getHalfFloat(base + 2).getFloat32();
        int packed = w.get(base + 4 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set(lane, new HalfFloat(scale * q + minimum));
    }

    // @formatter:off
    /**
     * {@code out[M,N] = A[M,K] x W[N,K]} for {@code Q5_K} weights.
     *
     * <p>The same staging shape as the {@code Q4_0} form: a round is 32 elements, which for Q5_K is
     * one sub-block of a 256-weight super-block, so the sub-block's scale and minimum are computed
     * once per round and a lane owns eight consecutive elements of one column.
     */
    // @formatter:on
    public static void projectionMMAQ5_K(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w,
            FloatArray out,
            int m,
            int n,
            int k) {
        int lane = ctx.localIdx;
        int colTiles = n / BN;
        int group = ctx.groupIdx;
        int rowTile = group / colTiles;
        int colTile = group - rowTile * colTiles;
        int blockRow = rowTile * BM;
        int blockCol = colTile * BN;
        int superBlocksPerRow = k / QK_K;

        // One allocation per operand, the second panel at a byte offset, as in
        // projectionMMAQ4_0. The tile geometry and the addressing are the same here: a B panel's
        // in-panel address is at most 254 before the swizzle, which permutes within the same 256
        // bytes, and an A panel's per-lane address reaches at most 496 of its 512. The A load
        // applies no swizzle; the B store and load apply it before adding the offset.
        int[] aTile = ctx.allocateIntLocalArray(2 * BM * BK / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(2 * PANEL * BK);

        float[] acc = ctx.mmaFragment(0.0f);

        int stageCol = lane >> 2;
        int stageFirst = (lane & 3) * 8;
        boolean stageHighHalf = stageFirst >= 16;
        int stageOffset = 0;
        if (stageHighHalf) {
            stageOffset = B_SUBTILE_BYTES;
        }
        int stageK = stageFirst & 15;

        int numRounds = k / QK;
        for (int round = 0; round < numRounds; round++) {
            int kBase = round * QK;

            for (int slot = 0; slot < 8; slot++) {
                int i = lane + slot * WARP_SIZE;
                int half = i >>> 7;
                int j = i & 127;
                int row = j >>> 3;
                int kk = (j & 7) << 1;
                int base = (blockRow + row) * k + kBase + half * BK + kk;
                // The same two adjacent halves into the same slot, packed the same way, copied
                // global-to-shared without the register round-trip. `base` is even -- the dispatch
                // guard makes k a whole number of blocks, and kBase, half * BK and kk are even --
                // so the source byte address is header + 2 * base and four-byte aligned.
                ctx.asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base);
            }

            int superBlock = round >> 3;
            int subBlock = round & 7;
            int base = ((blockCol + stageCol) * superBlocksPerRow + superBlock) * K_BLOCK_BYTES;
            // As in the Q4_1 kernel: the super-block stride is 176 and d and dmin sit at offsets 0
            // and 2, so both addresses are two-byte aligned. The six-bit sub-block scales at
            // K_SCALES_OFFSET stay byte reads -- they are not halves.
            float d = w.getHalfFloat(base).getFloat32();
            float dmin = w.getHalfFloat(base + 2).getFloat32();
            int packedScale = scaleAndMin(w, base + K_SCALES_OFFSET, subBlock);
            float scale = d * (packedScale >> 8);
            float minimum = dmin * (packedScale & 0xFF);

            int pairIndex = subBlock >> 1;
            int highNibble = subBlock & 1;
            int qsBase = base + K_QS_OFFSET + pairIndex * 32;
            int qhBase = base + K_QH_OFFSET;
            int bitShift = pairIndex * 2 + highNibble;

            for (int t = 0; t < 8; t++) {
                // Q5_K indexes its packed byte by the element's position in the whole sub-block,
                // 0..31 — the nibble half is a property of the sub-block, not of the element, which
                // is what makes this different from Q4_0's byte-and-nibble split.
                int posInSub = stageFirst + t;
                int qsByte = w.get(qsBase + posInSub) & 0xFF;
                int low = qsByte & 0xF;
                if (highNibble == 1) {
                    low = (qsByte >> 4) & 0xF;
                }
                int qhByte = w.get(qhBase + posInSub) & 0xFF;
                int high = (qhByte >> bitShift) & 1;
                HalfFloat value = new HalfFloat(scale * (low + high * 16) - minimum);
                ctx.mmaStoreBSwizzled(bTile, stageK + t, stageCol, PANEL, value, stageOffset);
            }
            // Commit and wait before the barrier that publishes both tiles: every lane issues
            // its own eight copies -- i = lane + slot * 32 covers 0..255 exactly once across the
            // warp -- and every lane waits, so no MMA reads a slot whose copy is still in flight.
            // The trailing barrier below keeps the next round's copies out of a tile this round is
            // still reading.
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, 0),
                            ctx.mmaLoadBSwizzled(bTile, BK, 0),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, A_SUBTILE_BYTES),
                            ctx.mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }

    // @formatter:off
    /**
     * {@code out = fp16(scale * (q - 8))} for a whole {@code Q4_0} matrix, one lane per element,
     * the decode expression of {@link #dequantizeQ4_0ToFP16}, written not row-major but in the
     * order the tiled GEMM stages its B tile in shared memory, so the GEMM can copy each tile
     * global-to-shared as contiguous four-byte words.
     *
     * <p><b>Layout.</b> The matrix is {@code n} rows (output columns of the projection) by {@code
     * k}. It is cut into tiles of 128 rows by 16 k, numbered {@code tile = (row / 128) * (k / 16) +
     * kk / 16}: all of a row block's k-steps in order, then the next row block. A tile holds 1024
     * packed pairs; pair {@code idx} (0..1023) holds rows {@code (idx >>> 6) * 8 + (idx & 3) * 2}
     * and that plus one at k {@code (idx & 63) >>> 2} within the tile — the index {@code gemmMMA}
     * gives {@code bTile[idx]} — with the even row in the low half. Half position {@code h = tile *
     * 2048 + idx * 2 + (row & 1)}.
     *
     * <p><b>Inverse.</b> From {@code h}: {@code idx = (h >>> 1) & 1023}, {@code tile = h >>> 11},
     * {@code row = (tile / (k / 16)) * 128 + ((idx >>> 6) << 3) + ((idx & 3) << 1) + (h & 1)},
     * {@code kk = (tile % (k / 16)) * 16 + ((idx & 63) >>> 2)}. Every half position maps to one
     * element and back; {@code n % 128 == 0} and {@code k % 16 == 0}, as the GEMM requires.
     *
     * <p>Worker: {@code n * k} lanes; the lane order is a fixed permutation of the half positions
     * (see the body), so every half is written exactly once.
     */
    // @formatter:on
    public static void dequantizeQ4_0ToFP16Tiled(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / GEMM_BK;
        // Lane to half position: not the identity. A warp of 32 lanes covers four rows by eight k
        // (four blocks read, four 32-byte sectors written) rather than the eight rows by four k
        // the identity would give (eight blocks read, two sectors written); measured 2-3% faster
        // over the production shapes. Lane bits, low to high: row parity, the low pair bit, three
        // low k bits, the high pair bit, the high k bit, then the sub-tile and tile.
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int pair = ((lane >>> 7) << 6) + (kk << 2) + pairInSub;
        int tile = pair >>> 10;
        int idx = pair & (GEMM_B_TILE_INTS - 1);
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * GEMM_BN + ((idx >>> 6) << 3) + ((idx & 3) << 1) + parity;
        int element = kStep * GEMM_BK + ((idx & 63) >>> 2);
        int blocksPerRow = k / QK;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set((pair << 1) + parity, new HalfFloat(scale * (q - 8)));
    }

    // @formatter:off
    /**
     * The causal convolution of {@link #causalConv1dScan} with a lane per (row, channel) instead of
     * a lane per channel walking the chunk.
     *
     * <p>The scan's output for row {@code r} of channel {@code c} is {@code w[0] * x[r-3] + w[1] *
     * x[r-2] + w[2] * x[r-1] + w[3] * x[r]} (kernel 4), where {@code x[i]} for {@code i < 0} is the
     * window the chunk started with; every row's output depends only on the input and that initial
     * window, not on other rows' outputs, so the rows are independent. This kernel reads the taps
     * in the same order and sums them in the same order as the scan, so the outputs are bit-equal
     * to it (asserted by the test). It does not touch the window: the scan shifts the window as it
     * goes and a lane per row cannot, since the rows below three read the initial window while
     * later rows would overwrite it. {@link #causalConv1dWindowUpdate} writes the chunk's final
     * window afterwards, as a task of its own.
     *
     * <p>Worker: {@code rows * channels} lanes; lanes past {@code activeRows * channels} return.
     */
    // @formatter:on
    public static void causalConv1dBatch(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray outBatch,
            int channels,
            int kernel,
            int windowOffset,
            IntArray batchInfo) {
        int lane = context.globalIdx;
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
        outBatch.set(lane, sum);
    }

    // @formatter:off
    /**
     * {@link #attentionBatchFP16PagedScored} with the first pass reading keys through a transposed
     * shared-memory tile.
     *
     * <p>The reference first pass has lane {@code p} read key row {@code p} on its own: across a
     * warp the addresses are a key row apart, so every load instruction touches thirty-two separate
     * segments. Here positions are taken 128 at a time and dimensions 32 at a time: the 128 lanes
     * load the tile's 128 x 32 halves with consecutive lanes reading consecutive dimensions of one
     * position (coalesced), widen them to FP32 (exact) and store them transposed, {@code
     * keyTile[dim * 129 + positionInTile]}, so that a lane's later reads of its own position's
     * dimensions are consecutive across the warp and the stores of one position's dimensions fall
     * in distinct banks (the padding to 129 is what separates them). Lane {@code tid} owns position
     * {@code tileStart + tid} — the reference's lane-to-position assignment — and accumulates its
     * one FP32 score over the dimensions in increasing order, tile after tile, the reference's
     * order; the unscaled score is stored and scaled exactly as before.
     *
     * <p>Every lane runs every tile's barriers; a lane whose position lies past the causal range
     * skips only its loads (storing zeros), its accumulation and its stores.
     *
     * <p>Shared memory: {@code 32 * 129} floats for the tile, plus the reference kernel's own. The
     * remaining passes, the score scratch, the launch geometry and the value pass are the reference
     * kernel's.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScoredStaged(
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
     * Batched FP16-KV attention on the tensor cores: {@code S = Q K^T} and {@code O = P V} as
     * {@code m16n8k16} FP16 MMAs with FP32 accumulators, one workgroup per (16-query tile, head),
     * three passes over the causal range as the scored kernels make. Experiment; {@link
     * #attentionBatchFP16PagedScoredWarp} is the control and the fallback.
     *
     * <p><b>Why three passes and not an online softmax.</b> The kernel language reaches an MMA
     * accumulator's values only through {@code mmaStore} to global memory; a fragment cannot be
     * read or rescaled in registers. So the scores go to the existing per-(row, head) score scratch
     * as the first pass computes them (the same unscaled FP32 scores the control stores), the
     * maximum and the denominator are taken from the scratch, and the third pass accumulates {@code
     * P V} against the settled maximum -- no running maximum, no output rescaling -- and divides
     * after the store. The equations are the control's: {@code m = max_k s_k / sqrt(d)}, {@code l =
     * sum_k exp(s_k / sqrt(d) - m)}, {@code O = (sum_k p_k v_k) / l}, all FP32 except where stated
     * below.
     *
     * <p><b>Tiles and ownership.</b> Workgroup {@code g}: query tile {@code g / heads} (rows {@code
     * 16 (g / heads) .. + 15} of the chunk), head {@code g % heads}, 128 lanes. Keys are taken 64
     * at a time. Pass 1: warp {@code w} computes the score tile for keys {@code 16 w .. 16 w + 15}
     * of the key tile (two 16x8 accumulators) over the 16 head-dimension steps and stores it to the
     * scratch. Pass 3: warp {@code w} accumulates output dimensions {@code 64 w .. 64 w + 63}
     * (eight 16x8 accumulators) over the two 16-key steps of each tile.
     *
     * <p><b>Shared buffers.</b> {@code qTile} (int[2048]): Q as the A operand, 16 blocks of [16
     * queries][16 dims], copied once per workgroup from the FP16 staging scratch (the queries are
     * converted into the workgroup's region of {@code stage} by {@code HalfFloatArray.set} -- there
     * is no float-to-half conversion into registers -- then {@code cp.async}ed). {@code kTile}
     * (HalfFloat[16384]): K^T as the swizzled B operand, sub-tile {@code 8 t + g} = dims {@code 16
     * t ..} by keys {@code 8 g ..}, restaged per key tile by element-wise swizzled stores (one lane
     * per element, coalesced reads), read until the barrier that ends the tile. {@code vTile}
     * (int[8192]): V as the B operand, sub-tile {@code 32 t + j} = keys {@code 16 t ..} by dims
     * {@code 8 j ..}, copied per key tile by {@code cp.async} (four-byte words of a V row). {@code
     * pTile} (int[512]): P as the A operand, 4 blocks of [16 queries][16 keys], per key tile
     * written as halves into the staging scratch and copied back by {@code cp.async}. {@code
     * rowStat} (float[16 + 16 + 8 * 16]): the row maxima, the row sums and the eight-lane partials
     * of the second pass.
     *
     * <p><b>Rounding.</b> Queries are rounded to FP16 <i>unscaled</i>; the {@code 1/sqrt(256)}
     * scale is applied in FP32 to the accumulated score. Keys and values are the stored FP16.
     * Scores, maxima, sums, the exponentials and the output accumulators are FP32. Each probability
     * is rounded to FP16 for the P V MMA; the denominator accumulates the FP32 probabilities, so
     * numerator and denominator differ by that rounding.
     *
     * <p><b>Masking and partial tiles.</b> Key {@code k} is valid for query row {@code q} iff
     * {@code k <= position(q) = startPos + 16 (g / heads) + q}; an invalid probability is zero (its
     * score is never read for the maximum or the sum, and its {@code P} entry is written as zero).
     * Every row has key 0, so its maximum is finite. The key-tile loops run to the tile's largest
     * position, a workgroup-uniform bound, so every lane reaches every barrier; rows at or past the
     * chunk's active count compute at their nominal position and their outputs land in the chunk's
     * padding rows, which nothing consumes (the control writes nothing there). Keys past the tile's
     * last valid key (the smaller of its largest position and the context capacity minus one) are
     * staged from that last valid key's row, a valid address, and are always masked.
     *
     * <p><b>Addressing.</b> Head {@code h} reads key/value head {@code h / kvMul}; each key row is
     * located by {@code KvBlockAddress.offset} on its own, so pages need not be contiguous.
     * Requires the chunk width a multiple of 16, a 256-wide head, 128-lane workgroups, and a
     * staging scratch of {@code (rows / 16) * heads * TC_STAGE_HALVES} halves. Worker: {@code (rows
     * / 16) * heads * 128} lanes, local 128.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedTensorCore(
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

        int[] qTile = context.allocateIntLocalArray(TC_QUERIES * TC_HEAD / 2);
        HalfFloat[] kTile = context.allocateHalfFloatLocalArray(TC_KEYS * TC_HEAD);
        int[] vTile = context.allocateIntLocalArray(TC_KEYS * TC_HEAD / 2);
        int[] pTile = context.allocateIntLocalArray(TC_QUERIES * TC_KEYS / 2);
        float[] rowStat = context.allocateFloatLocalArray(TC_QUERIES * 2 + TC_LANES);

        int stageBase = group * TC_STAGE_HALVES;
        int qBase = (rowBase * heads + head) * headSize;
        for (int i = tid; i < TC_QUERIES * TC_HEAD; i += TC_LANES) {
            int row = i >> 8;
            int d = i & 255;
            stage.set(
                    stageBase + i,
                    new HalfFloat(queryBatch.get(qBase + row * heads * headSize + d)));
        }
        context.localBarrier();
        for (int i = tid; i < TC_QUERIES * TC_HEAD / 2; i += TC_LANES) {
            int t = i >> 7;
            int within = i & 127;
            int row = within >> 3;
            int d = (t << 4) + ((within & 7) << 1);
            context.asyncCopyToLocal(qTile, i, stage, stageBase + (row << 8) + d);
        }
        context.asyncCopyCommit();
        context.asyncCopyWaitGroup(0);
        context.localBarrier();

        int maxPos = startPos + rowBase + TC_QUERIES - 1;
        int lastKey = capacity - 1;
        if (maxPos < lastKey) {
            lastKey = maxPos;
        }
        // The score scratch as a matrix with row stride heads * scoreStride: row r of this tile
        // is (rowBase + r) * heads + head spans, i.e. column offset head * scoreStride.
        int scoreLd = heads * scoreStride;
        int scoreCol = head * scoreStride;

        // Pass 1: S = Q K^T, 32 keys a tile, stored unscaled to the scratch.
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
            for (int i = 0; i < TC_KEYS * TC_HEAD / TC_LANES; i++) {
                int e = i * TC_LANES + tid;
                int key = e >> 8;
                int d = e & 255;
                int p = tileStart + key;
                if (p > lastKey) {
                    p = lastKey;
                }
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                context.mmaStoreBSwizzled(
                        kTile,
                        d & 15,
                        key & 7,
                        8,
                        // Widened and re-narrowed (exact): the route the store's lowering takes
                        // a half by value, rather than a half read from the array.
                        new HalfFloat(keyCache.get(base + d).getFloat32()),
                        (((d >> 4) << 2) + (key >> 3)) * 256);
            }
            context.localBarrier();
            float[] s0 = context.mmaFragment(0.0f);
            for (int t = 0; t < TC_HEAD / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(qTile, 16, t * 512);
                HalfFloat[] b0 = context.mmaLoadBSwizzled(kTile, 16, ((t << 2) + warp) * 256);
                s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
            }
            // This warp's eight keys of the tile; skipped whole where they lie past the capacity
            // (a warp-uniform choice, so no lane misses a barrier): those keys are never read.
            if (tileStart + (warp << 3) + 7 < capacity) {
                context.mmaStore(s0, scores, rowBase, scoreCol + tileStart + (warp << 3), scoreLd);
            }
            context.localBarrier();
        }

        // Pass 2: per row, the maximum and the denominator of the scaled scores, eight lanes a
        // row (row = tid / 8, keys tid % 8, + 8, ...), folded by shuffles within the eight.
        int statRow = tid >> 3;
        int statLane = tid & 7;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        int statBase = ((rowBase + statRow) * heads + head) * scoreStride;
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statLane; p <= statPos; p += 8) {
            rowMax = TornadoMath.max(rowMax, scores.get(statBase + p) * invSqrt);
        }
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 4));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 2));
        rowMax = TornadoMath.max(rowMax, context.simdShuffleDown(rowMax, 1));
        if (statLane == 0) {
            rowStat[statRow] = rowMax;
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statLane; p <= statPos; p += 8) {
            rowSum += TornadoMath.exp(scores.get(statBase + p) * invSqrt - m);
        }
        rowSum += context.simdShuffleDown(rowSum, 4);
        rowSum += context.simdShuffleDown(rowSum, 2);
        rowSum += context.simdShuffleDown(rowSum, 1);
        if (statLane == 0) {
            rowStat[TC_QUERIES + statRow] = rowSum;
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
        for (int tileStart = 0; tileStart <= lastKey; tileStart += TC_KEYS) {
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
                int dst = (((key >> 4) << 5) + (d >> 3)) * 64 + ((key & 15) << 2) + ((d & 7) >> 1);
                context.asyncCopyToLocal(vTile, dst, valueCache, base + d);
            }
            context.asyncCopyCommit();
            // P for this tile: lane covers (row = i / 64, key = i % 64), 8 per lane, as halves.
            for (int i = tid; i < TC_QUERIES * TC_KEYS; i += TC_LANES) {
                int row = i >> 5;
                int key = tileStart + (i & 31);
                float prob = 0.0f;
                if (key <= startPos + rowBase + row) {
                    prob =
                            TornadoMath.exp(
                                    scores.get(((rowBase + row) * heads + head) * scoreStride + key)
                                                    * invSqrt
                                            - rowStat[row]);
                }
                stage.set(pBase + i, new HalfFloat(prob));
            }
            context.localBarrier();
            for (int i = tid; i < TC_QUERIES * TC_KEYS / 2; i += TC_LANES) {
                int t = i >> 7;
                int q = (i >> 3) & 15;
                int kk = (t << 4) + ((i & 7) << 1);
                context.asyncCopyToLocal(pTile, i, stage, pBase + (q << 5) + kk);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(0);
            context.localBarrier();
            for (int t = 0; t < TC_KEYS / 16; t++) {
                HalfFloat[] a = context.mmaLoadA(pTile, 16, t * 512);
                int vBase = (t << 5) + (warp << 3);
                o0 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 0) * 256),
                                o0,
                                MMAShape.M16N8K16);
                o1 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 1) * 256),
                                o1,
                                MMAShape.M16N8K16);
                o2 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 2) * 256),
                                o2,
                                MMAShape.M16N8K16);
                o3 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 3) * 256),
                                o3,
                                MMAShape.M16N8K16);
                o4 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 4) * 256),
                                o4,
                                MMAShape.M16N8K16);
                o5 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 5) * 256),
                                o5,
                                MMAShape.M16N8K16);
                o6 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 6) * 256),
                                o6,
                                MMAShape.M16N8K16);
                o7 =
                        context.mma(
                                a,
                                context.mmaLoadB(vTile, 16, (vBase + 7) * 256),
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

    // @formatter:off
    /**
     * {@link #attentionBatchFP16Paged} with each causal query-key dot product computed once.
     *
     * <p>The reference kernel walks the causal range three times and recomputes every dot product
     * on each walk: once for the maximum, once for the denominator, once for the value weights.
     * This form computes it in the first walk, stores the <b>unscaled</b> FP32 sum in {@code
     * scores} and reads it back in the other two. The score is stored before {@code invSqrt} is
     * applied, so the scaled expression the later passes evaluate — {@code score * invSqrt -
     * globalMax} — is the same expression over the same operand bits, and the compiler's
     * contraction of it is the same in every pass. The dot product's own accumulation order, the
     * reductions, the exponentials and the weighted value sum are the reference kernel's.
     *
     * <p>{@code scores} is scratch indexed by {@code ((row * heads) + head) * scoreStride + p}, so
     * every (row, head) workgroup of a launch owns a disjoint span and no launch depends on what an
     * earlier one left: every position a workgroup reads in passes two and three is one its own
     * first pass wrote — the first walk covers {@code p = tid, tid + localSize, ...} up to the
     * row's position, which is exactly the range the later walks read — and the workgroup barriers
     * between the passes are what make one lane's global stores visible to the lanes that read them
     * in the value pass, where a position belongs to a different lane.
     *
     * @param scores per-launch scratch, at least {@code rows * heads * scoreStride} floats
     * @param scoreStride the stride between (row, head) spans; at least the largest position + 1
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScored(
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

        // Pass 1: this lane's slice of the causal range, tracking a running maximum; the unscaled
        // dot product is kept for the other two passes.
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
            scores.set(scoreBase + p, score);
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
     * {@link #processHeadsFlashAttentionSplitKVFP16Paged} for a head twice as wide.
     *
     * <p>Identical body, identical algorithm, identical FP16 key/value interpretation. The only
     * difference is the pair of constants that size the shared arrays, and they are the reason a
     * separate method exists: {@code accShared} is {@code MAX_LOCAL_SIZE * MAX_HEAD_SIZE} floats,
     * so the 128-wide kernel's 64 x 128 cannot be handed a 256-wide head — it would read and write
     * past the array. This one is <b>32 x 256</b>, the same 32 KiB, so the wider head costs no more
     * shared memory; what it costs is half the lanes per workgroup, which the splits give back many
     * times over.
     *
     * <p><b>The workgroup must therefore be launched with at most 32 lanes</b>, because {@code
     * mShared}, {@code lShared} and {@code corrShared} are indexed by {@code localIdx} and both
     * block reductions run to {@code localGroupSizeX}. Everything else — the strided query stage,
     * the position scan, the running max and sum, and the strided partial write — is written
     * against {@code headSize} and {@code localSize} and needs no change.
     *
     * <p>The partial layout is the one {@code combineSplitKVAttention} expects, unchanged: per head
     * {@code nSplits * headSize} numerators, then {@code nSplits} maxima, then {@code nSplits}
     * sums. That combine is already generic in {@code headSize} and already treats a split whose
     * chunk was empty as a zero contribution.
     */
    // @formatter:on
    public static void processHeadsFlashAttentionSplitKVFP16PagedWideHead(
            KernelContext context,
            FloatArray q,
            HalfFloatArray key_cache,
            HalfFloatArray value_cache,
            FloatArray att,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int nSplits) {

        final int MAX_HEAD_SIZE = 256;
        final int MAX_LOCAL_SIZE = 32;

        int tid = context.localIdx;
        int g = context.groupIdx; // 0 .. nHeads*nSplits - 1
        int localSize = context.localGroupSizeX;
        int h = g / nSplits;
        int s = g % nSplits;

        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int seqLen = pos + 1;
        int chunk = (seqLen + nSplits - 1) / nSplits;
        int startPos = s * chunk;
        int endPos = Math.min(startPos + chunk, seqLen); // exclusive

        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] q_shared = context.allocateFloatLocalArray(MAX_HEAD_SIZE);
        float[] accShared = context.allocateFloatLocalArray(MAX_LOCAL_SIZE * MAX_HEAD_SIZE);
        float[] mShared = context.allocateFloatLocalArray(MAX_LOCAL_SIZE);
        float[] lShared = context.allocateFloatLocalArray(MAX_LOCAL_SIZE);
        float[] corrShared = context.allocateFloatLocalArray(MAX_LOCAL_SIZE);
        float[] bcast = context.allocateFloatLocalArray(1);

        int headBase = h * nSplits * (headSize + 2);
        int outBase = headBase + s * headSize;
        int mBase = headBase + nSplits * headSize;
        int lBase = mBase + nSplits;

        for (int i = tid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }
        int rowBase = tid * headSize;
        for (int d = 0; d < headSize; d++) {
            accShared[rowBase + d] = 0.0f;
        }
        context.localBarrier();

        // Strided scan over this split's position chunk (no barriers).
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        for (int p = startPos + tid; p < endPos; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHeadIdx * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d += 2) {
                Half2 kPair = key_cache.getHalf2(base + d);
                score += q_shared[d] * Half2.lowFloat(kPair);
                score += q_shared[d + 1] * Half2.highFloat(kPair);
            }
            score *= invSqrt;
            float newM = Math.max(m, score);
            float corr = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - newM);
            float e = TornadoMath.exp(score - newM);
            for (int d = 0; d < headSize; d += 2) {
                Half2 vPair = value_cache.getHalf2(base + d);
                accShared[rowBase + d] = accShared[rowBase + d] * corr + e * Half2.lowFloat(vPair);
                accShared[rowBase + d + 1] =
                        accShared[rowBase + d + 1] * corr + e * Half2.highFloat(vPair);
            }
            l = l * corr + e;
            m = newM;
        }
        mShared[tid] = m;
        lShared[tid] = l;
        context.localBarrier();

        // Block max.
        if (tid == 0) {
            float blockMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < localSize; t++) {
                if (mShared[t] > blockMax) {
                    blockMax = mShared[t];
                }
            }
            bcast[0] = blockMax;
        }
        context.localBarrier();
        float M = bcast[0];

        corrShared[tid] =
                (mShared[tid] == Float.NEGATIVE_INFINITY)
                        ? 0.0f
                        : TornadoMath.exp(mShared[tid] - M);
        context.localBarrier();

        // Block sum L = Σ_t l_t · corr_t.
        if (tid == 0) {
            float blockSum = 0.0f;
            for (int t = 0; t < localSize; t++) {
                blockSum += lShared[t] * corrShared[t];
            }
            bcast[0] = blockSum;
        }
        context.localBarrier();
        float L = bcast[0];

        // Write UNNORMALIZED partial numerator (relative to block max M), plus M and L for the
        // combine.
        for (int d = tid; d < headSize; d += localSize) {
            float acc = 0.0f;
            for (int t = 0; t < localSize; t++) {
                acc += corrShared[t] * accShared[t * headSize + d];
            }
            att.set(outBase + d, acc);
        }
        if (tid == 0) {
            att.set(mBase + s, M);
            att.set(lBase + s, L);
        }
    }
}
