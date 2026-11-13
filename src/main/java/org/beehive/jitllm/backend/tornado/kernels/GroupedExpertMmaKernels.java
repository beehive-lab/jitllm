package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Grouped expert GEMMs on tensor cores (CUDA), for a batched mixture-of-experts prefill: one
 * workgroup per tile of one expert's assignments and 128 outputs, the expert's weights read in
 * place from the stacked tensor once per tile. Eight warps, two by four, each 32 rows by 32 outputs
 * as {@code M16N8K16} MMAs over FP16 inputs; the depth advances 16 at a time.
 *
 * <p>The weights enter the MMA as their integer codes, which FP16 holds exactly: a code's bits come
 * from a 256-entry table in shared memory, so no weight is converted to floating point. Each scale
 * group's partial products accumulate in their own fragments and are then scaled in FP32 by the
 * group's scale; for Q4_K and Q5_K the group's minimum is subtracted through the input rows' sums
 * per 32 inputs ({@link Lfm2BatchKernels#halvesWithBlockSums}). Example: a Q4_K weight {@code d *
 * sc * q - dmin * m} contributes {@code d * sc * (x . q) - dmin * m * sum(x)} per 32 inputs.
 */
// @formatter:on
public final class GroupedExpertMmaKernels {

    private GroupedExpertMmaKernels() {}

    /** Rows of a workgroup's tile: an expert tile holds at most 64. */
    private static final int BM = 64;

    /** Outputs of a workgroup. */
    private static final int BN = 128;

    /** Depth of one MMA step. */
    private static final int BK = 16;

    /** Bytes of one eight-output B operand in shared memory. */
    private static final int B_SUBTILE_BYTES = 256;

    /** Threads of a workgroup: eight warps. */
    public static final int THREADS = 256;

    /** The input row of a tile's local row {@code local}. */
    private static int sourceRow(IntArray sortedToken, int gather, int first, int local) {
        return gather != 0 ? sortedToken.get(first + local) : first + local;
    }

    /** Halves {@code i} and {@code i + 1} of {@code x} as one int. */
    private static int packHalves(HalfFloatArray x, int i) {
        return (x.get(i).getHalfFloatValue() & 0xFFFF)
                | ((x.get(i + 1).getHalfFloatValue() & 0xFFFF) << 16);
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q4_K experts on tensor cores (CUDA): {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)} per
     * tile, one workgroup per tile and 128 outputs. Row {@code i} of a tile reads FP16 input row
     * {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise. The
     * weights enter the MMA as their integer codes, exact in FP16, and each scale group's sums are
     * scaled in FP32 by its scale and minimum, the minimum through {@code blockSums}, the input
     * rows' sums per 32 inputs; only the tile's rows are written, to output row {@code first + i}.
     * Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups of {@link
     * #THREADS} threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertMmaQ4_K(
            KernelContext ctx,
            HalfFloatArray input16,
            FloatArray blockSums,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        int tid = ctx.localIdx;
        int lane = tid & 31;
        int warpM = tid >> 7;
        int warpN = (tid >> 5) & 3;
        int colBlocks = d >> 7;
        int grp = ctx.groupIdx;
        int tile = grp / colBlocks;
        if (tile >= tiles.get(0)) {
            return;
        }
        int blockCol = (grp - tile * colBlocks) << 7;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int rows = tiles.get(3 + 3 * tile);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);
        int[] codes = ctx.allocateIntLocalArray(THREADS);
        float[] colScale = ctx.allocateFloatLocalArray(BN);
        float[] colMin = ctx.allocateFloatLocalArray(BN);
        codes[tid] = TransformerBatchPrefillKernels.fp16BitsOf((float) (tid - 128));
        float[] s00 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float[] s01 = ctx.mmaFragment(0.0f);
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float[] s02 = ctx.mmaFragment(0.0f);
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float[] s03 = ctx.mmaFragment(0.0f);
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float[] s10 = ctx.mmaFragment(0.0f);
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float[] s11 = ctx.mmaFragment(0.0f);
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float[] s12 = ctx.mmaFragment(0.0f);
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float[] s13 = ctx.mmaFragment(0.0f);
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        int aIdx0 = tid;
        int aIdx1 = tid + 256;
        int aBase0 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx0 >>> 3, rows - 1)) * n
                        + ((aIdx0 & 7) << 1);
        int aBase1 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx1 >>> 3, rows - 1)) * n
                        + ((aIdx1 & 7) << 1);
        int pairCol = (tid >> 2) << 1;
        int kQuarter = (tid & 3) << 2;
        int bSlot = (pairCol >> 3) * 64 + kQuarter * 4 + ((pairCol & 7) >> 1);
        int wRow = expert * d + blockCol + pairCol;
        int blocksPerRow = n >> 8;

        int r0 = warpM * 32 + (lane >> 2);
        int r1 = r0 + 16;
        int lc0 = warpN * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int sumsPerRow = n >> 5;
        int xr0 = sourceRow(sortedToken, gather, first, Math.min(r0, rows - 1)) * sumsPerRow;
        int xr1 = sourceRow(sortedToken, gather, first, Math.min(r0 + 8, rows - 1)) * sumsPerRow;
        int xr2 = sourceRow(sortedToken, gather, first, Math.min(r1, rows - 1)) * sumsPerRow;
        int xr3 = sourceRow(sortedToken, gather, first, Math.min(r1 + 8, rows - 1)) * sumsPerRow;
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            aTile[aIdx0] = packHalves(input16, aBase0 + kBase);
            aTile[aIdx1] = packHalves(input16, aBase1 + kBase);
            int k = kBase + kQuarter;
            int blk = k >> 8;
            int within = k & 255;
            int pairIndex = within >> 6;
            int high = (within >> 5) & 1;
            int sub = (pairIndex << 1) + high;
            int base0 = (wRow * blocksPerRow + blk) * 144;
            int base1 = base0 + blocksPerRow * 144;
            int qs0 = w.getInt(base0 + 16 + pairIndex * 32 + (within & 31));
            int qs1 = w.getInt(base1 + 16 + pairIndex * 32 + (within & 31));
            if (high == 1) {
                qs0 = qs0 >>> 4;
                qs1 = qs1 >>> 4;
            }
            bTile[bSlot] = codes[128 + (qs0 & 15)] | (codes[128 + (qs1 & 15)] << 16);
            bTile[bSlot + 4] =
                    codes[128 + ((qs0 >>> 8) & 15)] | (codes[128 + ((qs1 >>> 8) & 15)] << 16);
            bTile[bSlot + 8] =
                    codes[128 + ((qs0 >>> 16) & 15)] | (codes[128 + ((qs1 >>> 16) & 15)] << 16);
            bTile[bSlot + 12] =
                    codes[128 + ((qs0 >>> 24) & 15)] | (codes[128 + ((qs1 >>> 24) & 15)] << 16);
            if (kQuarter == 0) {
                int sm0 = TransformerComputeKernelsQ4_K.scaleAndMin(w, base0 + 4, sub);
                int sm1 = TransformerComputeKernelsQ4_K.scaleAndMin(w, base1 + 4, sub);
                colScale[pairCol] = w.getHalfFloat(base0).getFloat32() * (sm0 >> 8);
                colScale[pairCol + 1] = w.getHalfFloat(base1).getFloat32() * (sm1 >> 8);
                colMin[pairCol] = w.getHalfFloat(base0 + 2).getFloat32() * (sm0 & 0xFF);
                colMin[pairCol + 1] = w.getHalfFloat(base1 + 2).getFloat32() * (sm1 & 0xFF);
            }
            ctx.localBarrier();
            // A warp whose rows all lie past the tile's multiplies nothing it stores.
            if (warpM * 32 < rows) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, warpM * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, warpM * 1024 + 512);
                HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 0) * B_SUBTILE_BYTES);
                HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 1) * B_SUBTILE_BYTES);
                HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 2) * B_SUBTILE_BYTES);
                HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 3) * B_SUBTILE_BYTES);
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
            }
            if ((kStep & 1) == 1) {
                int sb = kBase >> 5;
                float x0 = blockSums.get(xr0 + sb);
                float x1 = blockSums.get(xr1 + sb);
                float x2 = blockSums.get(xr2 + sb);
                float x3 = blockSums.get(xr3 + sb);
                t00_0 += colScale[lc0] * s00[0] - colMin[lc0] * x0;
                t00_1 += colScale[lc0 + 1] * s00[1] - colMin[lc0 + 1] * x0;
                t00_2 += colScale[lc0] * s00[2] - colMin[lc0] * x1;
                t00_3 += colScale[lc0 + 1] * s00[3] - colMin[lc0 + 1] * x1;
                t01_0 += colScale[lc1] * s01[0] - colMin[lc1] * x0;
                t01_1 += colScale[lc1 + 1] * s01[1] - colMin[lc1 + 1] * x0;
                t01_2 += colScale[lc1] * s01[2] - colMin[lc1] * x1;
                t01_3 += colScale[lc1 + 1] * s01[3] - colMin[lc1 + 1] * x1;
                t02_0 += colScale[lc2] * s02[0] - colMin[lc2] * x0;
                t02_1 += colScale[lc2 + 1] * s02[1] - colMin[lc2 + 1] * x0;
                t02_2 += colScale[lc2] * s02[2] - colMin[lc2] * x1;
                t02_3 += colScale[lc2 + 1] * s02[3] - colMin[lc2 + 1] * x1;
                t03_0 += colScale[lc3] * s03[0] - colMin[lc3] * x0;
                t03_1 += colScale[lc3 + 1] * s03[1] - colMin[lc3 + 1] * x0;
                t03_2 += colScale[lc3] * s03[2] - colMin[lc3] * x1;
                t03_3 += colScale[lc3 + 1] * s03[3] - colMin[lc3 + 1] * x1;
                t10_0 += colScale[lc0] * s10[0] - colMin[lc0] * x2;
                t10_1 += colScale[lc0 + 1] * s10[1] - colMin[lc0 + 1] * x2;
                t10_2 += colScale[lc0] * s10[2] - colMin[lc0] * x3;
                t10_3 += colScale[lc0 + 1] * s10[3] - colMin[lc0 + 1] * x3;
                t11_0 += colScale[lc1] * s11[0] - colMin[lc1] * x2;
                t11_1 += colScale[lc1 + 1] * s11[1] - colMin[lc1 + 1] * x2;
                t11_2 += colScale[lc1] * s11[2] - colMin[lc1] * x3;
                t11_3 += colScale[lc1 + 1] * s11[3] - colMin[lc1 + 1] * x3;
                t12_0 += colScale[lc2] * s12[0] - colMin[lc2] * x2;
                t12_1 += colScale[lc2 + 1] * s12[1] - colMin[lc2 + 1] * x2;
                t12_2 += colScale[lc2] * s12[2] - colMin[lc2] * x3;
                t12_3 += colScale[lc2 + 1] * s12[3] - colMin[lc2 + 1] * x3;
                t13_0 += colScale[lc3] * s13[0] - colMin[lc3] * x2;
                t13_1 += colScale[lc3 + 1] * s13[1] - colMin[lc3 + 1] * x2;
                t13_2 += colScale[lc3] * s13[2] - colMin[lc3] * x3;
                t13_3 += colScale[lc3 + 1] * s13[3] - colMin[lc3 + 1] * x3;
                s00 = ctx.mmaFragment(0.0f);
                s01 = ctx.mmaFragment(0.0f);
                s02 = ctx.mmaFragment(0.0f);
                s03 = ctx.mmaFragment(0.0f);
                s10 = ctx.mmaFragment(0.0f);
                s11 = ctx.mmaFragment(0.0f);
                s12 = ctx.mmaFragment(0.0f);
                s13 = ctx.mmaFragment(0.0f);
            }
            ctx.localBarrier();
        }

        int col0 = blockCol + lc0;
        int col1 = blockCol + lc1;
        int col2 = blockCol + lc2;
        int col3 = blockCol + lc3;
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col0, t00_0);
            outputBatch.set((first + r0) * d + col0 + 1, t00_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col0, t00_2);
            outputBatch.set((first + r0 + 8) * d + col0 + 1, t00_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col1, t01_0);
            outputBatch.set((first + r0) * d + col1 + 1, t01_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col1, t01_2);
            outputBatch.set((first + r0 + 8) * d + col1 + 1, t01_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col2, t02_0);
            outputBatch.set((first + r0) * d + col2 + 1, t02_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col2, t02_2);
            outputBatch.set((first + r0 + 8) * d + col2 + 1, t02_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col3, t03_0);
            outputBatch.set((first + r0) * d + col3 + 1, t03_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col3, t03_2);
            outputBatch.set((first + r0 + 8) * d + col3 + 1, t03_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col0, t10_0);
            outputBatch.set((first + r1) * d + col0 + 1, t10_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col0, t10_2);
            outputBatch.set((first + r1 + 8) * d + col0 + 1, t10_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col1, t11_0);
            outputBatch.set((first + r1) * d + col1 + 1, t11_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col1, t11_2);
            outputBatch.set((first + r1 + 8) * d + col1 + 1, t11_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col2, t12_0);
            outputBatch.set((first + r1) * d + col2 + 1, t12_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col2, t12_2);
            outputBatch.set((first + r1 + 8) * d + col2 + 1, t12_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col3, t13_0);
            outputBatch.set((first + r1) * d + col3 + 1, t13_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col3, t13_2);
            outputBatch.set((first + r1 + 8) * d + col3 + 1, t13_3);
        }
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q5_K experts on tensor cores (CUDA): {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)} per
     * tile, one workgroup per tile and 128 outputs. Row {@code i} of a tile reads FP16 input row
     * {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise. The
     * weights enter the MMA as their integer codes, exact in FP16, and each scale group's sums are
     * scaled in FP32 by its scale and minimum, the minimum through {@code blockSums}, the input
     * rows' sums per 32 inputs; only the tile's rows are written, to output row {@code first + i}.
     * Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups of {@link
     * #THREADS} threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertMmaQ5_K(
            KernelContext ctx,
            HalfFloatArray input16,
            FloatArray blockSums,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        int tid = ctx.localIdx;
        int lane = tid & 31;
        int warpM = tid >> 7;
        int warpN = (tid >> 5) & 3;
        int colBlocks = d >> 7;
        int grp = ctx.groupIdx;
        int tile = grp / colBlocks;
        if (tile >= tiles.get(0)) {
            return;
        }
        int blockCol = (grp - tile * colBlocks) << 7;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int rows = tiles.get(3 + 3 * tile);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);
        int[] codes = ctx.allocateIntLocalArray(THREADS);
        float[] colScale = ctx.allocateFloatLocalArray(BN);
        float[] colMin = ctx.allocateFloatLocalArray(BN);
        codes[tid] = TransformerBatchPrefillKernels.fp16BitsOf((float) (tid - 128));
        float[] s00 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float[] s01 = ctx.mmaFragment(0.0f);
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float[] s02 = ctx.mmaFragment(0.0f);
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float[] s03 = ctx.mmaFragment(0.0f);
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float[] s10 = ctx.mmaFragment(0.0f);
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float[] s11 = ctx.mmaFragment(0.0f);
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float[] s12 = ctx.mmaFragment(0.0f);
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float[] s13 = ctx.mmaFragment(0.0f);
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        int aIdx0 = tid;
        int aIdx1 = tid + 256;
        int aBase0 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx0 >>> 3, rows - 1)) * n
                        + ((aIdx0 & 7) << 1);
        int aBase1 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx1 >>> 3, rows - 1)) * n
                        + ((aIdx1 & 7) << 1);
        int pairCol = (tid >> 2) << 1;
        int kQuarter = (tid & 3) << 2;
        int bSlot = (pairCol >> 3) * 64 + kQuarter * 4 + ((pairCol & 7) >> 1);
        int wRow = expert * d + blockCol + pairCol;
        int blocksPerRow = n >> 8;

        int r0 = warpM * 32 + (lane >> 2);
        int r1 = r0 + 16;
        int lc0 = warpN * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int sumsPerRow = n >> 5;
        int xr0 = sourceRow(sortedToken, gather, first, Math.min(r0, rows - 1)) * sumsPerRow;
        int xr1 = sourceRow(sortedToken, gather, first, Math.min(r0 + 8, rows - 1)) * sumsPerRow;
        int xr2 = sourceRow(sortedToken, gather, first, Math.min(r1, rows - 1)) * sumsPerRow;
        int xr3 = sourceRow(sortedToken, gather, first, Math.min(r1 + 8, rows - 1)) * sumsPerRow;
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            aTile[aIdx0] = packHalves(input16, aBase0 + kBase);
            aTile[aIdx1] = packHalves(input16, aBase1 + kBase);
            int k = kBase + kQuarter;
            int blk = k >> 8;
            int within = k & 255;
            int pairIndex = within >> 6;
            int high = (within >> 5) & 1;
            int sub = (pairIndex << 1) + high;
            int base0 = (wRow * blocksPerRow + blk) * 176;
            int base1 = base0 + blocksPerRow * 176;
            int qs0 = w.getInt(base0 + 48 + pairIndex * 32 + (within & 31));
            int qs1 = w.getInt(base1 + 48 + pairIndex * 32 + (within & 31));
            int qh0 = w.getInt(base0 + 16 + (within & 31));
            int qh1 = w.getInt(base1 + 16 + (within & 31));
            if (high == 1) {
                qs0 = qs0 >>> 4;
                qs1 = qs1 >>> 4;
            }
            if (sub == 1) {
                qh0 = qh0 >>> 1;
                qh1 = qh1 >>> 1;
            } else if (sub == 2) {
                qh0 = qh0 >>> 2;
                qh1 = qh1 >>> 2;
            } else if (sub == 3) {
                qh0 = qh0 >>> 3;
                qh1 = qh1 >>> 3;
            } else if (sub == 4) {
                qh0 = qh0 >>> 4;
                qh1 = qh1 >>> 4;
            } else if (sub == 5) {
                qh0 = qh0 >>> 5;
                qh1 = qh1 >>> 5;
            } else if (sub == 6) {
                qh0 = qh0 >>> 6;
                qh1 = qh1 >>> 6;
            } else if (sub == 7) {
                qh0 = qh0 >>> 7;
                qh1 = qh1 >>> 7;
            }
            qs0 = (qs0 & 0x0F0F0F0F) | ((qh0 & 0x01010101) << 4);
            qs1 = (qs1 & 0x0F0F0F0F) | ((qh1 & 0x01010101) << 4);
            bTile[bSlot] = codes[128 + (qs0 & 31)] | (codes[128 + (qs1 & 31)] << 16);
            bTile[bSlot + 4] =
                    codes[128 + ((qs0 >>> 8) & 31)] | (codes[128 + ((qs1 >>> 8) & 31)] << 16);
            bTile[bSlot + 8] =
                    codes[128 + ((qs0 >>> 16) & 31)] | (codes[128 + ((qs1 >>> 16) & 31)] << 16);
            bTile[bSlot + 12] =
                    codes[128 + ((qs0 >>> 24) & 31)] | (codes[128 + ((qs1 >>> 24) & 31)] << 16);
            if (kQuarter == 0) {
                int sm0 = TransformerComputeKernelsQ4_K.scaleAndMin(w, base0 + 4, sub);
                int sm1 = TransformerComputeKernelsQ4_K.scaleAndMin(w, base1 + 4, sub);
                colScale[pairCol] = w.getHalfFloat(base0).getFloat32() * (sm0 >> 8);
                colScale[pairCol + 1] = w.getHalfFloat(base1).getFloat32() * (sm1 >> 8);
                colMin[pairCol] = w.getHalfFloat(base0 + 2).getFloat32() * (sm0 & 0xFF);
                colMin[pairCol + 1] = w.getHalfFloat(base1 + 2).getFloat32() * (sm1 & 0xFF);
            }
            ctx.localBarrier();
            // A warp whose rows all lie past the tile's multiplies nothing it stores.
            if (warpM * 32 < rows) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, warpM * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, warpM * 1024 + 512);
                HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 0) * B_SUBTILE_BYTES);
                HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 1) * B_SUBTILE_BYTES);
                HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 2) * B_SUBTILE_BYTES);
                HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 3) * B_SUBTILE_BYTES);
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
            }
            if ((kStep & 1) == 1) {
                int sb = kBase >> 5;
                float x0 = blockSums.get(xr0 + sb);
                float x1 = blockSums.get(xr1 + sb);
                float x2 = blockSums.get(xr2 + sb);
                float x3 = blockSums.get(xr3 + sb);
                t00_0 += colScale[lc0] * s00[0] - colMin[lc0] * x0;
                t00_1 += colScale[lc0 + 1] * s00[1] - colMin[lc0 + 1] * x0;
                t00_2 += colScale[lc0] * s00[2] - colMin[lc0] * x1;
                t00_3 += colScale[lc0 + 1] * s00[3] - colMin[lc0 + 1] * x1;
                t01_0 += colScale[lc1] * s01[0] - colMin[lc1] * x0;
                t01_1 += colScale[lc1 + 1] * s01[1] - colMin[lc1 + 1] * x0;
                t01_2 += colScale[lc1] * s01[2] - colMin[lc1] * x1;
                t01_3 += colScale[lc1 + 1] * s01[3] - colMin[lc1 + 1] * x1;
                t02_0 += colScale[lc2] * s02[0] - colMin[lc2] * x0;
                t02_1 += colScale[lc2 + 1] * s02[1] - colMin[lc2 + 1] * x0;
                t02_2 += colScale[lc2] * s02[2] - colMin[lc2] * x1;
                t02_3 += colScale[lc2 + 1] * s02[3] - colMin[lc2 + 1] * x1;
                t03_0 += colScale[lc3] * s03[0] - colMin[lc3] * x0;
                t03_1 += colScale[lc3 + 1] * s03[1] - colMin[lc3 + 1] * x0;
                t03_2 += colScale[lc3] * s03[2] - colMin[lc3] * x1;
                t03_3 += colScale[lc3 + 1] * s03[3] - colMin[lc3 + 1] * x1;
                t10_0 += colScale[lc0] * s10[0] - colMin[lc0] * x2;
                t10_1 += colScale[lc0 + 1] * s10[1] - colMin[lc0 + 1] * x2;
                t10_2 += colScale[lc0] * s10[2] - colMin[lc0] * x3;
                t10_3 += colScale[lc0 + 1] * s10[3] - colMin[lc0 + 1] * x3;
                t11_0 += colScale[lc1] * s11[0] - colMin[lc1] * x2;
                t11_1 += colScale[lc1 + 1] * s11[1] - colMin[lc1 + 1] * x2;
                t11_2 += colScale[lc1] * s11[2] - colMin[lc1] * x3;
                t11_3 += colScale[lc1 + 1] * s11[3] - colMin[lc1 + 1] * x3;
                t12_0 += colScale[lc2] * s12[0] - colMin[lc2] * x2;
                t12_1 += colScale[lc2 + 1] * s12[1] - colMin[lc2 + 1] * x2;
                t12_2 += colScale[lc2] * s12[2] - colMin[lc2] * x3;
                t12_3 += colScale[lc2 + 1] * s12[3] - colMin[lc2 + 1] * x3;
                t13_0 += colScale[lc3] * s13[0] - colMin[lc3] * x2;
                t13_1 += colScale[lc3 + 1] * s13[1] - colMin[lc3 + 1] * x2;
                t13_2 += colScale[lc3] * s13[2] - colMin[lc3] * x3;
                t13_3 += colScale[lc3 + 1] * s13[3] - colMin[lc3 + 1] * x3;
                s00 = ctx.mmaFragment(0.0f);
                s01 = ctx.mmaFragment(0.0f);
                s02 = ctx.mmaFragment(0.0f);
                s03 = ctx.mmaFragment(0.0f);
                s10 = ctx.mmaFragment(0.0f);
                s11 = ctx.mmaFragment(0.0f);
                s12 = ctx.mmaFragment(0.0f);
                s13 = ctx.mmaFragment(0.0f);
            }
            ctx.localBarrier();
        }

        int col0 = blockCol + lc0;
        int col1 = blockCol + lc1;
        int col2 = blockCol + lc2;
        int col3 = blockCol + lc3;
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col0, t00_0);
            outputBatch.set((first + r0) * d + col0 + 1, t00_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col0, t00_2);
            outputBatch.set((first + r0 + 8) * d + col0 + 1, t00_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col1, t01_0);
            outputBatch.set((first + r0) * d + col1 + 1, t01_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col1, t01_2);
            outputBatch.set((first + r0 + 8) * d + col1 + 1, t01_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col2, t02_0);
            outputBatch.set((first + r0) * d + col2 + 1, t02_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col2, t02_2);
            outputBatch.set((first + r0 + 8) * d + col2 + 1, t02_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col3, t03_0);
            outputBatch.set((first + r0) * d + col3 + 1, t03_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col3, t03_2);
            outputBatch.set((first + r0 + 8) * d + col3 + 1, t03_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col0, t10_0);
            outputBatch.set((first + r1) * d + col0 + 1, t10_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col0, t10_2);
            outputBatch.set((first + r1 + 8) * d + col0 + 1, t10_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col1, t11_0);
            outputBatch.set((first + r1) * d + col1 + 1, t11_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col1, t11_2);
            outputBatch.set((first + r1 + 8) * d + col1 + 1, t11_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col2, t12_0);
            outputBatch.set((first + r1) * d + col2 + 1, t12_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col2, t12_2);
            outputBatch.set((first + r1 + 8) * d + col2 + 1, t12_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col3, t13_0);
            outputBatch.set((first + r1) * d + col3 + 1, t13_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col3, t13_2);
            outputBatch.set((first + r1 + 8) * d + col3 + 1, t13_3);
        }
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q6_K experts on tensor cores (CUDA): {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)} per
     * tile, one workgroup per tile and 128 outputs. Row {@code i} of a tile reads FP16 input row
     * {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise. The
     * weights enter the MMA as their integer codes, exact in FP16, and each scale group's sums are
     * scaled in FP32 by its scale; only the tile's rows are written, to output row {@code first +
     * i}. Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups of {@link
     * #THREADS} threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertMmaQ6_K(
            KernelContext ctx,
            HalfFloatArray input16,
            FloatArray blockSums,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        int tid = ctx.localIdx;
        int lane = tid & 31;
        int warpM = tid >> 7;
        int warpN = (tid >> 5) & 3;
        int colBlocks = d >> 7;
        int grp = ctx.groupIdx;
        int tile = grp / colBlocks;
        if (tile >= tiles.get(0)) {
            return;
        }
        int blockCol = (grp - tile * colBlocks) << 7;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int rows = tiles.get(3 + 3 * tile);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);
        int[] codes = ctx.allocateIntLocalArray(THREADS);
        float[] colScale = ctx.allocateFloatLocalArray(BN);
        codes[tid] = TransformerBatchPrefillKernels.fp16BitsOf((float) (tid - 128));
        float[] s00 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float[] s01 = ctx.mmaFragment(0.0f);
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float[] s02 = ctx.mmaFragment(0.0f);
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float[] s03 = ctx.mmaFragment(0.0f);
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float[] s10 = ctx.mmaFragment(0.0f);
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float[] s11 = ctx.mmaFragment(0.0f);
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float[] s12 = ctx.mmaFragment(0.0f);
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float[] s13 = ctx.mmaFragment(0.0f);
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        int aIdx0 = tid;
        int aIdx1 = tid + 256;
        int aBase0 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx0 >>> 3, rows - 1)) * n
                        + ((aIdx0 & 7) << 1);
        int aBase1 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx1 >>> 3, rows - 1)) * n
                        + ((aIdx1 & 7) << 1);
        int pairCol = (tid >> 2) << 1;
        int kQuarter = (tid & 3) << 2;
        int bSlot = (pairCol >> 3) * 64 + kQuarter * 4 + ((pairCol & 7) >> 1);
        int wRow = expert * d + blockCol + pairCol;
        int blocksPerRow = n >> 8;

        int r0 = warpM * 32 + (lane >> 2);
        int r1 = r0 + 16;
        int lc0 = warpN * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            aTile[aIdx0] = packHalves(input16, aBase0 + kBase);
            aTile[aIdx1] = packHalves(input16, aBase1 + kBase);
            int k = kBase + kQuarter;
            int blk = k >> 8;
            int within = k & 255;
            int halfIndex = within >> 7;
            int groupInHalf = (within >> 5) & 3;
            int posInGroup = within & 31;
            int base0 = (wRow * blocksPerRow + blk) * 210;
            int base1 = base0 + blocksPerRow * 210;
            int ql0 = base0 + halfIndex * 64 + (groupInHalf & 1) * 32 + posInGroup;
            int ql1 = base1 + halfIndex * 64 + (groupInHalf & 1) * 32 + posInGroup;
            int qh0 = base0 + 128 + halfIndex * 32 + posInGroup;
            int qh1 = base1 + 128 + halfIndex * 32 + posInGroup;
            int ql0_0 = w.get(ql0 + 0) & 0xFF;
            int ql1_0 = w.get(ql1 + 0) & 0xFF;
            int qh0_0 = w.get(qh0 + 0) & 0xFF;
            int qh1_0 = w.get(qh1 + 0) & 0xFF;
            int ql0_1 = w.get(ql0 + 1) & 0xFF;
            int ql1_1 = w.get(ql1 + 1) & 0xFF;
            int qh0_1 = w.get(qh0 + 1) & 0xFF;
            int qh1_1 = w.get(qh1 + 1) & 0xFF;
            int ql0_2 = w.get(ql0 + 2) & 0xFF;
            int ql1_2 = w.get(ql1 + 2) & 0xFF;
            int qh0_2 = w.get(qh0 + 2) & 0xFF;
            int qh1_2 = w.get(qh1 + 2) & 0xFF;
            int ql0_3 = w.get(ql0 + 3) & 0xFF;
            int ql1_3 = w.get(ql1 + 3) & 0xFF;
            int qh0_3 = w.get(qh0 + 3) & 0xFF;
            int qh1_3 = w.get(qh1 + 3) & 0xFF;
            int v0_0 = ql0_0 & 15;
            int v1_0 = ql1_0 & 15;
            int v0_1 = ql0_1 & 15;
            int v1_1 = ql1_1 & 15;
            int v0_2 = ql0_2 & 15;
            int v1_2 = ql1_2 & 15;
            int v0_3 = ql0_3 & 15;
            int v1_3 = ql1_3 & 15;
            if ((groupInHalf >> 1) == 1) {
                v0_0 = (ql0_0 >> 4) & 15;
                v1_0 = (ql1_0 >> 4) & 15;
                v0_1 = (ql0_1 >> 4) & 15;
                v1_1 = (ql1_1 >> 4) & 15;
                v0_2 = (ql0_2 >> 4) & 15;
                v1_2 = (ql1_2 >> 4) & 15;
                v0_3 = (ql0_3 >> 4) & 15;
                v1_3 = (ql1_3 >> 4) & 15;
            }
            int h0_0 = qh0_0 & 3;
            int h1_0 = qh1_0 & 3;
            int h0_1 = qh0_1 & 3;
            int h1_1 = qh1_1 & 3;
            int h0_2 = qh0_2 & 3;
            int h1_2 = qh1_2 & 3;
            int h0_3 = qh0_3 & 3;
            int h1_3 = qh1_3 & 3;
            if (groupInHalf == 1) {
                h0_0 = (qh0_0 >> 2) & 3;
                h1_0 = (qh1_0 >> 2) & 3;
                h0_1 = (qh0_1 >> 2) & 3;
                h1_1 = (qh1_1 >> 2) & 3;
                h0_2 = (qh0_2 >> 2) & 3;
                h1_2 = (qh1_2 >> 2) & 3;
                h0_3 = (qh0_3 >> 2) & 3;
                h1_3 = (qh1_3 >> 2) & 3;
            } else if (groupInHalf == 2) {
                h0_0 = (qh0_0 >> 4) & 3;
                h1_0 = (qh1_0 >> 4) & 3;
                h0_1 = (qh0_1 >> 4) & 3;
                h1_1 = (qh1_1 >> 4) & 3;
                h0_2 = (qh0_2 >> 4) & 3;
                h1_2 = (qh1_2 >> 4) & 3;
                h0_3 = (qh0_3 >> 4) & 3;
                h1_3 = (qh1_3 >> 4) & 3;
            } else if (groupInHalf == 3) {
                h0_0 = (qh0_0 >> 6) & 3;
                h1_0 = (qh1_0 >> 6) & 3;
                h0_1 = (qh0_1 >> 6) & 3;
                h1_1 = (qh1_1 >> 6) & 3;
                h0_2 = (qh0_2 >> 6) & 3;
                h1_2 = (qh1_2 >> 6) & 3;
                h0_3 = (qh0_3 >> 6) & 3;
                h1_3 = (qh1_3 >> 6) & 3;
            }
            bTile[bSlot + 0] =
                    codes[96 + (v0_0 | (h0_0 << 4))] | (codes[96 + (v1_0 | (h1_0 << 4))] << 16);
            bTile[bSlot + 4] =
                    codes[96 + (v0_1 | (h0_1 << 4))] | (codes[96 + (v1_1 | (h1_1 << 4))] << 16);
            bTile[bSlot + 8] =
                    codes[96 + (v0_2 | (h0_2 << 4))] | (codes[96 + (v1_2 | (h1_2 << 4))] << 16);
            bTile[bSlot + 12] =
                    codes[96 + (v0_3 | (h0_3 << 4))] | (codes[96 + (v1_3 | (h1_3 << 4))] << 16);
            if (kQuarter == 0) {
                int sc0 = w.get(base0 + 192 + halfIndex * 8 + (posInGroup >> 4) + groupInHalf * 2);
                int sc1 = w.get(base1 + 192 + halfIndex * 8 + (posInGroup >> 4) + groupInHalf * 2);
                colScale[pairCol] = w.getHalfFloat(base0 + 208).getFloat32() * sc0;
                colScale[pairCol + 1] = w.getHalfFloat(base1 + 208).getFloat32() * sc1;
            }
            ctx.localBarrier();
            // A warp whose rows all lie past the tile's multiplies nothing it stores.
            if (warpM * 32 < rows) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, warpM * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, warpM * 1024 + 512);
                HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 0) * B_SUBTILE_BYTES);
                HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 1) * B_SUBTILE_BYTES);
                HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 2) * B_SUBTILE_BYTES);
                HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 3) * B_SUBTILE_BYTES);
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
            }
            if (true) {
                t00_0 += colScale[lc0] * s00[0];
                t00_1 += colScale[lc0 + 1] * s00[1];
                t00_2 += colScale[lc0] * s00[2];
                t00_3 += colScale[lc0 + 1] * s00[3];
                t01_0 += colScale[lc1] * s01[0];
                t01_1 += colScale[lc1 + 1] * s01[1];
                t01_2 += colScale[lc1] * s01[2];
                t01_3 += colScale[lc1 + 1] * s01[3];
                t02_0 += colScale[lc2] * s02[0];
                t02_1 += colScale[lc2 + 1] * s02[1];
                t02_2 += colScale[lc2] * s02[2];
                t02_3 += colScale[lc2 + 1] * s02[3];
                t03_0 += colScale[lc3] * s03[0];
                t03_1 += colScale[lc3 + 1] * s03[1];
                t03_2 += colScale[lc3] * s03[2];
                t03_3 += colScale[lc3 + 1] * s03[3];
                t10_0 += colScale[lc0] * s10[0];
                t10_1 += colScale[lc0 + 1] * s10[1];
                t10_2 += colScale[lc0] * s10[2];
                t10_3 += colScale[lc0 + 1] * s10[3];
                t11_0 += colScale[lc1] * s11[0];
                t11_1 += colScale[lc1 + 1] * s11[1];
                t11_2 += colScale[lc1] * s11[2];
                t11_3 += colScale[lc1 + 1] * s11[3];
                t12_0 += colScale[lc2] * s12[0];
                t12_1 += colScale[lc2 + 1] * s12[1];
                t12_2 += colScale[lc2] * s12[2];
                t12_3 += colScale[lc2 + 1] * s12[3];
                t13_0 += colScale[lc3] * s13[0];
                t13_1 += colScale[lc3 + 1] * s13[1];
                t13_2 += colScale[lc3] * s13[2];
                t13_3 += colScale[lc3 + 1] * s13[3];
                s00 = ctx.mmaFragment(0.0f);
                s01 = ctx.mmaFragment(0.0f);
                s02 = ctx.mmaFragment(0.0f);
                s03 = ctx.mmaFragment(0.0f);
                s10 = ctx.mmaFragment(0.0f);
                s11 = ctx.mmaFragment(0.0f);
                s12 = ctx.mmaFragment(0.0f);
                s13 = ctx.mmaFragment(0.0f);
            }
            ctx.localBarrier();
        }

        int col0 = blockCol + lc0;
        int col1 = blockCol + lc1;
        int col2 = blockCol + lc2;
        int col3 = blockCol + lc3;
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col0, t00_0);
            outputBatch.set((first + r0) * d + col0 + 1, t00_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col0, t00_2);
            outputBatch.set((first + r0 + 8) * d + col0 + 1, t00_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col1, t01_0);
            outputBatch.set((first + r0) * d + col1 + 1, t01_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col1, t01_2);
            outputBatch.set((first + r0 + 8) * d + col1 + 1, t01_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col2, t02_0);
            outputBatch.set((first + r0) * d + col2 + 1, t02_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col2, t02_2);
            outputBatch.set((first + r0 + 8) * d + col2 + 1, t02_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col3, t03_0);
            outputBatch.set((first + r0) * d + col3 + 1, t03_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col3, t03_2);
            outputBatch.set((first + r0 + 8) * d + col3 + 1, t03_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col0, t10_0);
            outputBatch.set((first + r1) * d + col0 + 1, t10_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col0, t10_2);
            outputBatch.set((first + r1 + 8) * d + col0 + 1, t10_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col1, t11_0);
            outputBatch.set((first + r1) * d + col1 + 1, t11_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col1, t11_2);
            outputBatch.set((first + r1 + 8) * d + col1 + 1, t11_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col2, t12_0);
            outputBatch.set((first + r1) * d + col2 + 1, t12_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col2, t12_2);
            outputBatch.set((first + r1 + 8) * d + col2 + 1, t12_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col3, t13_0);
            outputBatch.set((first + r1) * d + col3 + 1, t13_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col3, t13_2);
            outputBatch.set((first + r1 + 8) * d + col3 + 1, t13_3);
        }
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q8_0 experts on tensor cores (CUDA): {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)} per
     * tile, one workgroup per tile and 128 outputs. Row {@code i} of a tile reads FP16 input row
     * {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise. The
     * weights enter the MMA as their integer codes, exact in FP16, and each scale group's sums are
     * scaled in FP32 by its scale; only the tile's rows are written, to output row {@code first +
     * i}. Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups of {@link
     * #THREADS} threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertMmaQ8_0(
            KernelContext ctx,
            HalfFloatArray input16,
            FloatArray blockSums,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        int tid = ctx.localIdx;
        int lane = tid & 31;
        int warpM = tid >> 7;
        int warpN = (tid >> 5) & 3;
        int colBlocks = d >> 7;
        int grp = ctx.groupIdx;
        int tile = grp / colBlocks;
        if (tile >= tiles.get(0)) {
            return;
        }
        int blockCol = (grp - tile * colBlocks) << 7;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int rows = tiles.get(3 + 3 * tile);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);
        int[] codes = ctx.allocateIntLocalArray(THREADS);
        float[] colScale = ctx.allocateFloatLocalArray(BN);
        codes[tid] = TransformerBatchPrefillKernels.fp16BitsOf((float) (tid - 128));
        float[] s00 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float[] s01 = ctx.mmaFragment(0.0f);
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float[] s02 = ctx.mmaFragment(0.0f);
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float[] s03 = ctx.mmaFragment(0.0f);
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float[] s10 = ctx.mmaFragment(0.0f);
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float[] s11 = ctx.mmaFragment(0.0f);
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float[] s12 = ctx.mmaFragment(0.0f);
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float[] s13 = ctx.mmaFragment(0.0f);
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        int aIdx0 = tid;
        int aIdx1 = tid + 256;
        int aBase0 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx0 >>> 3, rows - 1)) * n
                        + ((aIdx0 & 7) << 1);
        int aBase1 =
                sourceRow(sortedToken, gather, first, Math.min(aIdx1 >>> 3, rows - 1)) * n
                        + ((aIdx1 & 7) << 1);
        int pairCol = (tid >> 2) << 1;
        int kQuarter = (tid & 3) << 2;
        int bSlot = (pairCol >> 3) * 64 + kQuarter * 4 + ((pairCol & 7) >> 1);
        int wRow = expert * d + blockCol + pairCol;
        int blocksPerRow = n >> 5;

        int r0 = warpM * 32 + (lane >> 2);
        int r1 = r0 + 16;
        int lc0 = warpN * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            aTile[aIdx0] = packHalves(input16, aBase0 + kBase);
            aTile[aIdx1] = packHalves(input16, aBase1 + kBase);
            int k = kBase + kQuarter;
            int base0 = (wRow * blocksPerRow + (k >> 5)) * 34;
            int base1 = base0 + blocksPerRow * 34;
            int q0 = base0 + 2 + (k & 31);
            int q1 = base1 + 2 + (k & 31);
            bTile[bSlot + 0] = codes[128 + w.get(q0 + 0)] | (codes[128 + w.get(q1 + 0)] << 16);
            bTile[bSlot + 4] = codes[128 + w.get(q0 + 1)] | (codes[128 + w.get(q1 + 1)] << 16);
            bTile[bSlot + 8] = codes[128 + w.get(q0 + 2)] | (codes[128 + w.get(q1 + 2)] << 16);
            bTile[bSlot + 12] = codes[128 + w.get(q0 + 3)] | (codes[128 + w.get(q1 + 3)] << 16);
            if (kQuarter == 0) {
                colScale[pairCol] = w.getHalfFloat(base0).getFloat32();
                colScale[pairCol + 1] = w.getHalfFloat(base1).getFloat32();
            }
            ctx.localBarrier();
            // A warp whose rows all lie past the tile's multiplies nothing it stores.
            if (warpM * 32 < rows) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, warpM * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, warpM * 1024 + 512);
                HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 0) * B_SUBTILE_BYTES);
                HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 1) * B_SUBTILE_BYTES);
                HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 2) * B_SUBTILE_BYTES);
                HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (warpN * 4 + 3) * B_SUBTILE_BYTES);
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
            }
            if ((kStep & 1) == 1) {
                t00_0 += colScale[lc0] * s00[0];
                t00_1 += colScale[lc0 + 1] * s00[1];
                t00_2 += colScale[lc0] * s00[2];
                t00_3 += colScale[lc0 + 1] * s00[3];
                t01_0 += colScale[lc1] * s01[0];
                t01_1 += colScale[lc1 + 1] * s01[1];
                t01_2 += colScale[lc1] * s01[2];
                t01_3 += colScale[lc1 + 1] * s01[3];
                t02_0 += colScale[lc2] * s02[0];
                t02_1 += colScale[lc2 + 1] * s02[1];
                t02_2 += colScale[lc2] * s02[2];
                t02_3 += colScale[lc2 + 1] * s02[3];
                t03_0 += colScale[lc3] * s03[0];
                t03_1 += colScale[lc3 + 1] * s03[1];
                t03_2 += colScale[lc3] * s03[2];
                t03_3 += colScale[lc3 + 1] * s03[3];
                t10_0 += colScale[lc0] * s10[0];
                t10_1 += colScale[lc0 + 1] * s10[1];
                t10_2 += colScale[lc0] * s10[2];
                t10_3 += colScale[lc0 + 1] * s10[3];
                t11_0 += colScale[lc1] * s11[0];
                t11_1 += colScale[lc1 + 1] * s11[1];
                t11_2 += colScale[lc1] * s11[2];
                t11_3 += colScale[lc1 + 1] * s11[3];
                t12_0 += colScale[lc2] * s12[0];
                t12_1 += colScale[lc2 + 1] * s12[1];
                t12_2 += colScale[lc2] * s12[2];
                t12_3 += colScale[lc2 + 1] * s12[3];
                t13_0 += colScale[lc3] * s13[0];
                t13_1 += colScale[lc3 + 1] * s13[1];
                t13_2 += colScale[lc3] * s13[2];
                t13_3 += colScale[lc3 + 1] * s13[3];
                s00 = ctx.mmaFragment(0.0f);
                s01 = ctx.mmaFragment(0.0f);
                s02 = ctx.mmaFragment(0.0f);
                s03 = ctx.mmaFragment(0.0f);
                s10 = ctx.mmaFragment(0.0f);
                s11 = ctx.mmaFragment(0.0f);
                s12 = ctx.mmaFragment(0.0f);
                s13 = ctx.mmaFragment(0.0f);
            }
            ctx.localBarrier();
        }

        int col0 = blockCol + lc0;
        int col1 = blockCol + lc1;
        int col2 = blockCol + lc2;
        int col3 = blockCol + lc3;
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col0, t00_0);
            outputBatch.set((first + r0) * d + col0 + 1, t00_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col0, t00_2);
            outputBatch.set((first + r0 + 8) * d + col0 + 1, t00_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col1, t01_0);
            outputBatch.set((first + r0) * d + col1 + 1, t01_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col1, t01_2);
            outputBatch.set((first + r0 + 8) * d + col1 + 1, t01_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col2, t02_0);
            outputBatch.set((first + r0) * d + col2 + 1, t02_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col2, t02_2);
            outputBatch.set((first + r0 + 8) * d + col2 + 1, t02_3);
        }
        if (r0 < rows) {
            outputBatch.set((first + r0) * d + col3, t03_0);
            outputBatch.set((first + r0) * d + col3 + 1, t03_1);
        }
        if (r0 + 8 < rows) {
            outputBatch.set((first + r0 + 8) * d + col3, t03_2);
            outputBatch.set((first + r0 + 8) * d + col3 + 1, t03_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col0, t10_0);
            outputBatch.set((first + r1) * d + col0 + 1, t10_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col0, t10_2);
            outputBatch.set((first + r1 + 8) * d + col0 + 1, t10_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col1, t11_0);
            outputBatch.set((first + r1) * d + col1 + 1, t11_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col1, t11_2);
            outputBatch.set((first + r1 + 8) * d + col1 + 1, t11_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col2, t12_0);
            outputBatch.set((first + r1) * d + col2 + 1, t12_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col2, t12_2);
            outputBatch.set((first + r1 + 8) * d + col2 + 1, t12_3);
        }
        if (r1 < rows) {
            outputBatch.set((first + r1) * d + col3, t13_0);
            outputBatch.set((first + r1) * d + col3 + 1, t13_1);
        }
        if (r1 + 8 < rows) {
            outputBatch.set((first + r1 + 8) * d + col3, t13_2);
            outputBatch.set((first + r1 + 8) * d + col3 + 1, t13_3);
        }
    }
}
