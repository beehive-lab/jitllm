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
 * as {@code M16N8K16} MMAs over FP16 inputs; the depth advances 16 at a time, the next step's loads
 * in flight while this step multiplies. Q4_K and Q5_K split a tile in two halves of 32 rows, the
 * typical expert's share of a 512-row chunk, so no warp multiplies empty rows: 256 outputs per
 * workgroup, the depth advancing 64 at a time.
 *
 * <p>The weights enter the MMA as their integer codes, which FP16 holds exactly: for Q6_K and Q8_0
 * a code's bits come from a table in shared memory; Q4_K and Q5_K build each warp's B operand in
 * its own registers ({@code mmaFragmentB}), since a warp is the only reader of its columns. Each
 * scale group's partial products accumulate in their own fragments and are then scaled in FP32 by
 * the group's scale; for Q4_K and Q5_K the group's minimum is subtracted through the input rows'
 * sums per 32 inputs ({@link Lfm2BatchKernels#halvesWithBlockSums}). Example: a Q4_K weight {@code
 * d * sc * q - dmin * m} contributes {@code d * sc * (x . q) - dmin * m * sum(x)} per 32 inputs.
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

    /** Rows of a Q4_K or Q5_K workgroup: half of an expert tile. */
    private static final int SPLIT_ROWS = 32;

    /** Outputs of a Q4_K or Q5_K workgroup: eight warps of 32. */
    private static final int SPLIT_OUTPUTS = 256;

    /** Depth of one step of the Q4_K and Q5_K kernels: a pair of sub-blocks. */
    private static final int SPLIT_DEPTH = 64;

    /** Ints of one input buffer of the Q4_K and Q5_K kernels: four 16-deep chunks of 32 rows. */
    private static final int SPLIT_INPUT_INTS = 1024;

    /**
     * Workgroups of the Q4_K and Q5_K kernels over {@code maxTiles} tiles and {@code d} outputs.
     */
    public static int splitWorkgroups(int maxTiles, int d) {
        return maxTiles * 2 * (d / SPLIT_OUTPUTS);
    }

    /** The input row of a tile's local row {@code local}. */
    private static int sourceRow(IntArray sortedToken, int gather, int first, int local) {
        return gather != 0 ? sortedToken.get(first + local) : first + local;
    }

    /**
     * {@code TransformerComputeKernelsQ4_K.scaleAndMin} of sub-block {@code sub} from the twelve
     * scale bytes held as three words {@code s0..s2}, little-endian: {@code scale << 8 | min}.
     */
    private static int scaleAndMinOf(int s0, int s1, int s2, int sub) {
        if (sub < 4) {
            int sh = 8 * sub;
            return (((s0 >>> sh) & 63) << 8) | ((s1 >>> sh) & 63);
        }
        int sh = 8 * (sub - 4);
        int low = (s2 >>> sh) & 0xFF;
        int sc = (low & 0xF) | ((((s0 >>> sh) & 0xFF) >> 6) << 4);
        int m = ((low >> 4) & 0xF) | ((((s1 >>> sh) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }

    /** Halves {@code i} and {@code i + 1} of {@code x} as one int. */
    private static int packHalves(HalfFloatArray x, int i) {
        return (x.get(i).getHalfFloatValue() & 0xFFFF)
                | ((x.get(i + 1).getHalfFloatValue() & 0xFFFF) << 16);
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q4_K experts on tensor cores (CUDA), {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it: one workgroup per half of a tile (32 rows)
     * and 256 outputs, eight warps of 32 rows by 32 outputs, the depth advancing 64 at a time, a
     * pair of sub-blocks. Row {@code i} of a half reads FP16 input row {@code sortedToken[first +
     * i]} when {@code gather} is 1, {@code first + i} otherwise, copied to local memory 4 bytes at
     * a time in the k order the weights take; the next step's copy and weight loads are in flight
     * while this step multiplies. The weights never reach local memory: lane {@code L} loads the
     * code bytes {@code 8 (L % 4)} to {@code 8 (L % 4) + 7} of its own column {@code L / 4}, one
     * 8-byte load per sub-block pair, and takes the words it multiplies, bytes {@code 4 (L % 4)}
     * and {@code 16 + 4 (L % 4)}, from its column's four lanes with {@code simdShuffle}: each MMA
     * sums the same 16 inputs as with 4-byte loads, so the result is bit-identical. It turns each
     * code into its exact half in registers and passes them to the MMA with {@code mmaFragmentB}.
     * Each scale group's sums are scaled in FP32 by its scale, and the group's minimum subtracted
     * through {@code blockSums}, the FP16 rows' sums per 32 inputs: the sums of one workgroup per
     * 64-row tile, in the same order. A half with no rows of its tile returns before its first
     * barrier. Example: LFM2.5-24B-A2B's gate, 1536 outputs over {@code n = 2048}: six workgroups
     * per half and 32 steps. Requires {@code n % 256 == 0} and {@code d % 256 == 0}. Grid: {@link
     * #splitWorkgroups} workgroups of {@link #THREADS} threads.
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
        int warp = tid >> 5;
        int colBlocks = d >> 8;
        int grp = ctx.groupIdx;
        int tileHalf = grp / colBlocks;
        int tile = tileHalf >> 1;
        if (tile >= tiles.get(0)) {
            return;
        }
        int rowStart = (tileHalf & 1) * SPLIT_ROWS;
        int tileRows = tiles.get(3 + 3 * tile);
        if (rowStart >= tileRows) {
            return;
        }
        int rows = Math.min(SPLIT_ROWS, tileRows - rowStart);
        int blockCol = (grp - tileHalf * colBlocks) << 8;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile) + rowStart;

        int[] aTile = ctx.allocateIntLocalArray(2 * SPLIT_INPUT_INTS);
        float[] colScale = ctx.allocateFloatLocalArray(2 * SPLIT_OUTPUTS);
        float[] colMin = ctx.allocateFloatLocalArray(2 * SPLIT_OUTPUTS);
        float[] s00 = ctx.mmaFragment(0.0f);
        float[] s01 = ctx.mmaFragment(0.0f);
        float[] s02 = ctx.mmaFragment(0.0f);
        float[] s03 = ctx.mmaFragment(0.0f);
        float[] s10 = ctx.mmaFragment(0.0f);
        float[] s11 = ctx.mmaFragment(0.0f);
        float[] s12 = ctx.mmaFragment(0.0f);
        float[] s13 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        // A: one 16-byte copy per thread and step, row tid / 8, eight halves of a 16-deep chunk.
        int aRow = tid >> 3;
        int aChunk = (tid & 7) >> 1;
        int aHalf = tid & 1;
        int aSrc =
                sourceRow(sortedToken, gather, first, Math.min(aRow, rows - 1)) * n
                        + aChunk * 16
                        + aHalf * 2;
        int aDst = aChunk * 256 + aRow * 8 + aHalf * 4;
        // B: four column pairs per thread, word i of each column's 32 code bytes for the step.
        int bCol = warp * 32 + (lane >> 2);
        int bByte = 8 * (lane & 3);
        // Lane t of a column's four holds code words 2t and 2t + 1 of its 8-byte load and needs
        // words t and 4 + t: two shuffles, each source offering its low or high word by which half
        // of the four it is in.
        int bLow = (lane & 3) < 2 ? 1 : 0;
        int bOdd = lane & 1;
        int bSrcA = (lane & ~3) + ((lane & 3) >> 1) + 2 * bOdd;
        int bSrcB = (lane & ~3) + ((lane & 3) >> 1) + 2 * (1 - bOdd);
        int blocksPerRow = n >> 8;
        int rowStride = blocksPerRow * 144;
        int wRow = expert * d + blockCol;
        // Scales: one column per thread, both sub-blocks of the step.
        int sRow = (wRow + tid) * blocksPerRow;

        int r0 = lane >> 2;
        int r1 = r0 + 16;
        int lc0 = warp * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int sumsPerRow = n >> 5;
        int xr0 = sourceRow(sortedToken, gather, first, Math.min(r0, rows - 1)) * sumsPerRow;
        int xr1 = sourceRow(sortedToken, gather, first, Math.min(r0 + 8, rows - 1)) * sumsPerRow;
        int xr2 = sourceRow(sortedToken, gather, first, Math.min(r1, rows - 1)) * sumsPerRow;
        int xr3 = sourceRow(sortedToken, gather, first, Math.min(r1 + 8, rows - 1)) * sumsPerRow;

        int q0a = 0;
        int q0b = 0;
        int q1a = 0;
        int q1b = 0;
        int q2a = 0;
        int q2b = 0;
        int q3a = 0;
        int q3b = 0;
        int rHead = 0;
        int rS0 = 0;
        int rS1 = 0;
        int rS2 = 0;
        int rPair = 0;
        ctx.asyncCopyToLocal(aTile, aDst, input16, aSrc);
        ctx.asyncCopyToLocal(aTile, aDst + 1, input16, aSrc + 4);
        ctx.asyncCopyToLocal(aTile, aDst + 2, input16, aSrc + 8);
        ctx.asyncCopyToLocal(aTile, aDst + 3, input16, aSrc + 12);
        ctx.asyncCopyCommit();
        {
            int k = 0;
            int pairIndex = (k & 255) >> 6;
            int kBlock = k >> 8;
            rPair = pairIndex;
            int qOff = kBlock * 144 + 16 + pairIndex * 32 + bByte;
            int c0 = (wRow + bCol) * rowStride + qOff;
            long q0aw = w.getLong(c0);
            q0a = (int) q0aw;
            q0b = (int) (q0aw >>> 32);
            int c1 = c0 + 8 * rowStride;
            long q1aw = w.getLong(c1);
            q1a = (int) q1aw;
            q1b = (int) (q1aw >>> 32);
            int c2 = c0 + 16 * rowStride;
            long q2aw = w.getLong(c2);
            q2a = (int) q2aw;
            q2b = (int) (q2aw >>> 32);
            int c3 = c0 + 24 * rowStride;
            long q3aw = w.getLong(c3);
            q3a = (int) q3aw;
            q3b = (int) (q3aw >>> 32);
            int sBase = (sRow + kBlock) * 144;
            long rHeadw = w.getLong(sBase);
            rHead = (int) rHeadw;
            rS0 = (int) (rHeadw >>> 32);
            long rS1w = w.getLong(sBase + 8);
            rS1 = (int) rS1w;
            rS2 = (int) (rS1w >>> 32);
        }

        int kSteps = n >> 6;
        for (int step = 0; step < kSteps; step++) {
            int kBase = step << 6;
            int buf = step & 1;
            // This step's code words; the registers are reloaded for the next step below.
            int u0x = ctx.simdShuffle(bLow == 1 ? q0a : q0b, bSrcA);
            int u0y = ctx.simdShuffle(bLow == 1 ? q0b : q0a, bSrcB);
            int u0a = bOdd == 0 ? u0x : u0y;
            int u0b = bOdd == 0 ? u0y : u0x;
            int u1x = ctx.simdShuffle(bLow == 1 ? q1a : q1b, bSrcA);
            int u1y = ctx.simdShuffle(bLow == 1 ? q1b : q1a, bSrcB);
            int u1a = bOdd == 0 ? u1x : u1y;
            int u1b = bOdd == 0 ? u1y : u1x;
            int u2x = ctx.simdShuffle(bLow == 1 ? q2a : q2b, bSrcA);
            int u2y = ctx.simdShuffle(bLow == 1 ? q2b : q2a, bSrcB);
            int u2a = bOdd == 0 ? u2x : u2y;
            int u2b = bOdd == 0 ? u2y : u2x;
            int u3x = ctx.simdShuffle(bLow == 1 ? q3a : q3b, bSrcA);
            int u3y = ctx.simdShuffle(bLow == 1 ? q3b : q3a, bSrcB);
            int u3a = bOdd == 0 ? u3x : u3y;
            int u3b = bOdd == 0 ? u3y : u3x;
            {
                // The step's scale words, loaded raw a step ahead, decoded after the multiplies
                // they overlapped.
                float dd = new HalfFloat((short) rHead).getFloat32();
                float dm = new HalfFloat((short) (rHead >>> 16)).getFloat32();
                int sm0 = scaleAndMinOf(rS0, rS1, rS2, 2 * rPair);
                int sm1 = scaleAndMinOf(rS0, rS1, rS2, 2 * rPair + 1);
                colScale[tid] = dd * (sm0 >> 8);
                colScale[SPLIT_OUTPUTS + tid] = dd * (sm1 >> 8);
                colMin[tid] = dm * (sm0 & 0xFF);
                colMin[SPLIT_OUTPUTS + tid] = dm * (sm1 & 0xFF);
            }
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();
            if (step + 1 < kSteps) {
                int nextDst = (buf ^ 1) * SPLIT_INPUT_INTS + aDst;
                int nextSrc = aSrc + kBase + SPLIT_DEPTH;
                ctx.asyncCopyToLocal(aTile, nextDst, input16, nextSrc);
                ctx.asyncCopyToLocal(aTile, nextDst + 1, input16, nextSrc + 4);
                ctx.asyncCopyToLocal(aTile, nextDst + 2, input16, nextSrc + 8);
                ctx.asyncCopyToLocal(aTile, nextDst + 3, input16, nextSrc + 12);
                ctx.asyncCopyCommit();
                {
                    int k = kBase + SPLIT_DEPTH;
                    int pairIndex = (k & 255) >> 6;
                    int kBlock = k >> 8;
                    rPair = pairIndex;
                    int qOff = kBlock * 144 + 16 + pairIndex * 32 + bByte;
                    int c0 = (wRow + bCol) * rowStride + qOff;
                    long q0aw = w.getLong(c0);
                    q0a = (int) q0aw;
                    q0b = (int) (q0aw >>> 32);
                    int c1 = c0 + 8 * rowStride;
                    long q1aw = w.getLong(c1);
                    q1a = (int) q1aw;
                    q1b = (int) (q1aw >>> 32);
                    int c2 = c0 + 16 * rowStride;
                    long q2aw = w.getLong(c2);
                    q2a = (int) q2aw;
                    q2b = (int) (q2aw >>> 32);
                    int c3 = c0 + 24 * rowStride;
                    long q3aw = w.getLong(c3);
                    q3a = (int) q3aw;
                    q3b = (int) (q3aw >>> 32);
                    int sBase = (sRow + kBlock) * 144;
                    long rHeadw = w.getLong(sBase);
                    rHead = (int) rHeadw;
                    rS0 = (int) (rHeadw >>> 32);
                    long rS1w = w.getLong(sBase + 8);
                    rS1 = (int) rS1w;
                    rS2 = (int) (rS1w >>> 32);
                }
            }
            int aByte = buf * SPLIT_INPUT_INTS * 4;
            for (int c = 0; c < 4; c++) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024 + 512);
                // Chunk c: words a (c even) or b (c odd), low nibbles (c < 2) or high; each code
                // becomes its exact half in the lane's own registers.
                int sh = (c >> 1) << 2;
                int v0 = (((c & 1) == 0 ? u0a : u0b) >>> sh) & 0x0F0F0F0F;
                int v1 = (((c & 1) == 0 ? u1a : u1b) >>> sh) & 0x0F0F0F0F;
                int v2 = (((c & 1) == 0 ? u2a : u2b) >>> sh) & 0x0F0F0F0F;
                int v3 = (((c & 1) == 0 ? u3a : u3b) >>> sh) & 0x0F0F0F0F;
                HalfFloat[] b0 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v0 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v0 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v0 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v0 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b1 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v1 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v1 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v1 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v1 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b2 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v2 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v2 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v2 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v2 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b3 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v3 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v3 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v3 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v3 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
                if ((c & 1) == 1) {
                    int g = (c >> 1) * SPLIT_OUTPUTS;
                    int sb = (kBase >> 5) + (c >> 1);
                    float x0 = blockSums.get(xr0 + sb);
                    float x1 = blockSums.get(xr1 + sb);
                    float x2 = blockSums.get(xr2 + sb);
                    float x3 = blockSums.get(xr3 + sb);
                    t00_0 += colScale[g + lc0] * s00[0] - colMin[g + lc0] * x0;
                    t00_1 += colScale[g + lc0 + 1] * s00[1] - colMin[g + lc0 + 1] * x0;
                    t00_2 += colScale[g + lc0] * s00[2] - colMin[g + lc0] * x1;
                    t00_3 += colScale[g + lc0 + 1] * s00[3] - colMin[g + lc0 + 1] * x1;
                    t01_0 += colScale[g + lc1] * s01[0] - colMin[g + lc1] * x0;
                    t01_1 += colScale[g + lc1 + 1] * s01[1] - colMin[g + lc1 + 1] * x0;
                    t01_2 += colScale[g + lc1] * s01[2] - colMin[g + lc1] * x1;
                    t01_3 += colScale[g + lc1 + 1] * s01[3] - colMin[g + lc1 + 1] * x1;
                    t02_0 += colScale[g + lc2] * s02[0] - colMin[g + lc2] * x0;
                    t02_1 += colScale[g + lc2 + 1] * s02[1] - colMin[g + lc2 + 1] * x0;
                    t02_2 += colScale[g + lc2] * s02[2] - colMin[g + lc2] * x1;
                    t02_3 += colScale[g + lc2 + 1] * s02[3] - colMin[g + lc2 + 1] * x1;
                    t03_0 += colScale[g + lc3] * s03[0] - colMin[g + lc3] * x0;
                    t03_1 += colScale[g + lc3 + 1] * s03[1] - colMin[g + lc3 + 1] * x0;
                    t03_2 += colScale[g + lc3] * s03[2] - colMin[g + lc3] * x1;
                    t03_3 += colScale[g + lc3 + 1] * s03[3] - colMin[g + lc3 + 1] * x1;
                    t10_0 += colScale[g + lc0] * s10[0] - colMin[g + lc0] * x2;
                    t10_1 += colScale[g + lc0 + 1] * s10[1] - colMin[g + lc0 + 1] * x2;
                    t10_2 += colScale[g + lc0] * s10[2] - colMin[g + lc0] * x3;
                    t10_3 += colScale[g + lc0 + 1] * s10[3] - colMin[g + lc0 + 1] * x3;
                    t11_0 += colScale[g + lc1] * s11[0] - colMin[g + lc1] * x2;
                    t11_1 += colScale[g + lc1 + 1] * s11[1] - colMin[g + lc1 + 1] * x2;
                    t11_2 += colScale[g + lc1] * s11[2] - colMin[g + lc1] * x3;
                    t11_3 += colScale[g + lc1 + 1] * s11[3] - colMin[g + lc1 + 1] * x3;
                    t12_0 += colScale[g + lc2] * s12[0] - colMin[g + lc2] * x2;
                    t12_1 += colScale[g + lc2 + 1] * s12[1] - colMin[g + lc2 + 1] * x2;
                    t12_2 += colScale[g + lc2] * s12[2] - colMin[g + lc2] * x3;
                    t12_3 += colScale[g + lc2 + 1] * s12[3] - colMin[g + lc2 + 1] * x3;
                    t13_0 += colScale[g + lc3] * s13[0] - colMin[g + lc3] * x2;
                    t13_1 += colScale[g + lc3 + 1] * s13[1] - colMin[g + lc3 + 1] * x2;
                    t13_2 += colScale[g + lc3] * s13[2] - colMin[g + lc3] * x3;
                    t13_3 += colScale[g + lc3 + 1] * s13[3] - colMin[g + lc3 + 1] * x3;
                    s00 = ctx.mmaFragment(0.0f);
                    s01 = ctx.mmaFragment(0.0f);
                    s02 = ctx.mmaFragment(0.0f);
                    s03 = ctx.mmaFragment(0.0f);
                    s10 = ctx.mmaFragment(0.0f);
                    s11 = ctx.mmaFragment(0.0f);
                    s12 = ctx.mmaFragment(0.0f);
                    s13 = ctx.mmaFragment(0.0f);
                }
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
     * The grouped expert GEMM over Q5_K experts on tensor cores (CUDA), {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it: one workgroup per half of a tile (32 rows)
     * and 256 outputs, eight warps of 32 rows by 32 outputs, the depth advancing 64 at a time, a
     * pair of sub-blocks, each fifth bit put above its nibble. Row {@code i} of a half reads FP16
     * input row {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i}
     * otherwise, copied to local memory 4 bytes at a time in the k order the weights take; the next
     * step's copy and weight loads are in flight while this step multiplies. The weights never
     * reach local memory: each lane loads the code and fifth-bit words of its own column, turns
     * each code into its exact half in registers and passes them to the MMA with {@code
     * mmaFragmentB}, as {@link #groupedExpertMmaQ4_K} does. Each scale group's sums are scaled in
     * FP32 by its scale, and the group's minimum subtracted through {@code blockSums}, the FP16
     * rows' sums per 32 inputs: the sums of one workgroup per 64-row tile, in the same order. A
     * half with no rows of its tile returns before its first barrier. Example: LFM2.5-24B-A2B's
     * down, 2048 outputs over {@code n = 1536}: eight workgroups per half and 24 steps. Requires
     * {@code n % 256 == 0} and {@code d % 256 == 0}. Grid: {@link #splitWorkgroups} workgroups of
     * {@link #THREADS} threads.
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
        int warp = tid >> 5;
        int colBlocks = d >> 8;
        int grp = ctx.groupIdx;
        int tileHalf = grp / colBlocks;
        int tile = tileHalf >> 1;
        if (tile >= tiles.get(0)) {
            return;
        }
        int rowStart = (tileHalf & 1) * SPLIT_ROWS;
        int tileRows = tiles.get(3 + 3 * tile);
        if (rowStart >= tileRows) {
            return;
        }
        int rows = Math.min(SPLIT_ROWS, tileRows - rowStart);
        int blockCol = (grp - tileHalf * colBlocks) << 8;
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile) + rowStart;

        int[] aTile = ctx.allocateIntLocalArray(2 * SPLIT_INPUT_INTS);
        float[] colScale = ctx.allocateFloatLocalArray(2 * SPLIT_OUTPUTS);
        float[] colMin = ctx.allocateFloatLocalArray(2 * SPLIT_OUTPUTS);
        float[] s00 = ctx.mmaFragment(0.0f);
        float[] s01 = ctx.mmaFragment(0.0f);
        float[] s02 = ctx.mmaFragment(0.0f);
        float[] s03 = ctx.mmaFragment(0.0f);
        float[] s10 = ctx.mmaFragment(0.0f);
        float[] s11 = ctx.mmaFragment(0.0f);
        float[] s12 = ctx.mmaFragment(0.0f);
        float[] s13 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
        float t13_0 = 0.0f;
        float t13_1 = 0.0f;
        float t13_2 = 0.0f;
        float t13_3 = 0.0f;

        // A: one 16-byte copy per thread and step, row tid / 8, eight halves of a 16-deep chunk.
        int aRow = tid >> 3;
        int aChunk = (tid & 7) >> 1;
        int aHalf = tid & 1;
        int aSrc =
                sourceRow(sortedToken, gather, first, Math.min(aRow, rows - 1)) * n
                        + aChunk * 16
                        + aHalf * 2;
        int aDst = aChunk * 256 + aRow * 8 + aHalf * 4;
        // B: four column pairs per thread, word i of each column's 32 code bytes for the step.
        int bCol = warp * 32 + (lane >> 2);
        int bByte = 8 * (lane & 3);
        // Lane t of a column's four holds code words 2t and 2t + 1 of its 8-byte load and needs
        // words t and 4 + t: two shuffles, each source offering its low or high word by which half
        // of the four it is in.
        int bLow = (lane & 3) < 2 ? 1 : 0;
        int bOdd = lane & 1;
        int bSrcA = (lane & ~3) + ((lane & 3) >> 1) + 2 * bOdd;
        int bSrcB = (lane & ~3) + ((lane & 3) >> 1) + 2 * (1 - bOdd);
        int blocksPerRow = n >> 8;
        int rowStride = blocksPerRow * 176;
        int wRow = expert * d + blockCol;
        // Scales: one column per thread, both sub-blocks of the step.
        int sRow = (wRow + tid) * blocksPerRow;

        int r0 = lane >> 2;
        int r1 = r0 + 16;
        int lc0 = warp * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int sumsPerRow = n >> 5;
        int xr0 = sourceRow(sortedToken, gather, first, Math.min(r0, rows - 1)) * sumsPerRow;
        int xr1 = sourceRow(sortedToken, gather, first, Math.min(r0 + 8, rows - 1)) * sumsPerRow;
        int xr2 = sourceRow(sortedToken, gather, first, Math.min(r1, rows - 1)) * sumsPerRow;
        int xr3 = sourceRow(sortedToken, gather, first, Math.min(r1 + 8, rows - 1)) * sumsPerRow;

        int q0a = 0;
        int q0b = 0;
        int h0a = 0;
        int h0b = 0;
        int q1a = 0;
        int q1b = 0;
        int h1a = 0;
        int h1b = 0;
        int q2a = 0;
        int q2b = 0;
        int h2a = 0;
        int h2b = 0;
        int q3a = 0;
        int q3b = 0;
        int h3a = 0;
        int h3b = 0;
        int rHead = 0;
        int rS0 = 0;
        int rS1 = 0;
        int rS2 = 0;
        int rPair = 0;
        ctx.asyncCopyToLocal(aTile, aDst, input16, aSrc);
        ctx.asyncCopyToLocal(aTile, aDst + 1, input16, aSrc + 4);
        ctx.asyncCopyToLocal(aTile, aDst + 2, input16, aSrc + 8);
        ctx.asyncCopyToLocal(aTile, aDst + 3, input16, aSrc + 12);
        ctx.asyncCopyCommit();
        {
            int k = 0;
            int pairIndex = (k & 255) >> 6;
            int kBlock = k >> 8;
            rPair = pairIndex;
            int qOff = kBlock * 176 + 48 + pairIndex * 32 + bByte;
            int hOff = kBlock * 176 + 16 + bByte;
            int c0 = (wRow + bCol + 0) * rowStride;
            long q0aw = w.getLong(c0 + qOff);
            q0a = (int) q0aw;
            q0b = (int) (q0aw >>> 32);
            long h0aw = w.getLong(c0 + hOff);
            h0a = (int) h0aw;
            h0b = (int) (h0aw >>> 32);
            int c1 = (wRow + bCol + 8) * rowStride;
            long q1aw = w.getLong(c1 + qOff);
            q1a = (int) q1aw;
            q1b = (int) (q1aw >>> 32);
            long h1aw = w.getLong(c1 + hOff);
            h1a = (int) h1aw;
            h1b = (int) (h1aw >>> 32);
            int c2 = (wRow + bCol + 16) * rowStride;
            long q2aw = w.getLong(c2 + qOff);
            q2a = (int) q2aw;
            q2b = (int) (q2aw >>> 32);
            long h2aw = w.getLong(c2 + hOff);
            h2a = (int) h2aw;
            h2b = (int) (h2aw >>> 32);
            int c3 = (wRow + bCol + 24) * rowStride;
            long q3aw = w.getLong(c3 + qOff);
            q3a = (int) q3aw;
            q3b = (int) (q3aw >>> 32);
            long h3aw = w.getLong(c3 + hOff);
            h3a = (int) h3aw;
            h3b = (int) (h3aw >>> 32);
            int sBase = (sRow + kBlock) * 176;
            long rHeadw = w.getLong(sBase);
            rHead = (int) rHeadw;
            rS0 = (int) (rHeadw >>> 32);
            long rS1w = w.getLong(sBase + 8);
            rS1 = (int) rS1w;
            rS2 = (int) (rS1w >>> 32);
        }

        int kSteps = n >> 6;
        for (int step = 0; step < kSteps; step++) {
            int kBase = step << 6;
            int buf = step & 1;
            int hShift = 2 * rPair;
            // This step's code and fifth-bit words; reloaded for the next step below.
            int uq0x = ctx.simdShuffle(bLow == 1 ? q0a : q0b, bSrcA);
            int uq0y = ctx.simdShuffle(bLow == 1 ? q0b : q0a, bSrcB);
            int uq0a = bOdd == 0 ? uq0x : uq0y;
            int uq0b = bOdd == 0 ? uq0y : uq0x;
            int uh0x = ctx.simdShuffle(bLow == 1 ? h0a : h0b, bSrcA);
            int uh0y = ctx.simdShuffle(bLow == 1 ? h0b : h0a, bSrcB);
            int uh0a = bOdd == 0 ? uh0x : uh0y;
            int uh0b = bOdd == 0 ? uh0y : uh0x;
            int uq1x = ctx.simdShuffle(bLow == 1 ? q1a : q1b, bSrcA);
            int uq1y = ctx.simdShuffle(bLow == 1 ? q1b : q1a, bSrcB);
            int uq1a = bOdd == 0 ? uq1x : uq1y;
            int uq1b = bOdd == 0 ? uq1y : uq1x;
            int uh1x = ctx.simdShuffle(bLow == 1 ? h1a : h1b, bSrcA);
            int uh1y = ctx.simdShuffle(bLow == 1 ? h1b : h1a, bSrcB);
            int uh1a = bOdd == 0 ? uh1x : uh1y;
            int uh1b = bOdd == 0 ? uh1y : uh1x;
            int uq2x = ctx.simdShuffle(bLow == 1 ? q2a : q2b, bSrcA);
            int uq2y = ctx.simdShuffle(bLow == 1 ? q2b : q2a, bSrcB);
            int uq2a = bOdd == 0 ? uq2x : uq2y;
            int uq2b = bOdd == 0 ? uq2y : uq2x;
            int uh2x = ctx.simdShuffle(bLow == 1 ? h2a : h2b, bSrcA);
            int uh2y = ctx.simdShuffle(bLow == 1 ? h2b : h2a, bSrcB);
            int uh2a = bOdd == 0 ? uh2x : uh2y;
            int uh2b = bOdd == 0 ? uh2y : uh2x;
            int uq3x = ctx.simdShuffle(bLow == 1 ? q3a : q3b, bSrcA);
            int uq3y = ctx.simdShuffle(bLow == 1 ? q3b : q3a, bSrcB);
            int uq3a = bOdd == 0 ? uq3x : uq3y;
            int uq3b = bOdd == 0 ? uq3y : uq3x;
            int uh3x = ctx.simdShuffle(bLow == 1 ? h3a : h3b, bSrcA);
            int uh3y = ctx.simdShuffle(bLow == 1 ? h3b : h3a, bSrcB);
            int uh3a = bOdd == 0 ? uh3x : uh3y;
            int uh3b = bOdd == 0 ? uh3y : uh3x;
            {
                // The step's scale words, loaded raw a step ahead, decoded after the multiplies
                // they overlapped.
                float dd = new HalfFloat((short) rHead).getFloat32();
                float dm = new HalfFloat((short) (rHead >>> 16)).getFloat32();
                int sm0 = scaleAndMinOf(rS0, rS1, rS2, 2 * rPair);
                int sm1 = scaleAndMinOf(rS0, rS1, rS2, 2 * rPair + 1);
                colScale[tid] = dd * (sm0 >> 8);
                colScale[SPLIT_OUTPUTS + tid] = dd * (sm1 >> 8);
                colMin[tid] = dm * (sm0 & 0xFF);
                colMin[SPLIT_OUTPUTS + tid] = dm * (sm1 & 0xFF);
            }
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();
            if (step + 1 < kSteps) {
                int nextDst = (buf ^ 1) * SPLIT_INPUT_INTS + aDst;
                int nextSrc = aSrc + kBase + SPLIT_DEPTH;
                ctx.asyncCopyToLocal(aTile, nextDst, input16, nextSrc);
                ctx.asyncCopyToLocal(aTile, nextDst + 1, input16, nextSrc + 4);
                ctx.asyncCopyToLocal(aTile, nextDst + 2, input16, nextSrc + 8);
                ctx.asyncCopyToLocal(aTile, nextDst + 3, input16, nextSrc + 12);
                ctx.asyncCopyCommit();
                {
                    int k = kBase + SPLIT_DEPTH;
                    int pairIndex = (k & 255) >> 6;
                    int kBlock = k >> 8;
                    rPair = pairIndex;
                    int qOff = kBlock * 176 + 48 + pairIndex * 32 + bByte;
                    int hOff = kBlock * 176 + 16 + bByte;
                    int c0 = (wRow + bCol + 0) * rowStride;
                    long q0aw = w.getLong(c0 + qOff);
                    q0a = (int) q0aw;
                    q0b = (int) (q0aw >>> 32);
                    long h0aw = w.getLong(c0 + hOff);
                    h0a = (int) h0aw;
                    h0b = (int) (h0aw >>> 32);
                    int c1 = (wRow + bCol + 8) * rowStride;
                    long q1aw = w.getLong(c1 + qOff);
                    q1a = (int) q1aw;
                    q1b = (int) (q1aw >>> 32);
                    long h1aw = w.getLong(c1 + hOff);
                    h1a = (int) h1aw;
                    h1b = (int) (h1aw >>> 32);
                    int c2 = (wRow + bCol + 16) * rowStride;
                    long q2aw = w.getLong(c2 + qOff);
                    q2a = (int) q2aw;
                    q2b = (int) (q2aw >>> 32);
                    long h2aw = w.getLong(c2 + hOff);
                    h2a = (int) h2aw;
                    h2b = (int) (h2aw >>> 32);
                    int c3 = (wRow + bCol + 24) * rowStride;
                    long q3aw = w.getLong(c3 + qOff);
                    q3a = (int) q3aw;
                    q3b = (int) (q3aw >>> 32);
                    long h3aw = w.getLong(c3 + hOff);
                    h3a = (int) h3aw;
                    h3b = (int) (h3aw >>> 32);
                    int sBase = (sRow + kBlock) * 176;
                    long rHeadw = w.getLong(sBase);
                    rHead = (int) rHeadw;
                    rS0 = (int) (rHeadw >>> 32);
                    long rS1w = w.getLong(sBase + 8);
                    rS1 = (int) rS1w;
                    rS2 = (int) (rS1w >>> 32);
                }
            }
            int aByte = buf * SPLIT_INPUT_INTS * 4;
            for (int c = 0; c < 4; c++) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024 + 512);
                // Chunk c: words a (c even) or b (c odd), low nibbles and fifth bit 2p (c < 2) or
                // high nibbles and bit 2p + 1; each code becomes its exact half in registers.
                int sh = (c >> 1) << 2;
                int hs = hShift + (c >> 1);
                int v0 =
                        ((((c & 1) == 0 ? uq0a : uq0b) >>> sh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? uh0a : uh0b) >>> hs) << 4) & 0x10101010);
                int v1 =
                        ((((c & 1) == 0 ? uq1a : uq1b) >>> sh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? uh1a : uh1b) >>> hs) << 4) & 0x10101010);
                int v2 =
                        ((((c & 1) == 0 ? uq2a : uq2b) >>> sh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? uh2a : uh2b) >>> hs) << 4) & 0x10101010);
                int v3 =
                        ((((c & 1) == 0 ? uq3a : uq3b) >>> sh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? uh3a : uh3b) >>> hs) << 4) & 0x10101010);
                HalfFloat[] b0 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v0 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v0 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v0 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v0 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b1 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v1 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v1 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v1 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v1 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b2 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v2 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v2 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v2 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v2 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b3 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) (v3 & 0xFF)).getHalfFloatValue() & 0xFFFF)
                                        | ((new HalfFloat((float) ((v3 >>> 8) & 0xFF))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v3 >>> 16) & 0xFF)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v3 >>> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                s00 = ctx.mma(a0, b0, s00, MMAShape.M16N8K16);
                s01 = ctx.mma(a0, b1, s01, MMAShape.M16N8K16);
                s02 = ctx.mma(a0, b2, s02, MMAShape.M16N8K16);
                s03 = ctx.mma(a0, b3, s03, MMAShape.M16N8K16);
                s10 = ctx.mma(a1, b0, s10, MMAShape.M16N8K16);
                s11 = ctx.mma(a1, b1, s11, MMAShape.M16N8K16);
                s12 = ctx.mma(a1, b2, s12, MMAShape.M16N8K16);
                s13 = ctx.mma(a1, b3, s13, MMAShape.M16N8K16);
                if ((c & 1) == 1) {
                    int g = (c >> 1) * SPLIT_OUTPUTS;
                    int sb = (kBase >> 5) + (c >> 1);
                    float x0 = blockSums.get(xr0 + sb);
                    float x1 = blockSums.get(xr1 + sb);
                    float x2 = blockSums.get(xr2 + sb);
                    float x3 = blockSums.get(xr3 + sb);
                    t00_0 += colScale[g + lc0] * s00[0] - colMin[g + lc0] * x0;
                    t00_1 += colScale[g + lc0 + 1] * s00[1] - colMin[g + lc0 + 1] * x0;
                    t00_2 += colScale[g + lc0] * s00[2] - colMin[g + lc0] * x1;
                    t00_3 += colScale[g + lc0 + 1] * s00[3] - colMin[g + lc0 + 1] * x1;
                    t01_0 += colScale[g + lc1] * s01[0] - colMin[g + lc1] * x0;
                    t01_1 += colScale[g + lc1 + 1] * s01[1] - colMin[g + lc1 + 1] * x0;
                    t01_2 += colScale[g + lc1] * s01[2] - colMin[g + lc1] * x1;
                    t01_3 += colScale[g + lc1 + 1] * s01[3] - colMin[g + lc1 + 1] * x1;
                    t02_0 += colScale[g + lc2] * s02[0] - colMin[g + lc2] * x0;
                    t02_1 += colScale[g + lc2 + 1] * s02[1] - colMin[g + lc2 + 1] * x0;
                    t02_2 += colScale[g + lc2] * s02[2] - colMin[g + lc2] * x1;
                    t02_3 += colScale[g + lc2 + 1] * s02[3] - colMin[g + lc2 + 1] * x1;
                    t03_0 += colScale[g + lc3] * s03[0] - colMin[g + lc3] * x0;
                    t03_1 += colScale[g + lc3 + 1] * s03[1] - colMin[g + lc3 + 1] * x0;
                    t03_2 += colScale[g + lc3] * s03[2] - colMin[g + lc3] * x1;
                    t03_3 += colScale[g + lc3 + 1] * s03[3] - colMin[g + lc3 + 1] * x1;
                    t10_0 += colScale[g + lc0] * s10[0] - colMin[g + lc0] * x2;
                    t10_1 += colScale[g + lc0 + 1] * s10[1] - colMin[g + lc0 + 1] * x2;
                    t10_2 += colScale[g + lc0] * s10[2] - colMin[g + lc0] * x3;
                    t10_3 += colScale[g + lc0 + 1] * s10[3] - colMin[g + lc0 + 1] * x3;
                    t11_0 += colScale[g + lc1] * s11[0] - colMin[g + lc1] * x2;
                    t11_1 += colScale[g + lc1 + 1] * s11[1] - colMin[g + lc1 + 1] * x2;
                    t11_2 += colScale[g + lc1] * s11[2] - colMin[g + lc1] * x3;
                    t11_3 += colScale[g + lc1 + 1] * s11[3] - colMin[g + lc1 + 1] * x3;
                    t12_0 += colScale[g + lc2] * s12[0] - colMin[g + lc2] * x2;
                    t12_1 += colScale[g + lc2 + 1] * s12[1] - colMin[g + lc2 + 1] * x2;
                    t12_2 += colScale[g + lc2] * s12[2] - colMin[g + lc2] * x3;
                    t12_3 += colScale[g + lc2 + 1] * s12[3] - colMin[g + lc2 + 1] * x3;
                    t13_0 += colScale[g + lc3] * s13[0] - colMin[g + lc3] * x2;
                    t13_1 += colScale[g + lc3 + 1] * s13[1] - colMin[g + lc3 + 1] * x2;
                    t13_2 += colScale[g + lc3] * s13[2] - colMin[g + lc3] * x3;
                    t13_3 += colScale[g + lc3 + 1] * s13[3] - colMin[g + lc3 + 1] * x3;
                    s00 = ctx.mmaFragment(0.0f);
                    s01 = ctx.mmaFragment(0.0f);
                    s02 = ctx.mmaFragment(0.0f);
                    s03 = ctx.mmaFragment(0.0f);
                    s10 = ctx.mmaFragment(0.0f);
                    s11 = ctx.mmaFragment(0.0f);
                    s12 = ctx.mmaFragment(0.0f);
                    s13 = ctx.mmaFragment(0.0f);
                }
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
     * scaled in FP32 by its scale. Each step's loads are issued before the previous step's
     * multiplies, so their latency overlaps them. Only the tile's rows are written, to output row
     * {@code first + i}. Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups
     * of {@link #THREADS} threads; a workgroup past the tile count returns before its first
     * barrier.
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
        float[] s01 = ctx.mmaFragment(0.0f);
        float[] s02 = ctx.mmaFragment(0.0f);
        float[] s03 = ctx.mmaFragment(0.0f);
        float[] s10 = ctx.mmaFragment(0.0f);
        float[] s11 = ctx.mmaFragment(0.0f);
        float[] s12 = ctx.mmaFragment(0.0f);
        float[] s13 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
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

        int ra0 = 0;
        int ra1 = 0;
        float rs0 = 0.0f;
        float rs1 = 0.0f;
        int rGroup = 0;
        int rl0_0 = 0;
        int rl1_0 = 0;
        int rh0_0 = 0;
        int rh1_0 = 0;
        int rl0_1 = 0;
        int rl1_1 = 0;
        int rh0_1 = 0;
        int rh1_1 = 0;
        int rl0_2 = 0;
        int rl1_2 = 0;
        int rh0_2 = 0;
        int rh1_2 = 0;
        int rl0_3 = 0;
        int rl1_3 = 0;
        int rh0_3 = 0;
        int rh1_3 = 0;
        ctx.localBarrier();
        ra0 = packHalves(input16, aBase0 + 0);
        ra1 = packHalves(input16, aBase1 + 0);
        {
            int k = 0 + kQuarter;
            int within = k & 255;
            int halfIndex = within >> 7;
            rGroup = (within >> 5) & 3;
            int posInGroup = within & 31;
            int base0 = (wRow * blocksPerRow + (k >> 8)) * 210;
            int base1 = base0 + blocksPerRow * 210;
            int ql0 = base0 + halfIndex * 64 + (rGroup & 1) * 32 + posInGroup;
            int ql1 = base1 + halfIndex * 64 + (rGroup & 1) * 32 + posInGroup;
            int qh0 = base0 + 128 + halfIndex * 32 + posInGroup;
            int qh1 = base1 + 128 + halfIndex * 32 + posInGroup;
            rl0_0 = w.get(ql0 + 0) & 0xFF;
            rl1_0 = w.get(ql1 + 0) & 0xFF;
            rh0_0 = w.get(qh0 + 0) & 0xFF;
            rh1_0 = w.get(qh1 + 0) & 0xFF;
            rl0_1 = w.get(ql0 + 1) & 0xFF;
            rl1_1 = w.get(ql1 + 1) & 0xFF;
            rh0_1 = w.get(qh0 + 1) & 0xFF;
            rh1_1 = w.get(qh1 + 1) & 0xFF;
            rl0_2 = w.get(ql0 + 2) & 0xFF;
            rl1_2 = w.get(ql1 + 2) & 0xFF;
            rh0_2 = w.get(qh0 + 2) & 0xFF;
            rh1_2 = w.get(qh1 + 2) & 0xFF;
            rl0_3 = w.get(ql0 + 3) & 0xFF;
            rl1_3 = w.get(ql1 + 3) & 0xFF;
            rh0_3 = w.get(qh0 + 3) & 0xFF;
            rh1_3 = w.get(qh1 + 3) & 0xFF;
            if (kQuarter == 0) {
                int sc = halfIndex * 8 + (posInGroup >> 4) + rGroup * 2;
                rs0 = w.getHalfFloat(base0 + 208).getFloat32() * w.get(base0 + 192 + sc);
                rs1 = w.getHalfFloat(base1 + 208).getFloat32() * w.get(base1 + 192 + sc);
            }
        }
        aTile[aIdx0] = ra0;
        aTile[aIdx1] = ra1;
        {
            int v0_0 = rl0_0 & 15;
            int v1_0 = rl1_0 & 15;
            int h0_0 = rh0_0 & 3;
            int h1_0 = rh1_0 & 3;
            int v0_1 = rl0_1 & 15;
            int v1_1 = rl1_1 & 15;
            int h0_1 = rh0_1 & 3;
            int h1_1 = rh1_1 & 3;
            int v0_2 = rl0_2 & 15;
            int v1_2 = rl1_2 & 15;
            int h0_2 = rh0_2 & 3;
            int h1_2 = rh1_2 & 3;
            int v0_3 = rl0_3 & 15;
            int v1_3 = rl1_3 & 15;
            int h0_3 = rh0_3 & 3;
            int h1_3 = rh1_3 & 3;
            if ((rGroup >> 1) == 1) {
                v0_0 = (rl0_0 >> 4) & 15;
                v1_0 = (rl1_0 >> 4) & 15;
                v0_1 = (rl0_1 >> 4) & 15;
                v1_1 = (rl1_1 >> 4) & 15;
                v0_2 = (rl0_2 >> 4) & 15;
                v1_2 = (rl1_2 >> 4) & 15;
                v0_3 = (rl0_3 >> 4) & 15;
                v1_3 = (rl1_3 >> 4) & 15;
            }
            if (rGroup == 1) {
                h0_0 = (rh0_0 >> 2) & 3;
                h1_0 = (rh1_0 >> 2) & 3;
                h0_1 = (rh0_1 >> 2) & 3;
                h1_1 = (rh1_1 >> 2) & 3;
                h0_2 = (rh0_2 >> 2) & 3;
                h1_2 = (rh1_2 >> 2) & 3;
                h0_3 = (rh0_3 >> 2) & 3;
                h1_3 = (rh1_3 >> 2) & 3;
            } else if (rGroup == 2) {
                h0_0 = (rh0_0 >> 4) & 3;
                h1_0 = (rh1_0 >> 4) & 3;
                h0_1 = (rh0_1 >> 4) & 3;
                h1_1 = (rh1_1 >> 4) & 3;
                h0_2 = (rh0_2 >> 4) & 3;
                h1_2 = (rh1_2 >> 4) & 3;
                h0_3 = (rh0_3 >> 4) & 3;
                h1_3 = (rh1_3 >> 4) & 3;
            } else if (rGroup == 3) {
                h0_0 = (rh0_0 >> 6) & 3;
                h1_0 = (rh1_0 >> 6) & 3;
                h0_1 = (rh0_1 >> 6) & 3;
                h1_1 = (rh1_1 >> 6) & 3;
                h0_2 = (rh0_2 >> 6) & 3;
                h1_2 = (rh1_2 >> 6) & 3;
                h0_3 = (rh0_3 >> 6) & 3;
                h1_3 = (rh1_3 >> 6) & 3;
            }
            bTile[bSlot + 0] =
                    codes[96 + (v0_0 | (h0_0 << 4))] | (codes[96 + (v1_0 | (h1_0 << 4))] << 16);
            bTile[bSlot + 4] =
                    codes[96 + (v0_1 | (h0_1 << 4))] | (codes[96 + (v1_1 | (h1_1 << 4))] << 16);
            bTile[bSlot + 8] =
                    codes[96 + (v0_2 | (h0_2 << 4))] | (codes[96 + (v1_2 | (h1_2 << 4))] << 16);
            bTile[bSlot + 12] =
                    codes[96 + (v0_3 | (h0_3 << 4))] | (codes[96 + (v1_3 | (h1_3 << 4))] << 16);
        }
        if (kQuarter == 0) {
            colScale[pairCol] = rs0;
            colScale[pairCol + 1] = rs1;
        }
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            boolean next = kStep + 1 < numKSteps;
            if (next) {
                ra0 = packHalves(input16, aBase0 + kBase + BK);
                ra1 = packHalves(input16, aBase1 + kBase + BK);
                {
                    int k = kBase + BK + kQuarter;
                    int within = k & 255;
                    int halfIndex = within >> 7;
                    rGroup = (within >> 5) & 3;
                    int posInGroup = within & 31;
                    int base0 = (wRow * blocksPerRow + (k >> 8)) * 210;
                    int base1 = base0 + blocksPerRow * 210;
                    int ql0 = base0 + halfIndex * 64 + (rGroup & 1) * 32 + posInGroup;
                    int ql1 = base1 + halfIndex * 64 + (rGroup & 1) * 32 + posInGroup;
                    int qh0 = base0 + 128 + halfIndex * 32 + posInGroup;
                    int qh1 = base1 + 128 + halfIndex * 32 + posInGroup;
                    rl0_0 = w.get(ql0 + 0) & 0xFF;
                    rl1_0 = w.get(ql1 + 0) & 0xFF;
                    rh0_0 = w.get(qh0 + 0) & 0xFF;
                    rh1_0 = w.get(qh1 + 0) & 0xFF;
                    rl0_1 = w.get(ql0 + 1) & 0xFF;
                    rl1_1 = w.get(ql1 + 1) & 0xFF;
                    rh0_1 = w.get(qh0 + 1) & 0xFF;
                    rh1_1 = w.get(qh1 + 1) & 0xFF;
                    rl0_2 = w.get(ql0 + 2) & 0xFF;
                    rl1_2 = w.get(ql1 + 2) & 0xFF;
                    rh0_2 = w.get(qh0 + 2) & 0xFF;
                    rh1_2 = w.get(qh1 + 2) & 0xFF;
                    rl0_3 = w.get(ql0 + 3) & 0xFF;
                    rl1_3 = w.get(ql1 + 3) & 0xFF;
                    rh0_3 = w.get(qh0 + 3) & 0xFF;
                    rh1_3 = w.get(qh1 + 3) & 0xFF;
                    if (kQuarter == 0) {
                        int sc = halfIndex * 8 + (posInGroup >> 4) + rGroup * 2;
                        rs0 = w.getHalfFloat(base0 + 208).getFloat32() * w.get(base0 + 192 + sc);
                        rs1 = w.getHalfFloat(base1 + 208).getFloat32() * w.get(base1 + 192 + sc);
                    }
                }
            }
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
            if (next) {
                aTile[aIdx0] = ra0;
                aTile[aIdx1] = ra1;
                {
                    int v0_0 = rl0_0 & 15;
                    int v1_0 = rl1_0 & 15;
                    int h0_0 = rh0_0 & 3;
                    int h1_0 = rh1_0 & 3;
                    int v0_1 = rl0_1 & 15;
                    int v1_1 = rl1_1 & 15;
                    int h0_1 = rh0_1 & 3;
                    int h1_1 = rh1_1 & 3;
                    int v0_2 = rl0_2 & 15;
                    int v1_2 = rl1_2 & 15;
                    int h0_2 = rh0_2 & 3;
                    int h1_2 = rh1_2 & 3;
                    int v0_3 = rl0_3 & 15;
                    int v1_3 = rl1_3 & 15;
                    int h0_3 = rh0_3 & 3;
                    int h1_3 = rh1_3 & 3;
                    if ((rGroup >> 1) == 1) {
                        v0_0 = (rl0_0 >> 4) & 15;
                        v1_0 = (rl1_0 >> 4) & 15;
                        v0_1 = (rl0_1 >> 4) & 15;
                        v1_1 = (rl1_1 >> 4) & 15;
                        v0_2 = (rl0_2 >> 4) & 15;
                        v1_2 = (rl1_2 >> 4) & 15;
                        v0_3 = (rl0_3 >> 4) & 15;
                        v1_3 = (rl1_3 >> 4) & 15;
                    }
                    if (rGroup == 1) {
                        h0_0 = (rh0_0 >> 2) & 3;
                        h1_0 = (rh1_0 >> 2) & 3;
                        h0_1 = (rh0_1 >> 2) & 3;
                        h1_1 = (rh1_1 >> 2) & 3;
                        h0_2 = (rh0_2 >> 2) & 3;
                        h1_2 = (rh1_2 >> 2) & 3;
                        h0_3 = (rh0_3 >> 2) & 3;
                        h1_3 = (rh1_3 >> 2) & 3;
                    } else if (rGroup == 2) {
                        h0_0 = (rh0_0 >> 4) & 3;
                        h1_0 = (rh1_0 >> 4) & 3;
                        h0_1 = (rh0_1 >> 4) & 3;
                        h1_1 = (rh1_1 >> 4) & 3;
                        h0_2 = (rh0_2 >> 4) & 3;
                        h1_2 = (rh1_2 >> 4) & 3;
                        h0_3 = (rh0_3 >> 4) & 3;
                        h1_3 = (rh1_3 >> 4) & 3;
                    } else if (rGroup == 3) {
                        h0_0 = (rh0_0 >> 6) & 3;
                        h1_0 = (rh1_0 >> 6) & 3;
                        h0_1 = (rh0_1 >> 6) & 3;
                        h1_1 = (rh1_1 >> 6) & 3;
                        h0_2 = (rh0_2 >> 6) & 3;
                        h1_2 = (rh1_2 >> 6) & 3;
                        h0_3 = (rh0_3 >> 6) & 3;
                        h1_3 = (rh1_3 >> 6) & 3;
                    }
                    bTile[bSlot + 0] =
                            codes[96 + (v0_0 | (h0_0 << 4))]
                                    | (codes[96 + (v1_0 | (h1_0 << 4))] << 16);
                    bTile[bSlot + 4] =
                            codes[96 + (v0_1 | (h0_1 << 4))]
                                    | (codes[96 + (v1_1 | (h1_1 << 4))] << 16);
                    bTile[bSlot + 8] =
                            codes[96 + (v0_2 | (h0_2 << 4))]
                                    | (codes[96 + (v1_2 | (h1_2 << 4))] << 16);
                    bTile[bSlot + 12] =
                            codes[96 + (v0_3 | (h0_3 << 4))]
                                    | (codes[96 + (v1_3 | (h1_3 << 4))] << 16);
                }
                if (kQuarter == 0) {
                    colScale[pairCol] = rs0;
                    colScale[pairCol + 1] = rs1;
                }
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
     * scaled in FP32 by its scale. Each step's loads are issued before the previous step's
     * multiplies, so their latency overlaps them. Only the tile's rows are written, to output row
     * {@code first + i}. Requires {@code d % 128 == 0}. Grid: {@code maxTiles * d / 128} workgroups
     * of {@link #THREADS} threads; a workgroup past the tile count returns before its first
     * barrier.
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
        float[] s01 = ctx.mmaFragment(0.0f);
        float[] s02 = ctx.mmaFragment(0.0f);
        float[] s03 = ctx.mmaFragment(0.0f);
        float[] s10 = ctx.mmaFragment(0.0f);
        float[] s11 = ctx.mmaFragment(0.0f);
        float[] s12 = ctx.mmaFragment(0.0f);
        float[] s13 = ctx.mmaFragment(0.0f);
        float t00_0 = 0.0f;
        float t00_1 = 0.0f;
        float t00_2 = 0.0f;
        float t00_3 = 0.0f;
        float t01_0 = 0.0f;
        float t01_1 = 0.0f;
        float t01_2 = 0.0f;
        float t01_3 = 0.0f;
        float t02_0 = 0.0f;
        float t02_1 = 0.0f;
        float t02_2 = 0.0f;
        float t02_3 = 0.0f;
        float t03_0 = 0.0f;
        float t03_1 = 0.0f;
        float t03_2 = 0.0f;
        float t03_3 = 0.0f;
        float t10_0 = 0.0f;
        float t10_1 = 0.0f;
        float t10_2 = 0.0f;
        float t10_3 = 0.0f;
        float t11_0 = 0.0f;
        float t11_1 = 0.0f;
        float t11_2 = 0.0f;
        float t11_3 = 0.0f;
        float t12_0 = 0.0f;
        float t12_1 = 0.0f;
        float t12_2 = 0.0f;
        float t12_3 = 0.0f;
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

        int ra0 = 0;
        int ra1 = 0;
        float rs0 = 0.0f;
        float rs1 = 0.0f;
        int rb0_0 = 0;
        int rb1_0 = 0;
        int rb0_1 = 0;
        int rb1_1 = 0;
        int rb0_2 = 0;
        int rb1_2 = 0;
        int rb0_3 = 0;
        int rb1_3 = 0;
        ctx.localBarrier();
        ra0 = packHalves(input16, aBase0 + 0);
        ra1 = packHalves(input16, aBase1 + 0);
        {
            int k = 0 + kQuarter;
            int base0 = (wRow * blocksPerRow + (k >> 5)) * 34;
            int base1 = base0 + blocksPerRow * 34;
            int q0 = base0 + 2 + (k & 31);
            int q1 = base1 + 2 + (k & 31);
            rb0_0 = w.get(q0 + 0);
            rb1_0 = w.get(q1 + 0);
            rb0_1 = w.get(q0 + 1);
            rb1_1 = w.get(q1 + 1);
            rb0_2 = w.get(q0 + 2);
            rb1_2 = w.get(q1 + 2);
            rb0_3 = w.get(q0 + 3);
            rb1_3 = w.get(q1 + 3);
            if (kQuarter == 0) {
                rs0 = w.getHalfFloat(base0).getFloat32();
                rs1 = w.getHalfFloat(base1).getFloat32();
            }
        }
        aTile[aIdx0] = ra0;
        aTile[aIdx1] = ra1;
        bTile[bSlot + 0] = codes[128 + rb0_0] | (codes[128 + rb1_0] << 16);
        bTile[bSlot + 4] = codes[128 + rb0_1] | (codes[128 + rb1_1] << 16);
        bTile[bSlot + 8] = codes[128 + rb0_2] | (codes[128 + rb1_2] << 16);
        bTile[bSlot + 12] = codes[128 + rb0_3] | (codes[128 + rb1_3] << 16);
        if (kQuarter == 0) {
            colScale[pairCol] = rs0;
            colScale[pairCol + 1] = rs1;
        }
        ctx.localBarrier();

        int numKSteps = n >> 4;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep << 4;
            boolean next = kStep + 1 < numKSteps;
            if (next) {
                ra0 = packHalves(input16, aBase0 + kBase + BK);
                ra1 = packHalves(input16, aBase1 + kBase + BK);
                {
                    int k = kBase + BK + kQuarter;
                    int base0 = (wRow * blocksPerRow + (k >> 5)) * 34;
                    int base1 = base0 + blocksPerRow * 34;
                    int q0 = base0 + 2 + (k & 31);
                    int q1 = base1 + 2 + (k & 31);
                    rb0_0 = w.get(q0 + 0);
                    rb1_0 = w.get(q1 + 0);
                    rb0_1 = w.get(q0 + 1);
                    rb1_1 = w.get(q1 + 1);
                    rb0_2 = w.get(q0 + 2);
                    rb1_2 = w.get(q1 + 2);
                    rb0_3 = w.get(q0 + 3);
                    rb1_3 = w.get(q1 + 3);
                    if (kQuarter == 0) {
                        rs0 = w.getHalfFloat(base0).getFloat32();
                        rs1 = w.getHalfFloat(base1).getFloat32();
                    }
                }
            }
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
            if (next) {
                aTile[aIdx0] = ra0;
                aTile[aIdx1] = ra1;
                bTile[bSlot + 0] = codes[128 + rb0_0] | (codes[128 + rb1_0] << 16);
                bTile[bSlot + 4] = codes[128 + rb0_1] | (codes[128 + rb1_1] << 16);
                bTile[bSlot + 8] = codes[128 + rb0_2] | (codes[128 + rb1_2] << 16);
                bTile[bSlot + 12] = codes[128 + rb0_3] | (codes[128 + rb1_3] << 16);
                if (kQuarter == 0) {
                    colScale[pairCol] = rs0;
                    colScale[pairCol + 1] = rs1;
                }
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
