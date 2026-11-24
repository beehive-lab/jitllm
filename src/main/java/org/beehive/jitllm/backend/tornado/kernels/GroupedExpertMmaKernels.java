/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
 * workgroup per half of a tile of one expert's assignments (32 rows, the typical expert's share of
 * a 512-row chunk, so no warp multiplies empty rows) and 256 outputs, the expert's weights read in
 * place from the stacked tensor. Eight warps, each 32 rows by 32 outputs as {@code M16N8K16} MMAs
 * over FP16 inputs; the depth advances 64 at a time, the next step's loads in flight while this
 * step multiplies.
 *
 * <p>The weights enter the MMA as their integer codes, which FP16 holds exactly: each of Q4_K,
 * Q5_K, Q6_K and Q8_0 (whose signed codes are sign-extended) builds each warp's B operand in its
 * own registers ({@code mmaFragmentB}), since a warp is the only reader of its columns. Each scale
 * group's partial products accumulate in their own fragments and are then scaled in FP32 by the
 * group's scale; for Q4_K and Q5_K the group's minimum is subtracted through the input rows' sums
 * per 32 inputs ({@link Lfm2BatchKernels#halvesWithBlockSums}). Example: a Q4_K weight {@code d *
 * sc * q - dmin * m} contributes {@code d * sc * (x . q) - dmin * m * sum(x)} per 32 inputs.
 */
// @formatter:on
public final class GroupedExpertMmaKernels {

    private GroupedExpertMmaKernels() {}

    /** Threads of a workgroup: eight warps. */
    public static final int THREADS = 256;

    /** Rows of a workgroup: half of an expert tile. */
    private static final int SPLIT_ROWS = 32;

    /** Outputs of a workgroup: eight warps of 32. */
    private static final int SPLIT_OUTPUTS = 256;

    /** Depth of one step: a pair of 32-weight sub-blocks or Q8_0 blocks. */
    private static final int SPLIT_DEPTH = 64;

    /** Ints of one input buffer of the split kernels: four 16-deep chunks of 32 rows. */
    private static final int SPLIT_INPUT_INTS = 1024;

    /** Workgroups of the kernels over {@code maxTiles} tiles and {@code d} outputs. */
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
     * The grouped expert GEMM over Q6_K experts on tensor cores (CUDA), {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it: one workgroup per half of a tile (32 rows)
     * and 256 outputs, eight warps of 32 rows by 32 outputs, the depth advancing 64 at a time, two
     * 32-weight groups of a super-block. Row {@code i} of a half reads FP16 input row {@code
     * sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise, copied to
     * local memory as {@link #groupedExpertMmaQ4_K} copies it; the next step's copy and weight
     * loads are in flight while this step multiplies. The weights never reach local memory: for
     * each 16-deep chunk, lane {@code L} loads the low-bit and high-bit words at byte {@code 4 (L %
     * 4)} of its own column {@code L / 4} (two 2-byte loads where a 210-byte block starts off a
     * word boundary), once per pair of steps: the second step of a pair takes the same bytes' high
     * nibbles and bits. It turns each code into its exact half less 32 in registers and passes them
     * to the MMA with {@code mmaFragmentB}. Each chunk's sums are scaled in FP32 by the chunk's
     * scale. A half with no rows of its tile returns before its first barrier. Example:
     * LFM2.5-24B-A2B's down in layers 7 and 8, 2048 outputs over {@code n = 1536}: eight workgroups
     * per half and 24 steps. Requires {@code n % 256 == 0} and {@code d % 256 == 0}. Grid: {@link
     * #splitWorkgroups} workgroups of {@link #THREADS} threads.
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
        float[] colScale = ctx.allocateFloatLocalArray(4 * SPLIT_OUTPUTS);
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
        // B: four columns per thread, eight apart; per chunk, the low-bit word and the high-bit
        // word at byte 4 (lane % 4) of the chunk's 16.
        int bCol = warp * 32 + (lane >> 2);
        int bByte = 4 * (lane & 3);
        int blocksPerRow = n >> 8;
        int rowStride = blocksPerRow * 210;
        int wRow = expert * d + blockCol;
        // Scales: one column per thread, the step's four chunks.
        int sRow = (wRow + tid) * blocksPerRow;

        int r0 = lane >> 2;
        int r1 = r0 + 16;
        int lc0 = warp * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int l00 = 0;
        int l01 = 0;
        int l02 = 0;
        int l03 = 0;
        int h00 = 0;
        int h01 = 0;
        int l10 = 0;
        int l11 = 0;
        int l12 = 0;
        int l13 = 0;
        int h10 = 0;
        int h11 = 0;
        int l20 = 0;
        int l21 = 0;
        int l22 = 0;
        int l23 = 0;
        int h20 = 0;
        int h21 = 0;
        int l30 = 0;
        int l31 = 0;
        int l32 = 0;
        int l33 = 0;
        int h30 = 0;
        int h31 = 0;
        int rHead = 0;
        int rS0 = 0;
        int rS1 = 0;
        int rS2 = 0;
        int rS3 = 0;
        ctx.asyncCopyToLocal(aTile, aDst, input16, aSrc);
        ctx.asyncCopyToLocal(aTile, aDst + 1, input16, aSrc + 4);
        ctx.asyncCopyToLocal(aTile, aDst + 2, input16, aSrc + 8);
        ctx.asyncCopyToLocal(aTile, aDst + 3, input16, aSrc + 12);
        ctx.asyncCopyCommit();
        {
            int k = 0;
            int kBlock = k >> 8;
            int half = (k & 255) >> 7;
            int c0 = (wRow + bCol) * rowStride + kBlock * 210 + half * 64 + bByte;
            int e0 = c0 + 128 - half * 32;
            l00 = word(w, c0);
            l01 = word(w, c0 + 16);
            l02 = word(w, c0 + 32);
            l03 = word(w, c0 + 48);
            h00 = word(w, e0);
            h01 = word(w, e0 + 16);
            int c1 = c0 + 8 * rowStride;
            int e1 = e0 + 8 * rowStride;
            l10 = word(w, c1);
            l11 = word(w, c1 + 16);
            l12 = word(w, c1 + 32);
            l13 = word(w, c1 + 48);
            h10 = word(w, e1);
            h11 = word(w, e1 + 16);
            int c2 = c0 + 16 * rowStride;
            int e2 = e0 + 16 * rowStride;
            l20 = word(w, c2);
            l21 = word(w, c2 + 16);
            l22 = word(w, c2 + 32);
            l23 = word(w, c2 + 48);
            h20 = word(w, e2);
            h21 = word(w, e2 + 16);
            int c3 = c0 + 24 * rowStride;
            int e3 = e0 + 24 * rowStride;
            l30 = word(w, c3);
            l31 = word(w, c3 + 16);
            l32 = word(w, c3 + 32);
            l33 = word(w, c3 + 48);
            h30 = word(w, e3);
            h31 = word(w, e3 + 16);
            int sBase = (sRow + kBlock) * 210;
            rHead = w.getHalfFloat(sBase + 208).getHalfFloatValue();
            int scAt = sBase + 192 + half * 8 + ((k & 127) >> 4);
            rS0 = w.get(scAt);
            rS1 = w.get(scAt + 1);
            rS2 = w.get(scAt + 2);
            rS3 = w.get(scAt + 3);
        }

        int kSteps = n >> 6;
        for (int step = 0; step < kSteps; step++) {
            int kBase = step << 6;
            int buf = step & 1;
            // This step's code words; the registers are reloaded for the next step below.
            int u00 = l00;
            int u01 = l01;
            int u02 = l02;
            int u03 = l03;
            int g00 = h00;
            int g01 = h01;
            int u10 = l10;
            int u11 = l11;
            int u12 = l12;
            int u13 = l13;
            int g10 = h10;
            int g11 = h11;
            int u20 = l20;
            int u21 = l21;
            int u22 = l22;
            int u23 = l23;
            int g20 = h20;
            int g21 = h21;
            int u30 = l30;
            int u31 = l31;
            int u32 = l32;
            int u33 = l33;
            int g30 = h30;
            int g31 = h31;
            {
                // The step's scale bytes, loaded raw a step ahead, scaled after the multiplies
                // they overlapped.
                float dd = new HalfFloat((short) rHead).getFloat32();
                colScale[tid] = dd * rS0;
                colScale[SPLIT_OUTPUTS + tid] = dd * rS1;
                colScale[2 * SPLIT_OUTPUTS + tid] = dd * rS2;
                colScale[3 * SPLIT_OUTPUTS + tid] = dd * rS3;
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
                    int kBlock = k >> 8;
                    int half = (k & 255) >> 7;
                    // Only an even step takes new words: the odd step after it multiplies the
                    // same bytes' high nibbles and bits.
                    if ((step & 1) == 1) {
                        int c0 = (wRow + bCol) * rowStride + kBlock * 210 + half * 64 + bByte;
                        int e0 = c0 + 128 - half * 32;
                        l00 = word(w, c0);
                        l01 = word(w, c0 + 16);
                        l02 = word(w, c0 + 32);
                        l03 = word(w, c0 + 48);
                        h00 = word(w, e0);
                        h01 = word(w, e0 + 16);
                        int c1 = c0 + 8 * rowStride;
                        int e1 = e0 + 8 * rowStride;
                        l10 = word(w, c1);
                        l11 = word(w, c1 + 16);
                        l12 = word(w, c1 + 32);
                        l13 = word(w, c1 + 48);
                        h10 = word(w, e1);
                        h11 = word(w, e1 + 16);
                        int c2 = c0 + 16 * rowStride;
                        int e2 = e0 + 16 * rowStride;
                        l20 = word(w, c2);
                        l21 = word(w, c2 + 16);
                        l22 = word(w, c2 + 32);
                        l23 = word(w, c2 + 48);
                        h20 = word(w, e2);
                        h21 = word(w, e2 + 16);
                        int c3 = c0 + 24 * rowStride;
                        int e3 = e0 + 24 * rowStride;
                        l30 = word(w, c3);
                        l31 = word(w, c3 + 16);
                        l32 = word(w, c3 + 32);
                        l33 = word(w, c3 + 48);
                        h30 = word(w, e3);
                        h31 = word(w, e3 + 16);
                    }
                    int sBase = (sRow + kBlock) * 210;
                    rHead = w.getHalfFloat(sBase + 208).getHalfFloatValue();
                    int scAt = sBase + 192 + half * 8 + ((k & 127) >> 4);
                    rS0 = w.get(scAt);
                    rS1 = w.get(scAt + 1);
                    rS2 = w.get(scAt + 2);
                    rS3 = w.get(scAt + 3);
                }
            }
            int aByte = buf * SPLIT_INPUT_INTS * 4;
            // Groups 2 (step % 2) and 2 (step % 2) + 1 of a half: low nibbles, then high nibbles;
            // high bits at shift 4 (step % 2), then two above.
            int lsh = (step & 1) << 2;
            for (int c = 0; c < 4; c++) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024 + 512);
                // Chunk c: low-bit word c, high-bit word c % 2; each code less 32 becomes its
                // exact half in the lane's own registers.
                int hsh = lsh + ((c >> 1) << 1);
                int v0 =
                        (((c == 0 ? u00 : c == 1 ? u01 : c == 2 ? u02 : u03) >>> lsh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? g00 : g01) >>> hsh) & 0x03030303) << 4);
                int v1 =
                        (((c == 0 ? u10 : c == 1 ? u11 : c == 2 ? u12 : u13) >>> lsh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? g10 : g11) >>> hsh) & 0x03030303) << 4);
                int v2 =
                        (((c == 0 ? u20 : c == 1 ? u21 : c == 2 ? u22 : u23) >>> lsh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? g20 : g21) >>> hsh) & 0x03030303) << 4);
                int v3 =
                        (((c == 0 ? u30 : c == 1 ? u31 : c == 2 ? u32 : u33) >>> lsh) & 0x0F0F0F0F)
                                | (((((c & 1) == 0 ? g30 : g31) >>> hsh) & 0x03030303) << 4);
                HalfFloat[] b0 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v0 & 0xFF) - 32)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (((v0 >>> 8) & 0xFF) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) (((v0 >>> 16) & 0xFF) - 32))
                                                        .getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v0 >>> 24) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b1 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v1 & 0xFF) - 32)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (((v1 >>> 8) & 0xFF) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) (((v1 >>> 16) & 0xFF) - 32))
                                                        .getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v1 >>> 24) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b2 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v2 & 0xFF) - 32)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (((v2 >>> 8) & 0xFF) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) (((v2 >>> 16) & 0xFF) - 32))
                                                        .getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v2 >>> 24) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b3 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v3 & 0xFF) - 32)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (((v3 >>> 8) & 0xFF) - 32))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) (((v3 >>> 16) & 0xFF) - 32))
                                                        .getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v3 >>> 24) - 32))
                                                                .getHalfFloatValue()
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
                int g = c * SPLIT_OUTPUTS;
                t00_0 += colScale[g + lc0] * s00[0];
                t00_1 += colScale[g + lc0 + 1] * s00[1];
                t00_2 += colScale[g + lc0] * s00[2];
                t00_3 += colScale[g + lc0 + 1] * s00[3];
                t01_0 += colScale[g + lc1] * s01[0];
                t01_1 += colScale[g + lc1 + 1] * s01[1];
                t01_2 += colScale[g + lc1] * s01[2];
                t01_3 += colScale[g + lc1 + 1] * s01[3];
                t02_0 += colScale[g + lc2] * s02[0];
                t02_1 += colScale[g + lc2 + 1] * s02[1];
                t02_2 += colScale[g + lc2] * s02[2];
                t02_3 += colScale[g + lc2 + 1] * s02[3];
                t03_0 += colScale[g + lc3] * s03[0];
                t03_1 += colScale[g + lc3 + 1] * s03[1];
                t03_2 += colScale[g + lc3] * s03[2];
                t03_3 += colScale[g + lc3 + 1] * s03[3];
                t10_0 += colScale[g + lc0] * s10[0];
                t10_1 += colScale[g + lc0 + 1] * s10[1];
                t10_2 += colScale[g + lc0] * s10[2];
                t10_3 += colScale[g + lc0 + 1] * s10[3];
                t11_0 += colScale[g + lc1] * s11[0];
                t11_1 += colScale[g + lc1 + 1] * s11[1];
                t11_2 += colScale[g + lc1] * s11[2];
                t11_3 += colScale[g + lc1 + 1] * s11[3];
                t12_0 += colScale[g + lc2] * s12[0];
                t12_1 += colScale[g + lc2 + 1] * s12[1];
                t12_2 += colScale[g + lc2] * s12[2];
                t12_3 += colScale[g + lc2 + 1] * s12[3];
                t13_0 += colScale[g + lc3] * s13[0];
                t13_1 += colScale[g + lc3 + 1] * s13[1];
                t13_2 += colScale[g + lc3] * s13[2];
                t13_3 += colScale[g + lc3 + 1] * s13[3];
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

    /**
     * Four bytes at {@code at} as a word: one load on a word boundary, else two two-byte loads, as
     * Q6_K's 210-byte and Q8_0's 34-byte blocks leave every other block two bytes off a word
     * boundary.
     */
    private static int word(ByteArray w, int at) {
        if ((at & 3) == 0) {
            return w.getInt(at);
        }
        return (w.getHalfFloat(at).getHalfFloatValue() & 0xFFFF)
                | ((w.getHalfFloat(at + 2).getHalfFloatValue() & 0xFFFF) << 16);
    }

    // @formatter:off
    /**
     * The grouped expert GEMM over Q8_0 experts on tensor cores (CUDA), {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it: one workgroup per half of a tile (32 rows)
     * and 256 outputs, eight warps of 32 rows by 32 outputs, the depth advancing 64 at a time, two
     * 34-byte blocks. Row {@code i} of a half reads FP16 input row {@code sortedToken[first + i]}
     * when {@code gather} is 1, {@code first + i} otherwise, copied to local memory as {@link
     * #groupedExpertMmaQ4_K} copies it; the next step's copy and weight loads are in flight while
     * this step multiplies. The weights never reach local memory: for each 16-deep chunk, lane
     * {@code L} loads the code word at byte {@code 4 (L % 4)} of its own column {@code L / 4} (two
     * 2-byte loads where a block's codes start off a word boundary), turns each code into its exact
     * half in registers and passes them to the MMA with {@code mmaFragmentB}. Each block's sums are
     * scaled in FP32 by its scale. A half with no rows of its tile returns before its first
     * barrier. Example: LFM2.5-24B-A2B's dense gate, 11776 outputs over {@code n = 2048}: 46
     * workgroups per half and 32 steps. Requires {@code n % 64 == 0} and {@code d % 256 == 0}.
     * Grid: {@link #splitWorkgroups} workgroups of {@link #THREADS} threads.
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
        // B: four columns per thread, eight apart; per chunk, the code word at byte 4 (lane % 4)
        // of the chunk's 16.
        int bCol = warp * 32 + (lane >> 2);
        int bByte = 2 + 4 * (lane & 3);
        int blocksPerRow = n >> 5;
        int rowStride = blocksPerRow * 34;
        int wRow = expert * d + blockCol;
        // Scales: one column per thread, both blocks of the step.
        int sRow = (wRow + tid) * blocksPerRow;

        int r0 = lane >> 2;
        int r1 = r0 + 16;
        int lc0 = warp * 32 + ((lane & 3) << 1);
        int lc1 = lc0 + 8;
        int lc2 = lc0 + 16;
        int lc3 = lc0 + 24;
        int q00 = 0;
        int q01 = 0;
        int q02 = 0;
        int q03 = 0;
        int q10 = 0;
        int q11 = 0;
        int q12 = 0;
        int q13 = 0;
        int q20 = 0;
        int q21 = 0;
        int q22 = 0;
        int q23 = 0;
        int q30 = 0;
        int q31 = 0;
        int q32 = 0;
        int q33 = 0;
        float rs0 = 0.0f;
        float rs1 = 0.0f;
        ctx.asyncCopyToLocal(aTile, aDst, input16, aSrc);
        ctx.asyncCopyToLocal(aTile, aDst + 1, input16, aSrc + 4);
        ctx.asyncCopyToLocal(aTile, aDst + 2, input16, aSrc + 8);
        ctx.asyncCopyToLocal(aTile, aDst + 3, input16, aSrc + 12);
        ctx.asyncCopyCommit();
        {
            int kBlock = 0;
            int c0 = (wRow + bCol) * rowStride + kBlock * 34 + bByte;
            q00 = word(w, c0);
            q01 = word(w, c0 + 16);
            q02 = word(w, c0 + 34);
            q03 = word(w, c0 + 50);
            int c1 = c0 + 8 * rowStride;
            q10 = word(w, c1);
            q11 = word(w, c1 + 16);
            q12 = word(w, c1 + 34);
            q13 = word(w, c1 + 50);
            int c2 = c0 + 16 * rowStride;
            q20 = word(w, c2);
            q21 = word(w, c2 + 16);
            q22 = word(w, c2 + 34);
            q23 = word(w, c2 + 50);
            int c3 = c0 + 24 * rowStride;
            q30 = word(w, c3);
            q31 = word(w, c3 + 16);
            q32 = word(w, c3 + 34);
            q33 = word(w, c3 + 50);
            int sBase = (sRow + kBlock) * 34;
            rs0 = w.getHalfFloat(sBase).getFloat32();
            rs1 = w.getHalfFloat(sBase + 34).getFloat32();
        }

        int kSteps = n >> 6;
        for (int step = 0; step < kSteps; step++) {
            int kBase = step << 6;
            int buf = step & 1;
            // This step's code words; the registers are reloaded for the next step below.
            int u00 = q00;
            int u01 = q01;
            int u02 = q02;
            int u03 = q03;
            int u10 = q10;
            int u11 = q11;
            int u12 = q12;
            int u13 = q13;
            int u20 = q20;
            int u21 = q21;
            int u22 = q22;
            int u23 = q23;
            int u30 = q30;
            int u31 = q31;
            int u32 = q32;
            int u33 = q33;
            colScale[tid] = rs0;
            colScale[SPLIT_OUTPUTS + tid] = rs1;
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
                    int kBlock = (kBase + SPLIT_DEPTH) >> 5;
                    int c0 = (wRow + bCol) * rowStride + kBlock * 34 + bByte;
                    q00 = word(w, c0);
                    q01 = word(w, c0 + 16);
                    q02 = word(w, c0 + 34);
                    q03 = word(w, c0 + 50);
                    int c1 = c0 + 8 * rowStride;
                    q10 = word(w, c1);
                    q11 = word(w, c1 + 16);
                    q12 = word(w, c1 + 34);
                    q13 = word(w, c1 + 50);
                    int c2 = c0 + 16 * rowStride;
                    q20 = word(w, c2);
                    q21 = word(w, c2 + 16);
                    q22 = word(w, c2 + 34);
                    q23 = word(w, c2 + 50);
                    int c3 = c0 + 24 * rowStride;
                    q30 = word(w, c3);
                    q31 = word(w, c3 + 16);
                    q32 = word(w, c3 + 34);
                    q33 = word(w, c3 + 50);
                    int sBase = (sRow + kBlock) * 34;
                    rs0 = w.getHalfFloat(sBase).getFloat32();
                    rs1 = w.getHalfFloat(sBase + 34).getFloat32();
                }
            }
            int aByte = buf * SPLIT_INPUT_INTS * 4;
            for (int c = 0; c < 4; c++) {
                HalfFloat[] a0 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024);
                HalfFloat[] a1 = ctx.mmaLoadA(aTile, 16, aByte + c * 1024 + 512);
                // Chunk c: word c; each signed code becomes its exact half in the lane's own
                // registers.
                int v0 = c == 0 ? u00 : c == 1 ? u01 : c == 2 ? u02 : u03;
                int v1 = c == 0 ? u10 : c == 1 ? u11 : c == 2 ? u12 : u13;
                int v2 = c == 0 ? u20 : c == 1 ? u21 : c == 2 ? u22 : u23;
                int v3 = c == 0 ? u30 : c == 1 ? u31 : c == 2 ? u32 : u33;
                HalfFloat[] b0 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v0 << 24) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v0 << 16) >> 24))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v0 << 8) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v0 >> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b1 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v1 << 24) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v1 << 16) >> 24))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v1 << 8) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v1 >> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b2 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v2 << 24) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v2 << 16) >> 24))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v2 << 8) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v2 >> 24)).getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16));
                HalfFloat[] b3 =
                        ctx.mmaFragmentB(
                                (new HalfFloat((float) ((v3 << 24) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) ((v3 << 16) >> 24))
                                                                .getHalfFloatValue()
                                                        & 0xFFFF)
                                                << 16),
                                (new HalfFloat((float) ((v3 << 8) >> 24)).getHalfFloatValue()
                                                & 0xFFFF)
                                        | ((new HalfFloat((float) (v3 >> 24)).getHalfFloatValue()
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
                // A block's two chunks summed, scaled by the block's scale.
                if ((c & 1) == 1) {
                    int g = (c >> 1) * SPLIT_OUTPUTS;
                    t00_0 += colScale[g + lc0] * s00[0];
                    t00_1 += colScale[g + lc0 + 1] * s00[1];
                    t00_2 += colScale[g + lc0] * s00[2];
                    t00_3 += colScale[g + lc0 + 1] * s00[3];
                    t01_0 += colScale[g + lc1] * s01[0];
                    t01_1 += colScale[g + lc1 + 1] * s01[1];
                    t01_2 += colScale[g + lc1] * s01[2];
                    t01_3 += colScale[g + lc1 + 1] * s01[3];
                    t02_0 += colScale[g + lc2] * s02[0];
                    t02_1 += colScale[g + lc2 + 1] * s02[1];
                    t02_2 += colScale[g + lc2] * s02[2];
                    t02_3 += colScale[g + lc2 + 1] * s02[3];
                    t03_0 += colScale[g + lc3] * s03[0];
                    t03_1 += colScale[g + lc3 + 1] * s03[1];
                    t03_2 += colScale[g + lc3] * s03[2];
                    t03_3 += colScale[g + lc3 + 1] * s03[3];
                    t10_0 += colScale[g + lc0] * s10[0];
                    t10_1 += colScale[g + lc0 + 1] * s10[1];
                    t10_2 += colScale[g + lc0] * s10[2];
                    t10_3 += colScale[g + lc0 + 1] * s10[3];
                    t11_0 += colScale[g + lc1] * s11[0];
                    t11_1 += colScale[g + lc1 + 1] * s11[1];
                    t11_2 += colScale[g + lc1] * s11[2];
                    t11_3 += colScale[g + lc1 + 1] * s11[3];
                    t12_0 += colScale[g + lc2] * s12[0];
                    t12_1 += colScale[g + lc2 + 1] * s12[1];
                    t12_2 += colScale[g + lc2] * s12[2];
                    t12_3 += colScale[g + lc2 + 1] * s12[3];
                    t13_0 += colScale[g + lc3] * s13[0];
                    t13_1 += colScale[g + lc3 + 1] * s13[1];
                    t13_2 += colScale[g + lc3] * s13[2];
                    t13_3 += colScale[g + lc3 + 1] * s13[3];
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
}
