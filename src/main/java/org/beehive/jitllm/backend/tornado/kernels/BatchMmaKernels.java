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
 * Batched tensor-core building blocks shared by the prefill paths of several model families: a
 * strided FP16 GEMM on the MMA units with FP32 accumulation, a causal softmax over score rows, and
 * FP32-to-FP16 conversion.
 */
// @formatter:on
public final class BatchMmaKernels {

    private BatchMmaKernels() {}

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#gemmMMA} with leading dimensions and a batch (each
     * operand's problem index shifted, and B's divided, as the offsets and {@code groupB} say):
     * problem {@code z = groupIdz} computes {@code C[r][colOffset + c] = A[r] . B[c]} for {@code r
     * < M, c < N} over {@code K}, where {@code A[r]} starts at {@code z * aBatch + r * lda}, {@code
     * B[c]} at {@code z * bBatch + c * ldb}, and {@code C}'s rows are {@code ldc} apart with {@code
     * colOffset = z * cBatchCols}. Requires M % 128 == 0, N % 128 == 0, K % 16 == 0. Worker:
     * WorkerGrid3D((M/128)*256, N/128, batches), local (256,1,1).
     */
    // @formatter:on
    public static void gemmMMAStrided(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols,
            IntArray batchInfo,
            int boundN,
            int boundK,
            int zOffsetA,
            int zOffsetB,
            int groupB,
            int zOffsetC) {
        // With a bound set, N or K stops at the chunk's last position rounded up to whole tiles:
        // the positions past it hold nothing the attention reads.
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        int limitN = boundN * Math.min(N, filled) + (1 - boundN) * N;
        int limitK = boundK * Math.min(K, filled) + (1 - boundK) * K;
        if (BN * ctx.groupIdy >= limitN) {
            return;
        }
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        // Problem z reads A's problem z + zOffsetA, B's (z + zOffsetB) / groupB (several query
        // heads
        // sharing one key/value head), and writes C's z + zOffsetC.
        int z = ctx.groupIdz;
        int aBase = (z + zOffsetA) * aBatch;
        int bBase = ((z + zOffsetB) / groupB) * bBatch;
        int colOffset = (z + zOffsetC) * cBatchCols;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = aBase + (blockRow + (aIdx0 >>> 3)) * lda + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = aBase + (blockRow + (aIdx1 >>> 3)) * lda + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = aBase + (blockRow + (aIdx2 >>> 3)) * lda + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = aBase + (blockRow + (aIdx3 >>> 3)) * lda + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 =
                bBase
                        + (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * ldb
                        + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 =
                bBase
                        + (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * ldb
                        + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 =
                bBase
                        + (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * ldb
                        + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 =
                bBase
                        + (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * ldb
                        + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + ldb);
        int bReg1 = packHalves(B, gB1, gB1 + ldb);
        int bReg2 = packHalves(B, gB2, gB2 + ldb);
        int bReg3 = packHalves(B, gB3, gB3 + ldb);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = limitK / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + ldb);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + ldb);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + ldb);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + ldb);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBaseTile = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = colOffset + blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, ldc);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, ldc);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, ldc);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, ldc);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, ldc);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, ldc);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, ldc);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, ldc);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, ldc);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, ldc);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, ldc);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, ldc);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, ldc);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, ldc);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, ldc);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, ldc);
    }

    // @formatter:off
    /**
     * The causal softmax of every (token, head) row of scores, in FP16: row {@code token * heads +
     * head} attends to positions {@code 0 .. start + token}, scaled; every other entry, and every
     * row past the active ones, is zero. A workgroup of {@link #GROUP} lanes per row.
     */
    // @formatter:on
    public static void causalSoftmax(
            KernelContext context,
            IntArray batchInfo,
            FloatArray scores,
            HalfFloatArray probs,
            int heads,
            int paddedPositions,
            float scale) {
        int row = context.groupIdx;
        int tid = context.localIdx;
        int t = row / heads;
        int base = row * paddedPositions;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        // Zero for a padding row, without a branch whose merged value the backend miscompiles.
        int active = Math.min(1, Math.max(0, batchInfo.get(1) - t));
        int valid = active * (batchInfo.get(0) + t + 1);
        float localMax = Float.NEGATIVE_INFINITY;
        for (int j = tid; j < valid; j += GROUP) {
            localMax = TornadoMath.max(localMax, scores.get(base + j) * scale);
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
            }
            context.localBarrier();
        }
        float max = reduce[0];
        context.localBarrier();
        float localSum = 0.0f;
        for (int j = tid; j < valid; j += GROUP) {
            localSum += TornadoMath.exp(scores.get(base + j) * scale - max);
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float inverse = 1.0f / TornadoMath.max(reduce[0], 1.0e-30f);
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        int width = Math.min(paddedPositions, filled);
        for (int j = tid; j < width; j += GROUP) {
            if (j < valid) {
                probs.set(
                        base + j,
                        new HalfFloat(
                                TornadoMath.exp(scores.get(base + j) * scale - max) * inverse));
            }
            if (j >= valid) {
                probs.set(base + j, new HalfFloat(0.0f));
            }
        }
    }

    /**
     * The attention output of a values GEMM run {@code wide} columns per head, back to {@code
     * headSize} columns per head and narrowed to FP16: {@code out[t][h * headSize + c] = in[t][h *
     * wide + c]} over {@code rows * heads * headSize} values. A head narrower than the GEMM's
     * 128-wide column tile is computed 128 wide; the extra columns are dropped here.
     */
    public static void narrowHeadsToHalf(
            KernelContext context,
            FloatArray in,
            HalfFloatArray out,
            int heads,
            int headSize,
            int wide,
            int n) {
        int i = context.globalIdx;
        if (i < n) {
            int rowWidth = heads * headSize;
            int t = i / rowWidth;
            int col = i - t * rowWidth;
            int h = col / headSize;
            int c = col - h * headSize;
            out.set(i, new HalfFloat(in.get(t * heads * wide + h * wide + c)));
        }
    }

    /** {@code out[i] = in[i]} narrowed to FP16, for {@code n} values. */
    public static void toHalf(KernelContext context, FloatArray in, HalfFloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            out.set(i, new HalfFloat(in.get(i)));
        }
    }

    private static int packHalves(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    private static final int WARPS_M = 4, WARPS_N = 2;

    private static final int WARP_SIZE = 32;

    private static final int BM = 128, BN = 128, BK = 16;

    private static final int B_SUBTILE_BYTES = 256;

    private static final int WM = BM / WARPS_M;

    private static final int WN = BN / WARPS_N;

    /** Lanes of a row-wise workgroup. */
    public static final int GROUP = 256;
}
