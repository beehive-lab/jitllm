package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Half;
import uk.ac.manchester.tornado.api.types.vectors.Float4;

// @formatter:off
/**
 * Grouped expert GEMMs on SIMD-group matrices (Metal), for a batched mixture-of-experts prefill:
 * the dense {@code TransformerBatchPrefillSimdgroupKernels.batchedGemm*} K-quant and Q8_0 kernels
 * run once per tile of one expert's assignments, so a chunk reads each selected expert's weights
 * once per tile instead of once per token. Each kernel is its dense twin's staging and arithmetic
 * with three changes: the row range comes from the tile table, the weight rows are offset to the
 * tile's expert, and the input rows may be gathered through the sorted assignment order. An output
 * row depends on its own input row only, so a token's result does not depend on which tile or chunk
 * it fell in.
 */
// @formatter:on
public final class GroupedExpertSimdgroupKernels {

    private GroupedExpertSimdgroupKernels() {}

    private static final int LDS = TransformerBatchPrefillSimdgroupKernels.LDS;

    /** Workgroup threads, as the dense twins'. */
    public static final int THREADS = TransformerBatchPrefillSimdgroupKernels.THREADS;

    /** The input row of a tile's local row {@code local}. */
    private static int sourceRow(IntArray sortedToken, int gather, int first, int local) {
        return gather != 0 ? sortedToken.get(first + local) : first + local;
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ4_K} over one tile of expert rows:
     * {@code tiles} as {@code Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first
     * sorted row, rows)} per tile, the expert's {@code d} weight rows read in place from the
     * stacked tensor. Row {@code i} of a tile reads input row {@code sortedToken[first + i]} when
     * {@code gather} is 1 (the chunk's normed tokens), {@code first + i} otherwise (an earlier
     * grouped GEMM's sorted output), and writes output row {@code first + i}. Grid: {@code ceil(d /
     * 64) * maxTiles} workgroups of {@code TransformerBatchPrefillSimdgroupKernels.THREADS}
     * threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertGemmQ4_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        int qn;
        int qs;
        int qsub;
        int qsm;
        float pmn;
        Float4 fx00 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 0, batch - 1))
                                        * n
                                + 0
                                + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 32, batch - 1))
                                        * n
                                + 0
                                + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 144;
        qsub = ((0) >> 5) & 7;
        qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, qsub);
        psc = w.getHalfFloat(qb).getFloat32() * (qsm >> 8);
        pmn = w.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
        qn = qb + 16 + (qsub >> 1) * 32 + wk0;
        qs = (qsub & 1) << 2;
        pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15);
        pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15);
        pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15);
        pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15);
        pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15);
        pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15);
        pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15);
        pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15);
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(vr + 0) * LDS + vc + 0] = px0_0;
            xs[(vr + 0) * LDS + vc + 1] = px0_1;
            xs[(vr + 0) * LDS + vc + 2] = px0_2;
            xs[(vr + 0) * LDS + vc + 3] = px0_3;
            xs[(vr + 32) * LDS + vc + 0] = px1_0;
            xs[(vr + 32) * LDS + vc + 1] = px1_1;
            xs[(vr + 32) * LDS + vc + 2] = px1_2;
            xs[(vr + 32) * LDS + vc + 3] = px1_3;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc - pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc - pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc - pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc - pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc - pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc - pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc - pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc - pmn;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 0, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 32, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb =
                        ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8)
                                        + ((k0 + 32) >> 8))
                                * 144;
                qsub = ((k0 + 32) >> 5) & 7;
                qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, qsub);
                psc = w.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                pmn = w.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                qn = qb + 16 + (qsub >> 1) * 32 + wk0;
                qs = (qsub & 1) << 2;
                pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15);
                pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15);
                pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15);
                pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15);
                pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15);
                pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15);
                pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15);
                pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15);
            }
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (sgRow < batch) {
                for (int kk = 0; kk < 32; kk += 8) {
                    Matrix8x8Float x0 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                    Matrix8x8Float x1 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                    Matrix8x8Float x2 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                    Matrix8x8Float x3 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                    Matrix8x8Float cw0 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                    Matrix8x8Float cw1 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                    c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                    c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                    c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                    c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                    c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                    c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
                }
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ8WithResidual} (its weight
     * staging, without the residual) over one tile of expert rows: {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)} per
     * tile, the expert's {@code d} weight rows read in place from the stacked tensor. Row {@code i}
     * of a tile reads input row {@code sortedToken[first + i]} when {@code gather} is 1 (the
     * chunk's normed tokens), {@code first + i} otherwise (an earlier grouped GEMM's sorted
     * output), and writes output row {@code first + i}. Grid: {@code ceil(d / 64) * maxTiles}
     * workgroups of {@code TransformerBatchPrefillSimdgroupKernels.THREADS} threads; a workgroup
     * past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertGemmQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        Float4 fx00 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 0, batch - 1))
                                        * n
                                + 0
                                + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 32, batch - 1))
                                        * n
                                + 0
                                + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 34;
        psc = w.getHalfFloat(qb).getFloat32();
        pw0 = (float) w.get(qb + 2 + wk0 + 0);
        pw1 = (float) w.get(qb + 2 + wk0 + 1);
        pw2 = (float) w.get(qb + 2 + wk0 + 2);
        pw3 = (float) w.get(qb + 2 + wk0 + 3);
        pw4 = (float) w.get(qb + 2 + wk0 + 4);
        pw5 = (float) w.get(qb + 2 + wk0 + 5);
        pw6 = (float) w.get(qb + 2 + wk0 + 6);
        pw7 = (float) w.get(qb + 2 + wk0 + 7);
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(vr + 0) * LDS + vc + 0] = px0_0;
            xs[(vr + 0) * LDS + vc + 1] = px0_1;
            xs[(vr + 0) * LDS + vc + 2] = px0_2;
            xs[(vr + 0) * LDS + vc + 3] = px0_3;
            xs[(vr + 32) * LDS + vc + 0] = px1_0;
            xs[(vr + 32) * LDS + vc + 1] = px1_1;
            xs[(vr + 32) * LDS + vc + 2] = px1_2;
            xs[(vr + 32) * LDS + vc + 3] = px1_3;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 0, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 32, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb =
                        ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5)
                                        + ((k0 + 32) >> 5))
                                * 34;
                psc = w.getHalfFloat(qb).getFloat32();
                pw0 = (float) w.get(qb + 2 + wk0 + 0);
                pw1 = (float) w.get(qb + 2 + wk0 + 1);
                pw2 = (float) w.get(qb + 2 + wk0 + 2);
                pw3 = (float) w.get(qb + 2 + wk0 + 3);
                pw4 = (float) w.get(qb + 2 + wk0 + 4);
                pw5 = (float) w.get(qb + 2 + wk0 + 5);
                pw6 = (float) w.get(qb + 2 + wk0 + 6);
                pw7 = (float) w.get(qb + 2 + wk0 + 7);
            }
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (sgRow < batch) {
                for (int kk = 0; kk < 32; kk += 8) {
                    Matrix8x8Float x0 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                    Matrix8x8Float x1 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                    Matrix8x8Float x2 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                    Matrix8x8Float x3 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                    Matrix8x8Float cw0 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                    Matrix8x8Float cw1 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                    c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                    c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                    c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                    c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                    c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                    c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
                }
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ5_K} over one tile of expert rows:
     * {@code tiles} as {@code Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first
     * sorted row, rows)} per tile, the expert's {@code d} weight rows read in place from the
     * stacked tensor. Row {@code i} of a tile reads input row {@code sortedToken[first + i]} when
     * {@code gather} is 1 (the chunk's normed tokens), {@code first + i} otherwise (an earlier
     * grouped GEMM's sorted output), and writes output row {@code first + i}. Grid: {@code ceil(d /
     * 64) * maxTiles} workgroups of {@code TransformerBatchPrefillSimdgroupKernels.THREADS}
     * threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertGemmQ5_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        int qn;
        int qs;
        int qh;
        int qlw0;
        int qlw1;
        int qhw0;
        int qhw1;
        int raw0;
        int raw1;
        int qsub;
        int qsm;
        float pmn;
        Float4 fx00 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 0, batch - 1))
                                        * n
                                + 0
                                + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 32, batch - 1))
                                        * n
                                + 0
                                + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 176;
        qsub = ((0) >> 5) & 7;
        qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, qsub);
        psc = w.getHalfFloat(qb).getFloat32() * (qsm >> 8);
        pmn = w.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
        qn = qb + 48 + (qsub >> 1) * 32 + wk0;
        qh = qb + 16 + wk0;
        qs = (qsub & 1) << 2;
        qlw0 =
                (w.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
        qlw1 =
                (w.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
        qhw0 =
                (w.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
        qhw1 =
                (w.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
        raw0 = ((qlw0 >>> qs) & 0x0F0F0F0F) | (((qhw0 >>> qsub) & 0x01010101) << 4);
        raw1 = ((qlw1 >>> qs) & 0x0F0F0F0F) | (((qhw1 >>> qsub) & 0x01010101) << 4);
        pw0 = (float) (raw0 & 0xFF);
        pw1 = (float) ((raw0 >>> 8) & 0xFF);
        pw2 = (float) ((raw0 >>> 16) & 0xFF);
        pw3 = (float) (raw0 >>> 24);
        pw4 = (float) (raw1 & 0xFF);
        pw5 = (float) ((raw1 >>> 8) & 0xFF);
        pw6 = (float) ((raw1 >>> 16) & 0xFF);
        pw7 = (float) (raw1 >>> 24);
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(vr + 0) * LDS + vc + 0] = px0_0;
            xs[(vr + 0) * LDS + vc + 1] = px0_1;
            xs[(vr + 0) * LDS + vc + 2] = px0_2;
            xs[(vr + 0) * LDS + vc + 3] = px0_3;
            xs[(vr + 32) * LDS + vc + 0] = px1_0;
            xs[(vr + 32) * LDS + vc + 1] = px1_1;
            xs[(vr + 32) * LDS + vc + 2] = px1_2;
            xs[(vr + 32) * LDS + vc + 3] = px1_3;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc - pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc - pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc - pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc - pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc - pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc - pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc - pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc - pmn;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 0, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 32, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb =
                        ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8)
                                        + ((k0 + 32) >> 8))
                                * 176;
                qsub = ((k0 + 32) >> 5) & 7;
                qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, qsub);
                psc = w.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                pmn = w.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                qn = qb + 48 + (qsub >> 1) * 32 + wk0;
                qh = qb + 16 + wk0;
                qs = (qsub & 1) << 2;
                qlw0 =
                        (w.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
                qlw1 =
                        (w.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
                qhw0 =
                        (w.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
                qhw1 =
                        (w.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
                raw0 = ((qlw0 >>> qs) & 0x0F0F0F0F) | (((qhw0 >>> qsub) & 0x01010101) << 4);
                raw1 = ((qlw1 >>> qs) & 0x0F0F0F0F) | (((qhw1 >>> qsub) & 0x01010101) << 4);
                pw0 = (float) (raw0 & 0xFF);
                pw1 = (float) ((raw0 >>> 8) & 0xFF);
                pw2 = (float) ((raw0 >>> 16) & 0xFF);
                pw3 = (float) (raw0 >>> 24);
                pw4 = (float) (raw1 & 0xFF);
                pw5 = (float) ((raw1 >>> 8) & 0xFF);
                pw6 = (float) ((raw1 >>> 16) & 0xFF);
                pw7 = (float) (raw1 >>> 24);
            }
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (sgRow < batch) {
                for (int kk = 0; kk < 32; kk += 8) {
                    Matrix8x8Float x0 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                    Matrix8x8Float x1 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                    Matrix8x8Float x2 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                    Matrix8x8Float x3 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                    Matrix8x8Float cw0 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                    Matrix8x8Float cw1 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                    c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                    c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                    c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                    c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                    c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                    c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
                }
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ6_K} over one tile of expert rows:
     * {@code tiles} as {@code Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first
     * sorted row, rows)} per tile, the expert's {@code d} weight rows read in place from the
     * stacked tensor. Row {@code i} of a tile reads input row {@code sortedToken[first + i]} when
     * {@code gather} is 1 (the chunk's normed tokens), {@code first + i} otherwise (an earlier
     * grouped GEMM's sorted output), and writes output row {@code first + i}. Grid: {@code ceil(d /
     * 64) * maxTiles} workgroups of {@code TransformerBatchPrefillSimdgroupKernels.THREADS}
     * threads; a workgroup past the tile count returns before its first barrier.
     */
    // @formatter:on
    public static void groupedExpertGemmQ6_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        int qn;
        int qs;
        int qsub;
        int qsm;
        int qh;
        int qhs;
        int qlw0;
        int qlw1;
        int qhw0;
        int qhw1;
        int raw0;
        int raw1;
        Float4 fx00 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 0, batch - 1))
                                        * n
                                + 0
                                + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 =
                inputBatch.getFloat4(
                        sourceRow(
                                                sortedToken,
                                                gather,
                                                first,
                                                TornadoMath.min(b0 + vr + 32, batch - 1))
                                        * n
                                + 0
                                + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 210;
        qsub = ((0) >> 5) & 7;
        qsm = w.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
        psc = w.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
        qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
        qh = qb + 128 + (qsub >> 2) * 32 + wk0;
        qs = ((qsub & 3) >> 1) << 2;
        qhs = (qsub & 3) << 1;
        qlw0 =
                (w.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
        qlw1 =
                (w.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
        qhw0 =
                (w.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
        qhw1 =
                (w.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                        | ((w.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
        raw0 = ((qlw0 >>> qs) & 0x0F0F0F0F) | (((qhw0 >>> qhs) & 0x03030303) << 4);
        raw1 = ((qlw1 >>> qs) & 0x0F0F0F0F) | (((qhw1 >>> qhs) & 0x03030303) << 4);
        pw0 = (float) (raw0 & 0xFF) - 32.0f;
        pw1 = (float) ((raw0 >>> 8) & 0xFF) - 32.0f;
        pw2 = (float) ((raw0 >>> 16) & 0xFF) - 32.0f;
        pw3 = (float) (raw0 >>> 24) - 32.0f;
        pw4 = (float) (raw1 & 0xFF) - 32.0f;
        pw5 = (float) ((raw1 >>> 8) & 0xFF) - 32.0f;
        pw6 = (float) ((raw1 >>> 16) & 0xFF) - 32.0f;
        pw7 = (float) (raw1 >>> 24) - 32.0f;
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(vr + 0) * LDS + vc + 0] = px0_0;
            xs[(vr + 0) * LDS + vc + 1] = px0_1;
            xs[(vr + 0) * LDS + vc + 2] = px0_2;
            xs[(vr + 0) * LDS + vc + 3] = px0_3;
            xs[(vr + 32) * LDS + vc + 0] = px1_0;
            xs[(vr + 32) * LDS + vc + 1] = px1_1;
            xs[(vr + 32) * LDS + vc + 2] = px1_2;
            xs[(vr + 32) * LDS + vc + 3] = px1_3;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 0, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                sourceRow(
                                                        sortedToken,
                                                        gather,
                                                        first,
                                                        TornadoMath.min(b0 + vr + 32, batch - 1))
                                                * n
                                        + k0
                                        + 32
                                        + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb =
                        ((expert * d + TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8)
                                        + ((k0 + 32) >> 8))
                                * 210;
                qsub = ((k0 + 32) >> 5) & 7;
                qsm = w.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
                psc = w.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
                qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
                qh = qb + 128 + (qsub >> 2) * 32 + wk0;
                qs = ((qsub & 3) >> 1) << 2;
                qhs = (qsub & 3) << 1;
                qlw0 =
                        (w.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
                qlw1 =
                        (w.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
                qhw0 =
                        (w.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
                qhw1 =
                        (w.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
                raw0 = ((qlw0 >>> qs) & 0x0F0F0F0F) | (((qhw0 >>> qhs) & 0x03030303) << 4);
                raw1 = ((qlw1 >>> qs) & 0x0F0F0F0F) | (((qhw1 >>> qhs) & 0x03030303) << 4);
                pw0 = (float) (raw0 & 0xFF) - 32.0f;
                pw1 = (float) ((raw0 >>> 8) & 0xFF) - 32.0f;
                pw2 = (float) ((raw0 >>> 16) & 0xFF) - 32.0f;
                pw3 = (float) (raw0 >>> 24) - 32.0f;
                pw4 = (float) (raw1 & 0xFF) - 32.0f;
                pw5 = (float) ((raw1 >>> 8) & 0xFF) - 32.0f;
                pw6 = (float) ((raw1 >>> 16) & 0xFF) - 32.0f;
                pw7 = (float) (raw1 >>> 24) - 32.0f;
            }
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (sgRow < batch) {
                for (int kk = 0; kk < 32; kk += 8) {
                    Matrix8x8Float x0 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                    Matrix8x8Float x1 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                    Matrix8x8Float x2 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                    Matrix8x8Float x3 =
                            context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                    Matrix8x8Float cw0 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                    Matrix8x8Float cw1 =
                            context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                    c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                    c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                    c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                    c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                    c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                    c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
                }
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (first + b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (first + b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (first + b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (first + b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ4_KHalfTiles} over one tile of
     * expert rows: the FP16 weight tiles of {@link #groupedExpertGemmQ4_K}'s twin, for a grid of at
     * least {@code TransformerBatchPrefillSimdgroupKernels.HALF_TILES_MIN_GROUPS} workgroups. Grid:
     * {@code ceil(d / 64) * maxTiles} workgroups of {@code
     * TransformerBatchPrefillSimdgroupKernels.HALF_TILES_THREADS} threads.
     */
    // @formatter:on
    public static void groupedExpertGemmQ4_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = (expert * d + TornadoMath.min(r0 + lr, d - 1)) * (n >> 8) * 144;
        int xi =
                sourceRow(sortedToken, gather, first, TornadoMath.min(b0 + lr, batch - 1)) * n + kh;
        int cbase = (first + b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
        Matrix8x8Float c0 = context.simdgroupMatrixZero();
        Matrix8x8Float c1 = context.simdgroupMatrixZero();
        Matrix8x8Float c2 = context.simdgroupMatrixZero();
        Matrix8x8Float c3 = context.simdgroupMatrixZero();
        Matrix8x8Float c4 = context.simdgroupMatrixZero();
        Matrix8x8Float c5 = context.simdgroupMatrixZero();
        Matrix8x8Float c6 = context.simdgroupMatrixZero();
        Matrix8x8Float c7 = context.simdgroupMatrixZero();
        Matrix8x8Float c8 = context.simdgroupMatrixZero();
        Matrix8x8Float c9 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c12 = context.simdgroupMatrixZero();
        Matrix8x8Float c13 = context.simdgroupMatrixZero();
        Matrix8x8Float c14 = context.simdgroupMatrixZero();
        Matrix8x8Float c15 = context.simdgroupMatrixZero();
        for (int k0 = 0; k0 < n; k0 += 32) {
            float a0;
            float a1;
            float a2;
            float a3;
            float a4;
            float a5;
            float a6;
            float a7;
            float a8;
            float a9;
            float a10;
            float a11;
            float a12;
            float a13;
            float a14;
            float a15;
            int qb = rb + (k0 >> 8) * 144;
            int sub = (k0 >> 5) & 7;
            int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, sub);
            float sc = w.getHalfFloat(qb).getFloat32() * (sm >> 8);
            float mn = w.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
            int qn = qb + 16 + (sub >> 1) * 32 + kh;
            int sh = (sub & 1) << 2;
            a0 = (float) (((w.get(qn + 0) & 0xFF) >> sh) & 15) * sc - mn;
            a1 = (float) (((w.get(qn + 1) & 0xFF) >> sh) & 15) * sc - mn;
            a2 = (float) (((w.get(qn + 2) & 0xFF) >> sh) & 15) * sc - mn;
            a3 = (float) (((w.get(qn + 3) & 0xFF) >> sh) & 15) * sc - mn;
            a4 = (float) (((w.get(qn + 4) & 0xFF) >> sh) & 15) * sc - mn;
            a5 = (float) (((w.get(qn + 5) & 0xFF) >> sh) & 15) * sc - mn;
            a6 = (float) (((w.get(qn + 6) & 0xFF) >> sh) & 15) * sc - mn;
            a7 = (float) (((w.get(qn + 7) & 0xFF) >> sh) & 15) * sc - mn;
            a8 = (float) (((w.get(qn + 8) & 0xFF) >> sh) & 15) * sc - mn;
            a9 = (float) (((w.get(qn + 9) & 0xFF) >> sh) & 15) * sc - mn;
            a10 = (float) (((w.get(qn + 10) & 0xFF) >> sh) & 15) * sc - mn;
            a11 = (float) (((w.get(qn + 11) & 0xFF) >> sh) & 15) * sc - mn;
            a12 = (float) (((w.get(qn + 12) & 0xFF) >> sh) & 15) * sc - mn;
            a13 = (float) (((w.get(qn + 13) & 0xFF) >> sh) & 15) * sc - mn;
            a14 = (float) (((w.get(qn + 14) & 0xFF) >> sh) & 15) * sc - mn;
            a15 = (float) (((w.get(qn + 15) & 0xFF) >> sh) & 15) * sc - mn;
            float y0 = inputBatch.get(xi + 0);
            float y1 = inputBatch.get(xi + 1);
            float y2 = inputBatch.get(xi + 2);
            float y3 = inputBatch.get(xi + 3);
            float y4 = inputBatch.get(xi + 4);
            float y5 = inputBatch.get(xi + 5);
            float y6 = inputBatch.get(xi + 6);
            float y7 = inputBatch.get(xi + 7);
            float y8 = inputBatch.get(xi + 8);
            float y9 = inputBatch.get(xi + 9);
            float y10 = inputBatch.get(xi + 10);
            float y11 = inputBatch.get(xi + 11);
            float y12 = inputBatch.get(xi + 12);
            float y13 = inputBatch.get(xi + 13);
            float y14 = inputBatch.get(xi + 14);
            float y15 = inputBatch.get(xi + 15);
            context.localBarrier();
            sa[aBase + 0] = new HalfFloat(a0);
            sa[aBase + 8] = new HalfFloat(a1);
            sa[aBase + 16] = new HalfFloat(a2);
            sa[aBase + 24] = new HalfFloat(a3);
            sa[aBase + 32] = new HalfFloat(a4);
            sa[aBase + 40] = new HalfFloat(a5);
            sa[aBase + 48] = new HalfFloat(a6);
            sa[aBase + 56] = new HalfFloat(a7);
            sa[aBase + 512] = new HalfFloat(a8);
            sa[aBase + 520] = new HalfFloat(a9);
            sa[aBase + 528] = new HalfFloat(a10);
            sa[aBase + 536] = new HalfFloat(a11);
            sa[aBase + 544] = new HalfFloat(a12);
            sa[aBase + 552] = new HalfFloat(a13);
            sa[aBase + 560] = new HalfFloat(a14);
            sa[aBase + 568] = new HalfFloat(a15);
            sb[bBase + 0] = y0;
            sb[bBase + 1] = y1;
            sb[bBase + 2] = y2;
            sb[bBase + 3] = y3;
            sb[bBase + 4] = y4;
            sb[bBase + 5] = y5;
            sb[bBase + 6] = y6;
            sb[bBase + 7] = y7;
            sb[bBase + 512] = y8;
            sb[bBase + 513] = y9;
            sb[bBase + 514] = y10;
            sb[bBase + 515] = y11;
            sb[bBase + 516] = y12;
            sb[bBase + 517] = y13;
            sb[bBase + 518] = y14;
            sb[bBase + 519] = y15;
            context.localBarrier();
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (((sg >> 1) << 5) < batch) {
                for (int ik = 0; ik < 4; ik++) {
                    int pa = ca + (ik << 9);
                    int pb = cb + (ik << 9);
                    Matrix8x8Half m0 = context.simdgroupMatrixLoad(sa, pa + 0, 8);
                    Matrix8x8Half m1 = context.simdgroupMatrixLoad(sa, pa + 64, 8);
                    Matrix8x8Half m2 = context.simdgroupMatrixLoad(sa, pa + 128, 8);
                    Matrix8x8Half m3 = context.simdgroupMatrixLoad(sa, pa + 192, 8);
                    Matrix8x8Float n0 = context.simdgroupMatrixLoad(sb, pb + 0, 8);
                    Matrix8x8Float n1 = context.simdgroupMatrixLoad(sb, pb + 64, 8);
                    Matrix8x8Float n2 = context.simdgroupMatrixLoad(sb, pb + 128, 8);
                    Matrix8x8Float n3 = context.simdgroupMatrixLoad(sb, pb + 192, 8);
                    c0 = context.simdgroupMatrixMultiplyAccumulate(n0, m0, c0);
                    c1 = context.simdgroupMatrixMultiplyAccumulate(n0, m1, c1);
                    c2 = context.simdgroupMatrixMultiplyAccumulate(n0, m2, c2);
                    c3 = context.simdgroupMatrixMultiplyAccumulate(n0, m3, c3);
                    c4 = context.simdgroupMatrixMultiplyAccumulate(n1, m0, c4);
                    c5 = context.simdgroupMatrixMultiplyAccumulate(n1, m1, c5);
                    c6 = context.simdgroupMatrixMultiplyAccumulate(n1, m2, c6);
                    c7 = context.simdgroupMatrixMultiplyAccumulate(n1, m3, c7);
                    c8 = context.simdgroupMatrixMultiplyAccumulate(n2, m0, c8);
                    c9 = context.simdgroupMatrixMultiplyAccumulate(n2, m1, c9);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(n2, m2, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(n2, m3, c11);
                    c12 = context.simdgroupMatrixMultiplyAccumulate(n3, m0, c12);
                    c13 = context.simdgroupMatrixMultiplyAccumulate(n3, m1, c13);
                    c14 = context.simdgroupMatrixMultiplyAccumulate(n3, m2, c14);
                    c15 = context.simdgroupMatrixMultiplyAccumulate(n3, m3, c15);
                }
            }
            xi += 32;
        }
        if (full) {
            context.simdgroupMatrixStore(c0, outputBatch, cbase + 0 * d + 0, d);
            context.simdgroupMatrixStore(c1, outputBatch, cbase + 0 * d + 8, d);
            context.simdgroupMatrixStore(c2, outputBatch, cbase + 0 * d + 16, d);
            context.simdgroupMatrixStore(c3, outputBatch, cbase + 0 * d + 24, d);
            context.simdgroupMatrixStore(c4, outputBatch, cbase + 8 * d + 0, d);
            context.simdgroupMatrixStore(c5, outputBatch, cbase + 8 * d + 8, d);
            context.simdgroupMatrixStore(c6, outputBatch, cbase + 8 * d + 16, d);
            context.simdgroupMatrixStore(c7, outputBatch, cbase + 8 * d + 24, d);
            context.simdgroupMatrixStore(c8, outputBatch, cbase + 16 * d + 0, d);
            context.simdgroupMatrixStore(c9, outputBatch, cbase + 16 * d + 8, d);
            context.simdgroupMatrixStore(c10, outputBatch, cbase + 16 * d + 16, d);
            context.simdgroupMatrixStore(c11, outputBatch, cbase + 16 * d + 24, d);
            context.simdgroupMatrixStore(c12, outputBatch, cbase + 24 * d + 0, d);
            context.simdgroupMatrixStore(c13, outputBatch, cbase + 24 * d + 8, d);
            context.simdgroupMatrixStore(c14, outputBatch, cbase + 24 * d + 16, d);
            context.simdgroupMatrixStore(c15, outputBatch, cbase + 24 * d + 24, d);
        } else {
            int t0 = b0 + ((sg >> 1) << 5);
            int rr = r0 + ((sg & 1) << 5);
            context.localBarrier();
            context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ5_KHalfTiles} over one tile of
     * expert rows: the FP16 weight tiles of {@link #groupedExpertGemmQ5_K}'s twin, for a grid of at
     * least {@code TransformerBatchPrefillSimdgroupKernels.HALF_TILES_MIN_GROUPS} workgroups. Grid:
     * {@code ceil(d / 64) * maxTiles} workgroups of {@code
     * TransformerBatchPrefillSimdgroupKernels.HALF_TILES_THREADS} threads.
     */
    // @formatter:on
    public static void groupedExpertGemmQ5_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = (expert * d + TornadoMath.min(r0 + lr, d - 1)) * (n >> 8) * 176;
        int xi =
                sourceRow(sortedToken, gather, first, TornadoMath.min(b0 + lr, batch - 1)) * n + kh;
        int cbase = (first + b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
        Matrix8x8Float c0 = context.simdgroupMatrixZero();
        Matrix8x8Float c1 = context.simdgroupMatrixZero();
        Matrix8x8Float c2 = context.simdgroupMatrixZero();
        Matrix8x8Float c3 = context.simdgroupMatrixZero();
        Matrix8x8Float c4 = context.simdgroupMatrixZero();
        Matrix8x8Float c5 = context.simdgroupMatrixZero();
        Matrix8x8Float c6 = context.simdgroupMatrixZero();
        Matrix8x8Float c7 = context.simdgroupMatrixZero();
        Matrix8x8Float c8 = context.simdgroupMatrixZero();
        Matrix8x8Float c9 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c12 = context.simdgroupMatrixZero();
        Matrix8x8Float c13 = context.simdgroupMatrixZero();
        Matrix8x8Float c14 = context.simdgroupMatrixZero();
        Matrix8x8Float c15 = context.simdgroupMatrixZero();
        for (int k0 = 0; k0 < n; k0 += 32) {
            float a0;
            float a1;
            float a2;
            float a3;
            float a4;
            float a5;
            float a6;
            float a7;
            float a8;
            float a9;
            float a10;
            float a11;
            float a12;
            float a13;
            float a14;
            float a15;
            int qb = rb + (k0 >> 8) * 176;
            int sub = (k0 >> 5) & 7;
            int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w, qb + 4, sub);
            float sc = w.getHalfFloat(qb).getFloat32() * (sm >> 8);
            float mn = w.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
            int qn = qb + 48 + (sub >> 1) * 32 + kh;
            int qh = qb + 16 + kh;
            int sh = (sub & 1) << 2;
            a0 =
                    (float)
                                            ((((w.get(qn + 0) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 0) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a1 =
                    (float)
                                            ((((w.get(qn + 1) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 1) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a2 =
                    (float)
                                            ((((w.get(qn + 2) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 2) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a3 =
                    (float)
                                            ((((w.get(qn + 3) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 3) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a4 =
                    (float)
                                            ((((w.get(qn + 4) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 4) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a5 =
                    (float)
                                            ((((w.get(qn + 5) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 5) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a6 =
                    (float)
                                            ((((w.get(qn + 6) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 6) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a7 =
                    (float)
                                            ((((w.get(qn + 7) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 7) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a8 =
                    (float)
                                            ((((w.get(qn + 8) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 8) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a9 =
                    (float)
                                            ((((w.get(qn + 9) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 9) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a10 =
                    (float)
                                            ((((w.get(qn + 10) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 10) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a11 =
                    (float)
                                            ((((w.get(qn + 11) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 11) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a12 =
                    (float)
                                            ((((w.get(qn + 12) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 12) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a13 =
                    (float)
                                            ((((w.get(qn + 13) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 13) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a14 =
                    (float)
                                            ((((w.get(qn + 14) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 14) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            a15 =
                    (float)
                                            ((((w.get(qn + 15) & 0xFF) >> sh) & 15)
                                                    | ((((w.get(qh + 15) & 0xFF) >> sub) & 1) << 4))
                                    * sc
                            - mn;
            float y0 = inputBatch.get(xi + 0);
            float y1 = inputBatch.get(xi + 1);
            float y2 = inputBatch.get(xi + 2);
            float y3 = inputBatch.get(xi + 3);
            float y4 = inputBatch.get(xi + 4);
            float y5 = inputBatch.get(xi + 5);
            float y6 = inputBatch.get(xi + 6);
            float y7 = inputBatch.get(xi + 7);
            float y8 = inputBatch.get(xi + 8);
            float y9 = inputBatch.get(xi + 9);
            float y10 = inputBatch.get(xi + 10);
            float y11 = inputBatch.get(xi + 11);
            float y12 = inputBatch.get(xi + 12);
            float y13 = inputBatch.get(xi + 13);
            float y14 = inputBatch.get(xi + 14);
            float y15 = inputBatch.get(xi + 15);
            context.localBarrier();
            sa[aBase + 0] = new HalfFloat(a0);
            sa[aBase + 8] = new HalfFloat(a1);
            sa[aBase + 16] = new HalfFloat(a2);
            sa[aBase + 24] = new HalfFloat(a3);
            sa[aBase + 32] = new HalfFloat(a4);
            sa[aBase + 40] = new HalfFloat(a5);
            sa[aBase + 48] = new HalfFloat(a6);
            sa[aBase + 56] = new HalfFloat(a7);
            sa[aBase + 512] = new HalfFloat(a8);
            sa[aBase + 520] = new HalfFloat(a9);
            sa[aBase + 528] = new HalfFloat(a10);
            sa[aBase + 536] = new HalfFloat(a11);
            sa[aBase + 544] = new HalfFloat(a12);
            sa[aBase + 552] = new HalfFloat(a13);
            sa[aBase + 560] = new HalfFloat(a14);
            sa[aBase + 568] = new HalfFloat(a15);
            sb[bBase + 0] = y0;
            sb[bBase + 1] = y1;
            sb[bBase + 2] = y2;
            sb[bBase + 3] = y3;
            sb[bBase + 4] = y4;
            sb[bBase + 5] = y5;
            sb[bBase + 6] = y6;
            sb[bBase + 7] = y7;
            sb[bBase + 512] = y8;
            sb[bBase + 513] = y9;
            sb[bBase + 514] = y10;
            sb[bBase + 515] = y11;
            sb[bBase + 516] = y12;
            sb[bBase + 517] = y13;
            sb[bBase + 518] = y14;
            sb[bBase + 519] = y15;
            context.localBarrier();
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (((sg >> 1) << 5) < batch) {
                for (int ik = 0; ik < 4; ik++) {
                    int pa = ca + (ik << 9);
                    int pb = cb + (ik << 9);
                    Matrix8x8Half m0 = context.simdgroupMatrixLoad(sa, pa + 0, 8);
                    Matrix8x8Half m1 = context.simdgroupMatrixLoad(sa, pa + 64, 8);
                    Matrix8x8Half m2 = context.simdgroupMatrixLoad(sa, pa + 128, 8);
                    Matrix8x8Half m3 = context.simdgroupMatrixLoad(sa, pa + 192, 8);
                    Matrix8x8Float n0 = context.simdgroupMatrixLoad(sb, pb + 0, 8);
                    Matrix8x8Float n1 = context.simdgroupMatrixLoad(sb, pb + 64, 8);
                    Matrix8x8Float n2 = context.simdgroupMatrixLoad(sb, pb + 128, 8);
                    Matrix8x8Float n3 = context.simdgroupMatrixLoad(sb, pb + 192, 8);
                    c0 = context.simdgroupMatrixMultiplyAccumulate(n0, m0, c0);
                    c1 = context.simdgroupMatrixMultiplyAccumulate(n0, m1, c1);
                    c2 = context.simdgroupMatrixMultiplyAccumulate(n0, m2, c2);
                    c3 = context.simdgroupMatrixMultiplyAccumulate(n0, m3, c3);
                    c4 = context.simdgroupMatrixMultiplyAccumulate(n1, m0, c4);
                    c5 = context.simdgroupMatrixMultiplyAccumulate(n1, m1, c5);
                    c6 = context.simdgroupMatrixMultiplyAccumulate(n1, m2, c6);
                    c7 = context.simdgroupMatrixMultiplyAccumulate(n1, m3, c7);
                    c8 = context.simdgroupMatrixMultiplyAccumulate(n2, m0, c8);
                    c9 = context.simdgroupMatrixMultiplyAccumulate(n2, m1, c9);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(n2, m2, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(n2, m3, c11);
                    c12 = context.simdgroupMatrixMultiplyAccumulate(n3, m0, c12);
                    c13 = context.simdgroupMatrixMultiplyAccumulate(n3, m1, c13);
                    c14 = context.simdgroupMatrixMultiplyAccumulate(n3, m2, c14);
                    c15 = context.simdgroupMatrixMultiplyAccumulate(n3, m3, c15);
                }
            }
            xi += 32;
        }
        if (full) {
            context.simdgroupMatrixStore(c0, outputBatch, cbase + 0 * d + 0, d);
            context.simdgroupMatrixStore(c1, outputBatch, cbase + 0 * d + 8, d);
            context.simdgroupMatrixStore(c2, outputBatch, cbase + 0 * d + 16, d);
            context.simdgroupMatrixStore(c3, outputBatch, cbase + 0 * d + 24, d);
            context.simdgroupMatrixStore(c4, outputBatch, cbase + 8 * d + 0, d);
            context.simdgroupMatrixStore(c5, outputBatch, cbase + 8 * d + 8, d);
            context.simdgroupMatrixStore(c6, outputBatch, cbase + 8 * d + 16, d);
            context.simdgroupMatrixStore(c7, outputBatch, cbase + 8 * d + 24, d);
            context.simdgroupMatrixStore(c8, outputBatch, cbase + 16 * d + 0, d);
            context.simdgroupMatrixStore(c9, outputBatch, cbase + 16 * d + 8, d);
            context.simdgroupMatrixStore(c10, outputBatch, cbase + 16 * d + 16, d);
            context.simdgroupMatrixStore(c11, outputBatch, cbase + 16 * d + 24, d);
            context.simdgroupMatrixStore(c12, outputBatch, cbase + 24 * d + 0, d);
            context.simdgroupMatrixStore(c13, outputBatch, cbase + 24 * d + 8, d);
            context.simdgroupMatrixStore(c14, outputBatch, cbase + 24 * d + 16, d);
            context.simdgroupMatrixStore(c15, outputBatch, cbase + 24 * d + 24, d);
        } else {
            int t0 = b0 + ((sg >> 1) << 5);
            int rr = r0 + ((sg & 1) << 5);
            context.localBarrier();
            context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillSimdgroupKernels.batchedGemmQ6_KHalfTiles} over one tile of
     * expert rows: the FP16 weight tiles of {@link #groupedExpertGemmQ6_K}'s twin, for a grid of at
     * least {@code TransformerBatchPrefillSimdgroupKernels.HALF_TILES_MIN_GROUPS} workgroups. Grid:
     * {@code ceil(d / 64) * maxTiles} workgroups of {@code
     * TransformerBatchPrefillSimdgroupKernels.HALF_TILES_THREADS} threads.
     */
    // @formatter:on
    public static void groupedExpertGemmQ6_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int tile = grp / rowTiles;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int first = tiles.get(2 + 3 * tile);
        int batch = tiles.get(3 + 3 * tile);
        int b0 = 0;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = (expert * d + TornadoMath.min(r0 + lr, d - 1)) * (n >> 8) * 210;
        int xi =
                sourceRow(sortedToken, gather, first, TornadoMath.min(b0 + lr, batch - 1)) * n + kh;
        int cbase = (first + b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
        Matrix8x8Float c0 = context.simdgroupMatrixZero();
        Matrix8x8Float c1 = context.simdgroupMatrixZero();
        Matrix8x8Float c2 = context.simdgroupMatrixZero();
        Matrix8x8Float c3 = context.simdgroupMatrixZero();
        Matrix8x8Float c4 = context.simdgroupMatrixZero();
        Matrix8x8Float c5 = context.simdgroupMatrixZero();
        Matrix8x8Float c6 = context.simdgroupMatrixZero();
        Matrix8x8Float c7 = context.simdgroupMatrixZero();
        Matrix8x8Float c8 = context.simdgroupMatrixZero();
        Matrix8x8Float c9 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c12 = context.simdgroupMatrixZero();
        Matrix8x8Float c13 = context.simdgroupMatrixZero();
        Matrix8x8Float c14 = context.simdgroupMatrixZero();
        Matrix8x8Float c15 = context.simdgroupMatrixZero();
        for (int k0 = 0; k0 < n; k0 += 32) {
            float a0;
            float a1;
            float a2;
            float a3;
            float a4;
            float a5;
            float a6;
            float a7;
            float a8;
            float a9;
            float a10;
            float a11;
            float a12;
            float a13;
            float a14;
            float a15;
            int qb = rb + (k0 >> 8) * 210;
            int sub = (k0 >> 5) & 7;
            int sv = w.get(qb + 192 + (sub >> 2) * 8 + (kh >> 4) + (sub & 3) * 2) & 0xFF;
            float sc = w.getHalfFloat(qb + 208).getFloat32() * (sv - ((sv >> 7) << 8));
            int qn = qb + (sub >> 2) * 64 + (sub & 1) * 32 + kh;
            int qh = qb + 128 + (sub >> 2) * 32 + kh;
            int sl = ((sub & 3) >> 1) << 2;
            int shh = (sub & 3) << 1;
            a0 =
                    ((float)
                                            ((((w.get(qn + 0) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 0) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a1 =
                    ((float)
                                            ((((w.get(qn + 1) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 1) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a2 =
                    ((float)
                                            ((((w.get(qn + 2) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 2) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a3 =
                    ((float)
                                            ((((w.get(qn + 3) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 3) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a4 =
                    ((float)
                                            ((((w.get(qn + 4) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 4) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a5 =
                    ((float)
                                            ((((w.get(qn + 5) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 5) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a6 =
                    ((float)
                                            ((((w.get(qn + 6) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 6) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a7 =
                    ((float)
                                            ((((w.get(qn + 7) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 7) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a8 =
                    ((float)
                                            ((((w.get(qn + 8) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 8) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a9 =
                    ((float)
                                            ((((w.get(qn + 9) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 9) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a10 =
                    ((float)
                                            ((((w.get(qn + 10) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 10) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a11 =
                    ((float)
                                            ((((w.get(qn + 11) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 11) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a12 =
                    ((float)
                                            ((((w.get(qn + 12) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 12) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a13 =
                    ((float)
                                            ((((w.get(qn + 13) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 13) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a14 =
                    ((float)
                                            ((((w.get(qn + 14) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 14) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            a15 =
                    ((float)
                                            ((((w.get(qn + 15) & 0xFF) >> sl) & 15)
                                                    | ((((w.get(qh + 15) & 0xFF) >> shh) & 3) << 4))
                                    - 32.0f)
                            * sc;
            float y0 = inputBatch.get(xi + 0);
            float y1 = inputBatch.get(xi + 1);
            float y2 = inputBatch.get(xi + 2);
            float y3 = inputBatch.get(xi + 3);
            float y4 = inputBatch.get(xi + 4);
            float y5 = inputBatch.get(xi + 5);
            float y6 = inputBatch.get(xi + 6);
            float y7 = inputBatch.get(xi + 7);
            float y8 = inputBatch.get(xi + 8);
            float y9 = inputBatch.get(xi + 9);
            float y10 = inputBatch.get(xi + 10);
            float y11 = inputBatch.get(xi + 11);
            float y12 = inputBatch.get(xi + 12);
            float y13 = inputBatch.get(xi + 13);
            float y14 = inputBatch.get(xi + 14);
            float y15 = inputBatch.get(xi + 15);
            context.localBarrier();
            sa[aBase + 0] = new HalfFloat(a0);
            sa[aBase + 8] = new HalfFloat(a1);
            sa[aBase + 16] = new HalfFloat(a2);
            sa[aBase + 24] = new HalfFloat(a3);
            sa[aBase + 32] = new HalfFloat(a4);
            sa[aBase + 40] = new HalfFloat(a5);
            sa[aBase + 48] = new HalfFloat(a6);
            sa[aBase + 56] = new HalfFloat(a7);
            sa[aBase + 512] = new HalfFloat(a8);
            sa[aBase + 520] = new HalfFloat(a9);
            sa[aBase + 528] = new HalfFloat(a10);
            sa[aBase + 536] = new HalfFloat(a11);
            sa[aBase + 544] = new HalfFloat(a12);
            sa[aBase + 552] = new HalfFloat(a13);
            sa[aBase + 560] = new HalfFloat(a14);
            sa[aBase + 568] = new HalfFloat(a15);
            sb[bBase + 0] = y0;
            sb[bBase + 1] = y1;
            sb[bBase + 2] = y2;
            sb[bBase + 3] = y3;
            sb[bBase + 4] = y4;
            sb[bBase + 5] = y5;
            sb[bBase + 6] = y6;
            sb[bBase + 7] = y7;
            sb[bBase + 512] = y8;
            sb[bBase + 513] = y9;
            sb[bBase + 514] = y10;
            sb[bBase + 515] = y11;
            sb[bBase + 516] = y12;
            sb[bBase + 517] = y13;
            sb[bBase + 518] = y14;
            sb[bBase + 519] = y15;
            context.localBarrier();
            // A SIMD group whose rows all lie past the tile's multiplies nothing it stores.
            if (((sg >> 1) << 5) < batch) {
                for (int ik = 0; ik < 4; ik++) {
                    int pa = ca + (ik << 9);
                    int pb = cb + (ik << 9);
                    Matrix8x8Half m0 = context.simdgroupMatrixLoad(sa, pa + 0, 8);
                    Matrix8x8Half m1 = context.simdgroupMatrixLoad(sa, pa + 64, 8);
                    Matrix8x8Half m2 = context.simdgroupMatrixLoad(sa, pa + 128, 8);
                    Matrix8x8Half m3 = context.simdgroupMatrixLoad(sa, pa + 192, 8);
                    Matrix8x8Float n0 = context.simdgroupMatrixLoad(sb, pb + 0, 8);
                    Matrix8x8Float n1 = context.simdgroupMatrixLoad(sb, pb + 64, 8);
                    Matrix8x8Float n2 = context.simdgroupMatrixLoad(sb, pb + 128, 8);
                    Matrix8x8Float n3 = context.simdgroupMatrixLoad(sb, pb + 192, 8);
                    c0 = context.simdgroupMatrixMultiplyAccumulate(n0, m0, c0);
                    c1 = context.simdgroupMatrixMultiplyAccumulate(n0, m1, c1);
                    c2 = context.simdgroupMatrixMultiplyAccumulate(n0, m2, c2);
                    c3 = context.simdgroupMatrixMultiplyAccumulate(n0, m3, c3);
                    c4 = context.simdgroupMatrixMultiplyAccumulate(n1, m0, c4);
                    c5 = context.simdgroupMatrixMultiplyAccumulate(n1, m1, c5);
                    c6 = context.simdgroupMatrixMultiplyAccumulate(n1, m2, c6);
                    c7 = context.simdgroupMatrixMultiplyAccumulate(n1, m3, c7);
                    c8 = context.simdgroupMatrixMultiplyAccumulate(n2, m0, c8);
                    c9 = context.simdgroupMatrixMultiplyAccumulate(n2, m1, c9);
                    c10 = context.simdgroupMatrixMultiplyAccumulate(n2, m2, c10);
                    c11 = context.simdgroupMatrixMultiplyAccumulate(n2, m3, c11);
                    c12 = context.simdgroupMatrixMultiplyAccumulate(n3, m0, c12);
                    c13 = context.simdgroupMatrixMultiplyAccumulate(n3, m1, c13);
                    c14 = context.simdgroupMatrixMultiplyAccumulate(n3, m2, c14);
                    c15 = context.simdgroupMatrixMultiplyAccumulate(n3, m3, c15);
                }
            }
            xi += 32;
        }
        if (full) {
            context.simdgroupMatrixStore(c0, outputBatch, cbase + 0 * d + 0, d);
            context.simdgroupMatrixStore(c1, outputBatch, cbase + 0 * d + 8, d);
            context.simdgroupMatrixStore(c2, outputBatch, cbase + 0 * d + 16, d);
            context.simdgroupMatrixStore(c3, outputBatch, cbase + 0 * d + 24, d);
            context.simdgroupMatrixStore(c4, outputBatch, cbase + 8 * d + 0, d);
            context.simdgroupMatrixStore(c5, outputBatch, cbase + 8 * d + 8, d);
            context.simdgroupMatrixStore(c6, outputBatch, cbase + 8 * d + 16, d);
            context.simdgroupMatrixStore(c7, outputBatch, cbase + 8 * d + 24, d);
            context.simdgroupMatrixStore(c8, outputBatch, cbase + 16 * d + 0, d);
            context.simdgroupMatrixStore(c9, outputBatch, cbase + 16 * d + 8, d);
            context.simdgroupMatrixStore(c10, outputBatch, cbase + 16 * d + 16, d);
            context.simdgroupMatrixStore(c11, outputBatch, cbase + 16 * d + 24, d);
            context.simdgroupMatrixStore(c12, outputBatch, cbase + 24 * d + 0, d);
            context.simdgroupMatrixStore(c13, outputBatch, cbase + 24 * d + 8, d);
            context.simdgroupMatrixStore(c14, outputBatch, cbase + 24 * d + 16, d);
            context.simdgroupMatrixStore(c15, outputBatch, cbase + 24 * d + 24, d);
        } else {
            int t0 = b0 + ((sg >> 1) << 5);
            int rr = r0 + ((sg & 1) << 5);
            context.localBarrier();
            context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set((first + t) * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }
}
