package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;
import uk.ac.manchester.tornado.api.types.vectors.Float4;

// @formatter:off
/**
 * Grouped expert GEMMs on SIMD-group matrices (Metal), for a batched mixture-of-experts prefill:
 * the dense {@code TransformerBatchPrefillSimdgroupKernels.batchedGemm*} K-quant kernels run once
 * per tile of one expert's assignments, so a chunk reads each selected expert's weights once per
 * tile instead of once per token. Each kernel is its dense twin's staging and arithmetic with three
 * changes: the row range comes from the tile table, the weight rows are offset to the tile's
 * expert, and the input rows may be gathered through the sorted assignment order. An output row
 * depends on its own input row only, so a token's result does not depend on which tile or chunk it
 * fell in.
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
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
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
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
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
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
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
}
