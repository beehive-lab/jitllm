package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Half;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Half4;

// @formatter:off
/**
 * Batched-prefill projections on Apple SIMD-group matrices ({@code simdgroup_float8x8} and {@code
 * simdgroup_half8x8}): the same products as the tiled kernels in {@link
 * TransformerBatchPrefillKernels}, {@code out[b, r] = sum_k x[b, k] * w[r, k]}, with the
 * multiply-accumulate done by {@code simdgroup_multiply_accumulate} on 8x8 fragments instead of
 * scalar FMAs.
 *
 * <p>A workgroup of 256 threads (eight SIMD groups in a 2x4 layout) computes a 64-token tile of the
 * output, 64 weight rows wide (32 for the gate/up kernels, which compute two projections). Per step
 * a 32-wide slice of the activations and of the weights ({@code half} for FP16, dequantized {@code
 * float} for Q8_0) is staged in threadgroup memory, rows padded to {@value #LDS} elements;
 * activations and FP16 weights are read four columns at a time ({@code getFloat4} / {@code
 * getHalf4}), so row lengths must be multiples of 4. Each SIMD group multiplies a 32-token block of
 * its share of the rows, reading weight fragments with transposed loads because weight rows hold
 * the contraction dimension contiguously. The next slice is read into registers while the current
 * one is multiplied and then stored into the second of two staging buffers, so a slice costs one
 * barrier; the contraction length must therefore be a multiple of 64. Each thread stages a Q8_0
 * row's consecutive columns so that it reads the block scale once. Accumulation is in {@code
 * float}. Whole tiles are stored straight to device memory (the residual kernels start from the
 * output instead of zero); tiles that cross the end of the batch or of the rows go one 8x8 fragment
 * at a time through the staging buffers so that the elements past the ends are skipped.
 */
// @formatter:on
public final class TransformerBatchPrefillSimdgroupKernels {

    /** Threadgroup row stride of a staged 32-wide slice, padded against bank conflicts. */
    public static final int LDS = 40;

    /** Threads per workgroup: eight SIMD groups. */
    public static final int THREADS = 256;

    /** Threads per workgroup of the {@code ...HalfTiles} GEMMs. */
    public static final int HALF_TILES_THREADS = 128;

    private TransformerBatchPrefillSimdgroupKernels() {}

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQKVFP16} on SIMD-group matrices: Q, K and V
     * rows taken as one {@code qDim + 2 * kvDim}-row matrix whose 64-row tiles each lie in one
     * projection (the caller checks both sizes are multiples of 64). {@code dim} must be a multiple
     * of 64. Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQKVFP16(
            KernelContext context,
            HalfFloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        Half4 hx00 = x.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = x.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        if (which == 0) {
            Half4 hw00 = wq.getHalf4((local0 + vr + 0) * dim + 0 + vc);
            pw0_0 = hw00.getX().getFloat32();
            pw0_1 = hw00.getY().getFloat32();
            pw0_2 = hw00.getZ().getFloat32();
            pw0_3 = hw00.getW().getFloat32();
            Half4 hw01 = wq.getHalf4((local0 + vr + 32) * dim + 0 + vc);
            pw1_0 = hw01.getX().getFloat32();
            pw1_1 = hw01.getY().getFloat32();
            pw1_2 = hw01.getZ().getFloat32();
            pw1_3 = hw01.getW().getFloat32();
        } else if (which == 1) {
            Half4 hw00 = wk.getHalf4((local0 + vr + 0) * dim + 0 + vc);
            pw0_0 = hw00.getX().getFloat32();
            pw0_1 = hw00.getY().getFloat32();
            pw0_2 = hw00.getZ().getFloat32();
            pw0_3 = hw00.getW().getFloat32();
            Half4 hw01 = wk.getHalf4((local0 + vr + 32) * dim + 0 + vc);
            pw1_0 = hw01.getX().getFloat32();
            pw1_1 = hw01.getY().getFloat32();
            pw1_2 = hw01.getZ().getFloat32();
            pw1_3 = hw01.getW().getFloat32();
        } else {
            Half4 hw00 = wv.getHalf4((local0 + vr + 0) * dim + 0 + vc);
            pw0_0 = hw00.getX().getFloat32();
            pw0_1 = hw00.getY().getFloat32();
            pw0_2 = hw00.getZ().getFloat32();
            pw0_3 = hw00.getW().getFloat32();
            Half4 hw01 = wv.getHalf4((local0 + vr + 32) * dim + 0 + vc);
            pw1_0 = hw01.getX().getFloat32();
            pw1_1 = hw01.getY().getFloat32();
            pw1_2 = hw01.getZ().getFloat32();
            pw1_3 = hw01.getW().getFloat32();
        }
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < dim; k0 += 64) {
            Half4 hxk0320 =
                    x.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
            px0_0 = hxk0320.getX().getFloat32();
            px0_1 = hxk0320.getY().getFloat32();
            px0_2 = hxk0320.getZ().getFloat32();
            px0_3 = hxk0320.getW().getFloat32();
            Half4 hxk0321 =
                    x.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
            px1_0 = hxk0321.getX().getFloat32();
            px1_1 = hxk0321.getY().getFloat32();
            px1_2 = hxk0321.getZ().getFloat32();
            px1_3 = hxk0321.getW().getFloat32();
            if (which == 0) {
                Half4 hwk0320 = wq.getHalf4((local0 + vr + 0) * dim + k0 + 32 + vc);
                pw0_0 = hwk0320.getX().getFloat32();
                pw0_1 = hwk0320.getY().getFloat32();
                pw0_2 = hwk0320.getZ().getFloat32();
                pw0_3 = hwk0320.getW().getFloat32();
                Half4 hwk0321 = wq.getHalf4((local0 + vr + 32) * dim + k0 + 32 + vc);
                pw1_0 = hwk0321.getX().getFloat32();
                pw1_1 = hwk0321.getY().getFloat32();
                pw1_2 = hwk0321.getZ().getFloat32();
                pw1_3 = hwk0321.getW().getFloat32();
            } else if (which == 1) {
                Half4 hwk0320 = wk.getHalf4((local0 + vr + 0) * dim + k0 + 32 + vc);
                pw0_0 = hwk0320.getX().getFloat32();
                pw0_1 = hwk0320.getY().getFloat32();
                pw0_2 = hwk0320.getZ().getFloat32();
                pw0_3 = hwk0320.getW().getFloat32();
                Half4 hwk0321 = wk.getHalf4((local0 + vr + 32) * dim + k0 + 32 + vc);
                pw1_0 = hwk0321.getX().getFloat32();
                pw1_1 = hwk0321.getY().getFloat32();
                pw1_2 = hwk0321.getZ().getFloat32();
                pw1_3 = hwk0321.getW().getFloat32();
            } else {
                Half4 hwk0320 = wv.getHalf4((local0 + vr + 0) * dim + k0 + 32 + vc);
                pw0_0 = hwk0320.getX().getFloat32();
                pw0_1 = hwk0320.getY().getFloat32();
                pw0_2 = hwk0320.getZ().getFloat32();
                pw0_3 = hwk0320.getW().getFloat32();
                Half4 hwk0321 = wv.getHalf4((local0 + vr + 32) * dim + k0 + 32 + vc);
                pw1_0 = hwk0321.getX().getFloat32();
                pw1_1 = hwk0321.getY().getFloat32();
                pw1_2 = hwk0321.getZ().getFloat32();
                pw1_3 = hwk0321.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
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
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < dim) {
                Half4 hxk0640 =
                        x.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 64 + vc);
                px0_0 = hxk0640.getX().getFloat32();
                px0_1 = hxk0640.getY().getFloat32();
                px0_2 = hxk0640.getZ().getFloat32();
                px0_3 = hxk0640.getW().getFloat32();
                Half4 hxk0641 =
                        x.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 64 + vc);
                px1_0 = hxk0641.getX().getFloat32();
                px1_1 = hxk0641.getY().getFloat32();
                px1_2 = hxk0641.getZ().getFloat32();
                px1_3 = hxk0641.getW().getFloat32();
                if (which == 0) {
                    Half4 hwk0640 = wq.getHalf4((local0 + vr + 0) * dim + k0 + 64 + vc);
                    pw0_0 = hwk0640.getX().getFloat32();
                    pw0_1 = hwk0640.getY().getFloat32();
                    pw0_2 = hwk0640.getZ().getFloat32();
                    pw0_3 = hwk0640.getW().getFloat32();
                    Half4 hwk0641 = wq.getHalf4((local0 + vr + 32) * dim + k0 + 64 + vc);
                    pw1_0 = hwk0641.getX().getFloat32();
                    pw1_1 = hwk0641.getY().getFloat32();
                    pw1_2 = hwk0641.getZ().getFloat32();
                    pw1_3 = hwk0641.getW().getFloat32();
                } else if (which == 1) {
                    Half4 hwk0640 = wk.getHalf4((local0 + vr + 0) * dim + k0 + 64 + vc);
                    pw0_0 = hwk0640.getX().getFloat32();
                    pw0_1 = hwk0640.getY().getFloat32();
                    pw0_2 = hwk0640.getZ().getFloat32();
                    pw0_3 = hwk0640.getW().getFloat32();
                    Half4 hwk0641 = wk.getHalf4((local0 + vr + 32) * dim + k0 + 64 + vc);
                    pw1_0 = hwk0641.getX().getFloat32();
                    pw1_1 = hwk0641.getY().getFloat32();
                    pw1_2 = hwk0641.getZ().getFloat32();
                    pw1_3 = hwk0641.getW().getFloat32();
                } else {
                    Half4 hwk0640 = wv.getHalf4((local0 + vr + 0) * dim + k0 + 64 + vc);
                    pw0_0 = hwk0640.getX().getFloat32();
                    pw0_1 = hwk0640.getY().getFloat32();
                    pw0_2 = hwk0640.getZ().getFloat32();
                    pw0_3 = hwk0640.getW().getFloat32();
                    Half4 hwk0641 = wv.getHalf4((local0 + vr + 32) * dim + k0 + 64 + vc);
                    pw1_0 = hwk0641.getX().getFloat32();
                    pw1_1 = hwk0641.getY().getFloat32();
                    pw1_2 = hwk0641.getZ().getFloat32();
                    pw1_3 = hwk0641.getW().getFloat32();
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            if (k0 + 64 < dim) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmFP16WithResidual} on SIMD-group matrices:
     * {@code out[b, r] += sum_k x[b, k] * w[r, k]} with FP16 weights. {@code n} must be a multiple
     * of 64. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmFP16WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            HalfFloatArray w,
            int n,
            int d,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        Half4 hw00 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + 0 + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < n; k0 += 64) {
            Float4 fxk0320 =
                    inputBatch.getFloat4(
                            TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
            px0_0 = fxk0320.getX();
            px0_1 = fxk0320.getY();
            px0_2 = fxk0320.getZ();
            px0_3 = fxk0320.getW();
            Float4 fxk0321 =
                    inputBatch.getFloat4(
                            TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
            px1_0 = fxk0321.getX();
            px1_1 = fxk0321.getY();
            px1_2 = fxk0321.getZ();
            px1_3 = fxk0321.getW();
            Half4 hwk0320 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 32 + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
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
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < n) {
                Float4 fxk0640 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 64 + vc);
                px0_0 = fxk0640.getX();
                px0_1 = fxk0640.getY();
                px0_2 = fxk0640.getZ();
                px0_3 = fxk0640.getW();
                Float4 fxk0641 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 64 + vc);
                px1_0 = fxk0641.getX();
                px1_1 = fxk0641.getY();
                px1_2 = fxk0641.getZ();
                px1_3 = fxk0641.getW();
                Half4 hwk0640 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 64 + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            if (k0 + 64 < n) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmRmsNormFFNGateUpFP16} on SIMD-group
     * matrices: the RMS scaling is applied as the activations are staged, gate and up accumulate
     * from the same staged activations, and {@code SiLU(gate) * up} is written per element. {@code
     * dim} must be a multiple of 64. Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)}
     * workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpFP16(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        Half4 hw00 = w1.getHalf4((TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 =
                w3.getHalf4((TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1)) * dim + 0 + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < dim; k0 += 64) {
            Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
            Float4 fxk0320 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
            px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
            px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
            px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
            px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
            Float4 fxk0321 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
            px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
            px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
            px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
            px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
            Half4 hwk0320 =
                    w1.getHalf4((TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 =
                    w3.getHalf4(
                            (TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1)) * dim
                                    + k0
                                    + 32
                                    + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Half uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < dim) {
                Float4 rvk064 = rmsWeights.getFloat4(k0 + 64 + vc);
                Float4 fxk0640 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 64 + vc);
                px0_0 = rvk064.getX() * xsc0 * fxk0640.getX();
                px0_1 = rvk064.getY() * xsc0 * fxk0640.getY();
                px0_2 = rvk064.getZ() * xsc0 * fxk0640.getZ();
                px0_3 = rvk064.getW() * xsc0 * fxk0640.getW();
                Float4 fxk0641 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 64 + vc);
                px1_0 = rvk064.getX() * xsc1 * fxk0641.getX();
                px1_1 = rvk064.getY() * xsc1 * fxk0641.getY();
                px1_2 = rvk064.getZ() * xsc1 * fxk0641.getZ();
                px1_3 = rvk064.getW() * xsc1 * fxk0641.getW();
                Half4 hwk0640 =
                        w1.getHalf4(
                                (TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 =
                        w3.getHalf4(
                                (TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1)) * dim
                                        + k0
                                        + 64
                                        + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half gw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Half uw0 =
                        context.simdgroupMatrixLoadTransposed(
                                ws1, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            if (k0 + 64 < dim) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, xs1, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, xs1, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, xs1, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, xs1, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * xs1[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQKVQ8} on SIMD-group matrices; Q8_0 weights
     * are dequantized to {@code float} as they are staged. {@code dim} must be a multiple of 64.
     * Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value #THREADS}
     * threads.
     */
    // @formatter:on
    public static void batchedGemmQKVQ8(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
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
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
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
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        if (which == 0) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wq.getHalfFloat(qb).getFloat32();
            pw0 = (float) wq.get(qb + 2 + wk0 + 0);
            pw1 = (float) wq.get(qb + 2 + wk0 + 1);
            pw2 = (float) wq.get(qb + 2 + wk0 + 2);
            pw3 = (float) wq.get(qb + 2 + wk0 + 3);
            pw4 = (float) wq.get(qb + 2 + wk0 + 4);
            pw5 = (float) wq.get(qb + 2 + wk0 + 5);
            pw6 = (float) wq.get(qb + 2 + wk0 + 6);
            pw7 = (float) wq.get(qb + 2 + wk0 + 7);
        } else if (which == 1) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wk.getHalfFloat(qb).getFloat32();
            pw0 = (float) wk.get(qb + 2 + wk0 + 0);
            pw1 = (float) wk.get(qb + 2 + wk0 + 1);
            pw2 = (float) wk.get(qb + 2 + wk0 + 2);
            pw3 = (float) wk.get(qb + 2 + wk0 + 3);
            pw4 = (float) wk.get(qb + 2 + wk0 + 4);
            pw5 = (float) wk.get(qb + 2 + wk0 + 5);
            pw6 = (float) wk.get(qb + 2 + wk0 + 6);
            pw7 = (float) wk.get(qb + 2 + wk0 + 7);
        } else {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wv.getHalfFloat(qb).getFloat32();
            pw0 = (float) wv.get(qb + 2 + wk0 + 0);
            pw1 = (float) wv.get(qb + 2 + wk0 + 1);
            pw2 = (float) wv.get(qb + 2 + wk0 + 2);
            pw3 = (float) wv.get(qb + 2 + wk0 + 3);
            pw4 = (float) wv.get(qb + 2 + wk0 + 4);
            pw5 = (float) wv.get(qb + 2 + wk0 + 5);
            pw6 = (float) wv.get(qb + 2 + wk0 + 6);
            pw7 = (float) wv.get(qb + 2 + wk0 + 7);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                if (which == 0) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wq.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wq.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wq.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wq.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wq.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wq.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wq.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wq.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wq.get(qb + 2 + wk0 + 7);
                } else if (which == 1) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wk.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wk.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wk.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wk.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wk.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wk.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wk.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wk.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wk.get(qb + 2 + wk0 + 7);
                } else {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wv.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wv.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wv.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wv.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wv.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wv.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wv.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wv.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wv.get(qb + 2 + wk0 + 7);
                }
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
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQ8WithResidual} on SIMD-group matrices; Q8_0
     * weights are dequantized to {@code float} as they are staged. {@code n} must be a multiple of
     * 64. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ8WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 34;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 34;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmRmsNormFFNGateUpQ8} on SIMD-group matrices;
     * Q8_0 weights are dequantized to {@code float} as they are staged. {@code dim} must be a
     * multiple of 64. Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ8(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = w1.getHalfFloat(qb).getFloat32();
            pw0 = (float) w1.get(qb + 2 + wk0 + 0);
            pw1 = (float) w1.get(qb + 2 + wk0 + 1);
            pw2 = (float) w1.get(qb + 2 + wk0 + 2);
            pw3 = (float) w1.get(qb + 2 + wk0 + 3);
            pw4 = (float) w1.get(qb + 2 + wk0 + 4);
            pw5 = (float) w1.get(qb + 2 + wk0 + 5);
            pw6 = (float) w1.get(qb + 2 + wk0 + 6);
            pw7 = (float) w1.get(qb + 2 + wk0 + 7);
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = w3.getHalfFloat(qb).getFloat32();
            pw0 = (float) w3.get(qb + 2 + wk0 + 0);
            pw1 = (float) w3.get(qb + 2 + wk0 + 1);
            pw2 = (float) w3.get(qb + 2 + wk0 + 2);
            pw3 = (float) w3.get(qb + 2 + wk0 + 3);
            pw4 = (float) w3.get(qb + 2 + wk0 + 4);
            pw5 = (float) w3.get(qb + 2 + wk0 + 5);
            pw6 = (float) w3.get(qb + 2 + wk0 + 6);
            pw7 = (float) w3.get(qb + 2 + wk0 + 7);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = w1.getHalfFloat(qb).getFloat32();
                    pw0 = (float) w1.get(qb + 2 + wk0 + 0);
                    pw1 = (float) w1.get(qb + 2 + wk0 + 1);
                    pw2 = (float) w1.get(qb + 2 + wk0 + 2);
                    pw3 = (float) w1.get(qb + 2 + wk0 + 3);
                    pw4 = (float) w1.get(qb + 2 + wk0 + 4);
                    pw5 = (float) w1.get(qb + 2 + wk0 + 5);
                    pw6 = (float) w1.get(qb + 2 + wk0 + 6);
                    pw7 = (float) w1.get(qb + 2 + wk0 + 7);
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = w3.getHalfFloat(qb).getFloat32();
                    pw0 = (float) w3.get(qb + 2 + wk0 + 0);
                    pw1 = (float) w3.get(qb + 2 + wk0 + 1);
                    pw2 = (float) w3.get(qb + 2 + wk0 + 2);
                    pw3 = (float) w3.get(qb + 2 + wk0 + 3);
                    pw4 = (float) w3.get(qb + 2 + wk0 + 4);
                    pw5 = (float) w3.get(qb + 2 + wk0 + 5);
                    pw6 = (float) w3.get(qb + 2 + wk0 + 6);
                    pw7 = (float) w3.get(qb + 2 + wk0 + 7);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQKVFP16} with Q, K and V as consecutive rows of one weight matrix (Phi-3's
     * {@code wqkv}) and FP32 activations. {@code dim} must be a multiple of 64. Grid: {@code ((qDim
     * + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQKVFusedFP16(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wqkv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        Half4 hw00 = wqkv.getHalf4((r0 + vr + 0) * dim + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 = wqkv.getHalf4((r0 + vr + 32) * dim + 0 + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < dim; k0 += 64) {
            Float4 fxk0320 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
            px0_0 = fxk0320.getX();
            px0_1 = fxk0320.getY();
            px0_2 = fxk0320.getZ();
            px0_3 = fxk0320.getW();
            Float4 fxk0321 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
            px1_0 = fxk0321.getX();
            px1_1 = fxk0321.getY();
            px1_2 = fxk0321.getZ();
            px1_3 = fxk0321.getW();
            Half4 hwk0320 = wqkv.getHalf4((r0 + vr + 0) * dim + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 = wqkv.getHalf4((r0 + vr + 32) * dim + k0 + 32 + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
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
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < dim) {
                Float4 fxk0640 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 64 + vc);
                px0_0 = fxk0640.getX();
                px0_1 = fxk0640.getY();
                px0_2 = fxk0640.getZ();
                px0_3 = fxk0640.getW();
                Float4 fxk0641 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 64 + vc);
                px1_0 = fxk0641.getX();
                px1_1 = fxk0641.getY();
                px1_2 = fxk0641.getZ();
                px1_3 = fxk0641.getW();
                Half4 hwk0640 = wqkv.getHalf4((r0 + vr + 0) * dim + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 = wqkv.getHalf4((r0 + vr + 32) * dim + k0 + 64 + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            if (k0 + 64 < dim) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpFP16} with the gate and the up rows in one {@code [2 *
     * hiddenDim x dim]} weight matrix, gate rows first (Phi-3's {@code wUp}). Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpFusedFP16(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            HalfFloatArray wUp,
            int dim,
            int hiddenDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int vr = tid >> 3;
        int vc = (tid & 7) << 2;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        Half4 hw00 = wUp.getHalf4((TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 =
                wUp.getHalf4(
                        (hiddenDim + TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1)) * dim
                                + 0
                                + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < dim; k0 += 64) {
            Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
            Float4 fxk0320 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
            px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
            px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
            px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
            px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
            Float4 fxk0321 =
                    x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
            px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
            px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
            px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
            px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
            Half4 hwk0320 =
                    wUp.getHalf4(
                            (TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 =
                    wUp.getHalf4(
                            (hiddenDim + TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1)) * dim
                                    + k0
                                    + 32
                                    + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Half uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < dim) {
                Float4 rvk064 = rmsWeights.getFloat4(k0 + 64 + vc);
                Float4 fxk0640 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 64 + vc);
                px0_0 = rvk064.getX() * xsc0 * fxk0640.getX();
                px0_1 = rvk064.getY() * xsc0 * fxk0640.getY();
                px0_2 = rvk064.getZ() * xsc0 * fxk0640.getZ();
                px0_3 = rvk064.getW() * xsc0 * fxk0640.getW();
                Float4 fxk0641 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 64 + vc);
                px1_0 = rvk064.getX() * xsc1 * fxk0641.getX();
                px1_1 = rvk064.getY() * xsc1 * fxk0641.getY();
                px1_2 = rvk064.getZ() * xsc1 * fxk0641.getZ();
                px1_3 = rvk064.getW() * xsc1 * fxk0641.getW();
                Half4 hwk0640 =
                        wUp.getHalf4(
                                (TornadoMath.min(r0 + vr + 0, hiddenDim - 1)) * dim + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 =
                        wUp.getHalf4(
                                (hiddenDim + TornadoMath.min(r0 + vr + 32 - 32, hiddenDim - 1))
                                                * dim
                                        + k0
                                        + 64
                                        + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half gw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Half uw0 =
                        context.simdgroupMatrixLoadTransposed(
                                ws1, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            if (k0 + 64 < dim) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, xs1, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, xs1, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, xs1, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, xs1, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * xs1[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQKVQ8} with Q, K and V as consecutive rows of one weight matrix (Phi-3's
     * {@code wqkv}). Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of
     * {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQKVFusedQ8(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wqkv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
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
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
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
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((r0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
        psc = wqkv.getHalfFloat(qb).getFloat32();
        pw0 = (float) wqkv.get(qb + 2 + wk0 + 0);
        pw1 = (float) wqkv.get(qb + 2 + wk0 + 1);
        pw2 = (float) wqkv.get(qb + 2 + wk0 + 2);
        pw3 = (float) wqkv.get(qb + 2 + wk0 + 3);
        pw4 = (float) wqkv.get(qb + 2 + wk0 + 4);
        pw5 = (float) wqkv.get(qb + 2 + wk0 + 5);
        pw6 = (float) wqkv.get(qb + 2 + wk0 + 6);
        pw7 = (float) wqkv.get(qb + 2 + wk0 + 7);
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((r0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                psc = wqkv.getHalfFloat(qb).getFloat32();
                pw0 = (float) wqkv.get(qb + 2 + wk0 + 0);
                pw1 = (float) wqkv.get(qb + 2 + wk0 + 1);
                pw2 = (float) wqkv.get(qb + 2 + wk0 + 2);
                pw3 = (float) wqkv.get(qb + 2 + wk0 + 3);
                pw4 = (float) wqkv.get(qb + 2 + wk0 + 4);
                pw5 = (float) wqkv.get(qb + 2 + wk0 + 5);
                pw6 = (float) wqkv.get(qb + 2 + wk0 + 6);
                pw7 = (float) wqkv.get(qb + 2 + wk0 + 7);
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
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ8} with the gate and the up rows in one weight matrix,
     * gate rows first (Phi-3's {@code wUp}). Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)}
     * workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpFusedQ8(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray wUp,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wUp.getHalfFloat(qb).getFloat32();
            pw0 = (float) wUp.get(qb + 2 + wk0 + 0);
            pw1 = (float) wUp.get(qb + 2 + wk0 + 1);
            pw2 = (float) wUp.get(qb + 2 + wk0 + 2);
            pw3 = (float) wUp.get(qb + 2 + wk0 + 3);
            pw4 = (float) wUp.get(qb + 2 + wk0 + 4);
            pw5 = (float) wUp.get(qb + 2 + wk0 + 5);
            pw6 = (float) wUp.get(qb + 2 + wk0 + 6);
            pw7 = (float) wUp.get(qb + 2 + wk0 + 7);
        } else {
            qb =
                    ((hiddenDim + TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5)
                                    + ((0) >> 5))
                            * 34;
            psc = wUp.getHalfFloat(qb).getFloat32();
            pw0 = (float) wUp.get(qb + 2 + wk0 + 0);
            pw1 = (float) wUp.get(qb + 2 + wk0 + 1);
            pw2 = (float) wUp.get(qb + 2 + wk0 + 2);
            pw3 = (float) wUp.get(qb + 2 + wk0 + 3);
            pw4 = (float) wUp.get(qb + 2 + wk0 + 4);
            pw5 = (float) wUp.get(qb + 2 + wk0 + 5);
            pw6 = (float) wUp.get(qb + 2 + wk0 + 6);
            pw7 = (float) wUp.get(qb + 2 + wk0 + 7);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = wUp.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wUp.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wUp.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wUp.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wUp.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wUp.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wUp.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wUp.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wUp.get(qb + 2 + wk0 + 7);
                } else {
                    qb =
                            ((hiddenDim + TornadoMath.min(r0 + wrow - 32, hiddenDim - 1))
                                                    * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = wUp.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wUp.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wUp.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wUp.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wUp.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wUp.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wUp.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wUp.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wUp.get(qb + 2 + wk0 + 7);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmFP16WithResidual} with the product scaled before it is added: {@code
     * out[b, r] += residualScale * sum_k x[b, k] * w[r, k]} (Granite's residual multiplier). Grid:
     * {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmFP16WithScaledResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            HalfFloatArray w,
            int n,
            int d,
            int batch,
            float residualScale) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        Half4 hw00 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + 0 + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < n; k0 += 64) {
            Float4 fxk0320 =
                    inputBatch.getFloat4(
                            TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
            px0_0 = fxk0320.getX();
            px0_1 = fxk0320.getY();
            px0_2 = fxk0320.getZ();
            px0_3 = fxk0320.getW();
            Float4 fxk0321 =
                    inputBatch.getFloat4(
                            TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
            px1_0 = fxk0321.getX();
            px1_1 = fxk0321.getY();
            px1_2 = fxk0321.getZ();
            px1_3 = fxk0321.getW();
            Half4 hwk0320 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 32 + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
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
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < n) {
                Float4 fxk0640 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 64 + vc);
                px0_0 = fxk0640.getX();
                px0_1 = fxk0640.getY();
                px0_2 = fxk0640.getZ();
                px0_3 = fxk0640.getW();
                Float4 fxk0641 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 64 + vc);
                px1_0 = fxk0641.getX();
                px1_1 = fxk0641.getY();
                px1_2 = fxk0641.getZ();
                px1_3 = fxk0641.getW();
                Half4 hwk0640 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 64 + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            if (k0 + 64 < n) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        if (full) {
            if (tid < 64) {
                xs[tid] = (tid % 9 == 0) ? residualScale : 0.0f;
            }
            context.localBarrier();
            Matrix8x8Float rs = context.simdgroupMatrixLoad(xs, 0, 8);
            c00 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c00,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c01,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c10,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c11,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c20,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c21,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c30,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c31,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ8WithResidual} with the product scaled by {@code residualScale} before it
     * is added (Granite's residual multiplier). Grid: {@code ceil(d / 64) * ceil(batch / 64)}
     * workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ8WithScaledResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            float residualScale) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 34;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 34;
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
            if (tid < 64) {
                xs[tid] = (tid % 9 == 0) ? residualScale : 0.0f;
            }
            context.localBarrier();
            Matrix8x8Float rs = context.simdgroupMatrixLoad(xs, 0, 8);
            c00 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c00,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c01,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c10,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c11,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c20,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c21,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c30,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d));
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixMultiplyAccumulate(
                            c31,
                            rs,
                            context.simdgroupMatrixLoad(
                                    outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d));
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(
                            t * d + r,
                            outputBatch.get(t * d + r) + residualScale * xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQKVQ8} for Q4_0 weights, dequantized to {@code float} as they are staged.
     * Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value #THREADS}
     * threads.
     */
    // @formatter:on
    public static void batchedGemmQKVQ4_0(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
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
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
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
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        if (which == 0) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 18;
            psc = wq.getHalfFloat(qb).getFloat32();
            qn = qb + 2 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((wq.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
            pw1 = (float) (((wq.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
            pw2 = (float) (((wq.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
            pw3 = (float) (((wq.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
            pw4 = (float) (((wq.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
            pw5 = (float) (((wq.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
            pw6 = (float) (((wq.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
            pw7 = (float) (((wq.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
        } else if (which == 1) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 18;
            psc = wk.getHalfFloat(qb).getFloat32();
            qn = qb + 2 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((wk.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
            pw1 = (float) (((wk.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
            pw2 = (float) (((wk.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
            pw3 = (float) (((wk.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
            pw4 = (float) (((wk.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
            pw5 = (float) (((wk.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
            pw6 = (float) (((wk.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
            pw7 = (float) (((wk.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
        } else {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 18;
            psc = wv.getHalfFloat(qb).getFloat32();
            qn = qb + 2 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((wv.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
            pw1 = (float) (((wv.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
            pw2 = (float) (((wv.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
            pw3 = (float) (((wv.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
            pw4 = (float) (((wv.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
            pw5 = (float) (((wv.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
            pw6 = (float) (((wv.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
            pw7 = (float) (((wv.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                if (which == 0) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 18;
                    psc = wq.getHalfFloat(qb).getFloat32();
                    qn = qb + 2 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((wq.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                    pw1 = (float) (((wq.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                    pw2 = (float) (((wq.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                    pw3 = (float) (((wq.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                    pw4 = (float) (((wq.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                    pw5 = (float) (((wq.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                    pw6 = (float) (((wq.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                    pw7 = (float) (((wq.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
                } else if (which == 1) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 18;
                    psc = wk.getHalfFloat(qb).getFloat32();
                    qn = qb + 2 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((wk.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                    pw1 = (float) (((wk.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                    pw2 = (float) (((wk.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                    pw3 = (float) (((wk.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                    pw4 = (float) (((wk.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                    pw5 = (float) (((wk.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                    pw6 = (float) (((wk.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                    pw7 = (float) (((wk.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
                } else {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 18;
                    psc = wv.getHalfFloat(qb).getFloat32();
                    qn = qb + 2 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((wv.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                    pw1 = (float) (((wv.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                    pw2 = (float) (((wv.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                    pw3 = (float) (((wv.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                    pw4 = (float) (((wv.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                    pw5 = (float) (((wv.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                    pw6 = (float) (((wv.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                    pw7 = (float) (((wv.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
                }
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
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = xs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ8WithResidual} for Q4_0 weights, dequantized to {@code float} as they are
     * staged. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS}
     * threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_0WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 18;
        psc = w.getHalfFloat(qb).getFloat32();
        qn = qb + 2 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
        pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
        pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
        pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
        pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
        pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
        pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
        pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
        pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 18;
                psc = w.getHalfFloat(qb).getFloat32();
                qn = qb + 2 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
                pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@code out[b, r] = sum_k x[b, k] * w[r, k]} on SIMD-group matrices with Q4_0 weights,
     * dequantized to {@code float} as they are staged. {@code n} must be a multiple of 64. Grid:
     * {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_0(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 18;
        psc = w.getHalfFloat(qb).getFloat32();
        qn = qb + 2 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
        pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
        pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
        pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
        pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
        pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
        pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
        pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
        pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 18;
                psc = w.getHalfFloat(qb).getFloat32();
                qn = qb + 2 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
                pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ8} for Q4_0 weights, dequantized to {@code float} as they
     * are staged. Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_0(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 18;
            psc = w1.getHalfFloat(qb).getFloat32();
            qn = qb + 2 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
            pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
            pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
            pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
            pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
            pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
            pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
            pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 18;
            psc = w3.getHalfFloat(qb).getFloat32();
            qn = qb + 2 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
            pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
            pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
            pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
            pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
            pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
            pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
            pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 18;
                    psc = w1.getHalfFloat(qb).getFloat32();
                    qn = qb + 2 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                    pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                    pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                    pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                    pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                    pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                    pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                    pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 18;
                    psc = w3.getHalfFloat(qb).getFloat32();
                    qn = qb + 2 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                    pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                    pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                    pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                    pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                    pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                    pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                    pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0} for Q4_K weights, each 32-wide slice one sub-block of a 256-weight
     * super-block, dequantized to {@code float} as it is staged. {@code n} must be a multiple of
     * 256. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 144;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 144;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0WithResidual} for Q4_K weights; see {@link #batchedGemmQ4_K}. Grid:
     * {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_KWithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 144;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 144;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_0} for Q4_K weights; see {@link #batchedGemmQ4_K}.
     * Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of {@value #THREADS}
     * threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 144;
            qsub = ((0) >> 5) & 7;
            qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, qsub);
            psc = w1.getHalfFloat(qb).getFloat32() * (qsm >> 8);
            pmn = w1.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
            qn = qb + 16 + (qsub >> 1) * 32 + wk0;
            qs = (qsub & 1) << 2;
            pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15);
            pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15);
            pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15);
            pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15);
            pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15);
            pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15);
            pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15);
            pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15);
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 144;
            qsub = ((0) >> 5) & 7;
            qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, qsub);
            psc = w3.getHalfFloat(qb).getFloat32() * (qsm >> 8);
            pmn = w3.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
            qn = qb + 16 + (qsub >> 1) * 32 + wk0;
            qs = (qsub & 1) << 2;
            pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15);
            pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15);
            pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15);
            pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15);
            pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15);
            pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15);
            pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15);
            pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 144;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, qsub);
                    psc = w1.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                    pmn = w1.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                    qn = qb + 16 + (qsub >> 1) * 32 + wk0;
                    qs = (qsub & 1) << 2;
                    pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15);
                    pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15);
                    pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15);
                    pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15);
                    pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15);
                    pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15);
                    pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15);
                    pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15);
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 144;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, qsub);
                    psc = w3.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                    pmn = w3.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                    qn = qb + 16 + (qsub >> 1) * 32 + wk0;
                    qs = (qsub & 1) << 2;
                    pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15);
                    pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15);
                    pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15);
                    pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15);
                    pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15);
                    pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15);
                    pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15);
                    pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0} for Q6_K weights, each 32-wide slice one group of a 256-weight
     * super-block, dequantized to {@code float} as it is staged. {@code n} must be a multiple of
     * 256. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ6_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 210;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 210;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0WithResidual} for Q6_K weights; see {@link #batchedGemmQ6_K}. Grid:
     * {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ6_KWithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 210;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 210;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0} for Q4_1 weights: each 32-wide slice one 20-byte block, {@code d},
     * {@code m} and sixteen bytes of nibbles, dequantized to {@code d * q + m} as it is staged.
     */
    // @formatter:on
    public static void batchedGemmQ4_1(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
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
        float pmn;
        int qb;
        int qn;
        int qs;
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 20;
        psc = w.getHalfFloat(qb).getFloat32();
        pmn = w.getHalfFloat(qb + 2).getFloat32();
        qn = qb + 4 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
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
            ws[wrow * LDS + wk0 + 0] = pw0 * psc + pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc + pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc + pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc + pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc + pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc + pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc + pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc + pmn;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 20;
                psc = w.getHalfFloat(qb).getFloat32();
                pmn = w.getHalfFloat(qb + 2).getFloat32();
                qn = qb + 4 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0WithResidual} for Q4_1 weights: each 32-wide slice one 20-byte block,
     * {@code d}, {@code m} and sixteen bytes of nibbles, dequantized to {@code d * q + m} as it is
     * staged.
     */
    // @formatter:on
    public static void batchedGemmQ4_1WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        float pmn;
        int qb;
        int qn;
        int qs;
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 20;
        psc = w.getHalfFloat(qb).getFloat32();
        pmn = w.getHalfFloat(qb + 2).getFloat32();
        qn = qb + 4 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
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
            ws[wrow * LDS + wk0 + 0] = pw0 * psc + pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc + pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc + pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc + pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc + pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc + pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc + pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc + pmn;
            context.localBarrier();
            if (k0 + 32 < n) {
                Float4 fxk0320 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 20;
                psc = w.getHalfFloat(qb).getFloat32();
                pmn = w.getHalfFloat(qb + 2).getFloat32();
                qn = qb + 4 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_0} for Q4_1 weights: each 32-wide slice one 20-byte
     * block, {@code d}, {@code m} and sixteen bytes of nibbles, dequantized to {@code d * q + m} as
     * it is staged.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_1(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float pmn;
        int qb;
        int qn;
        int qs;
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 20;
            psc = w1.getHalfFloat(qb).getFloat32();
            pmn = w1.getHalfFloat(qb + 2).getFloat32();
            qn = qb + 4 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15);
            pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15);
            pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15);
            pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15);
            pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15);
            pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15);
            pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15);
            pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15);
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 20;
            psc = w3.getHalfFloat(qb).getFloat32();
            pmn = w3.getHalfFloat(qb + 2).getFloat32();
            qn = qb + 4 + (wk0 & 15);
            qs = (wk0 & 16) >> 2;
            pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15);
            pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15);
            pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15);
            pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15);
            pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15);
            pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15);
            pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15);
            pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
            xs[(vr + 0) * LDS + vc + 0] = px0_0;
            xs[(vr + 0) * LDS + vc + 1] = px0_1;
            xs[(vr + 0) * LDS + vc + 2] = px0_2;
            xs[(vr + 0) * LDS + vc + 3] = px0_3;
            xs[(vr + 32) * LDS + vc + 0] = px1_0;
            xs[(vr + 32) * LDS + vc + 1] = px1_1;
            xs[(vr + 32) * LDS + vc + 2] = px1_2;
            xs[(vr + 32) * LDS + vc + 3] = px1_3;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc + pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc + pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc + pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc + pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc + pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc + pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc + pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc + pmn;
            context.localBarrier();
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 20;
                    psc = w1.getHalfFloat(qb).getFloat32();
                    pmn = w1.getHalfFloat(qb + 2).getFloat32();
                    qn = qb + 4 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((w1.get(qn + 0) & 0xFF) >> qs) & 15);
                    pw1 = (float) (((w1.get(qn + 1) & 0xFF) >> qs) & 15);
                    pw2 = (float) (((w1.get(qn + 2) & 0xFF) >> qs) & 15);
                    pw3 = (float) (((w1.get(qn + 3) & 0xFF) >> qs) & 15);
                    pw4 = (float) (((w1.get(qn + 4) & 0xFF) >> qs) & 15);
                    pw5 = (float) (((w1.get(qn + 5) & 0xFF) >> qs) & 15);
                    pw6 = (float) (((w1.get(qn + 6) & 0xFF) >> qs) & 15);
                    pw7 = (float) (((w1.get(qn + 7) & 0xFF) >> qs) & 15);
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 20;
                    psc = w3.getHalfFloat(qb).getFloat32();
                    pmn = w3.getHalfFloat(qb + 2).getFloat32();
                    qn = qb + 4 + (wk0 & 15);
                    qs = (wk0 & 16) >> 2;
                    pw0 = (float) (((w3.get(qn + 0) & 0xFF) >> qs) & 15);
                    pw1 = (float) (((w3.get(qn + 1) & 0xFF) >> qs) & 15);
                    pw2 = (float) (((w3.get(qn + 2) & 0xFF) >> qs) & 15);
                    pw3 = (float) (((w3.get(qn + 3) & 0xFF) >> qs) & 15);
                    pw4 = (float) (((w3.get(qn + 4) & 0xFF) >> qs) & 15);
                    pw5 = (float) (((w3.get(qn + 5) & 0xFF) >> qs) & 15);
                    pw6 = (float) (((w3.get(qn + 6) & 0xFF) >> qs) & 15);
                    pw7 = (float) (((w3.get(qn + 7) & 0xFF) >> qs) & 15);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_K} for Q5_K weights: the Q4_K super-block with a fifth bit per weight
     * in {@code qh}, bit {@code j} for sub-block {@code j}. 176 bytes to 256 weights.
     */
    // @formatter:on
    public static void batchedGemmQ5_K(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 176;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 176;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_KWithResidual} for Q5_K weights: the Q4_K super-block with a fifth bit
     * per weight in {@code qh}, bit {@code j} for sub-block {@code j}. 176 bytes to 256 weights.
     */
    // @formatter:on
    public static void batchedGemmQ5_KWithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
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
        Float4 fx00 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = fx00.getX();
        px0_1 = fx00.getY();
        px0_2 = fx00.getZ();
        px0_3 = fx00.getW();
        Float4 fx01 = inputBatch.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = fx01.getX();
        px1_1 = fx01.getY();
        px1_2 = fx01.getZ();
        px1_3 = fx01.getW();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 176;
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
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = fxk0320.getX();
                px0_1 = fxk0320.getY();
                px0_2 = fxk0320.getZ();
                px0_3 = fxk0320.getW();
                Float4 fxk0321 =
                        inputBatch.getFloat4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = fxk0321.getX();
                px1_1 = fxk0321.getY();
                px1_2 = fxk0321.getZ();
                px1_3 = fxk0321.getW();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 176;
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
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_K} for Q5_K weights: the Q4_K super-block with a fifth
     * bit per weight in {@code qh}, bit {@code j} for sub-block {@code j}. 176 bytes to 256
     * weights.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ5_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 176;
            qsub = ((0) >> 5) & 7;
            qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, qsub);
            psc = w1.getHalfFloat(qb).getFloat32() * (qsm >> 8);
            pmn = w1.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
            qn = qb + 48 + (qsub >> 1) * 32 + wk0;
            qh = qb + 16 + wk0;
            qs = (qsub & 1) << 2;
            qlw0 =
                    (w1.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qlw1 =
                    (w1.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
            qhw0 =
                    (w1.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qhw1 =
                    (w1.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
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
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 176;
            qsub = ((0) >> 5) & 7;
            qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, qsub);
            psc = w3.getHalfFloat(qb).getFloat32() * (qsm >> 8);
            pmn = w3.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
            qn = qb + 48 + (qsub >> 1) * 32 + wk0;
            qh = qb + 16 + wk0;
            qs = (qsub & 1) << 2;
            qlw0 =
                    (w3.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qlw1 =
                    (w3.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
            qhw0 =
                    (w3.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qhw1 =
                    (w3.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 176;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, qsub);
                    psc = w1.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                    pmn = w1.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                    qn = qb + 48 + (qsub >> 1) * 32 + wk0;
                    qh = qb + 16 + wk0;
                    qs = (qsub & 1) << 2;
                    qlw0 =
                            (w1.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qlw1 =
                            (w1.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw0 =
                            (w1.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw1 =
                            (w1.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
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
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 176;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, qsub);
                    psc = w3.getHalfFloat(qb).getFloat32() * (qsm >> 8);
                    pmn = w3.getHalfFloat(qb + 2).getFloat32() * (qsm & 0xFF);
                    qn = qb + 48 + (qsub >> 1) * 32 + wk0;
                    qh = qb + 16 + wk0;
                    qs = (qsub & 1) << 2;
                    qlw0 =
                            (w3.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qlw1 =
                            (w3.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw0 =
                            (w3.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw1 =
                            (w3.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
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
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_K} for Q6_K weights, staged as {@link #batchedGemmQ6_K}
     * stages them: one fp16 {@code d}, a signed scale per sixteen weights, and the six-bit quantum
     * assembled from {@code ql} and {@code qh}, recentred by 32.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ6_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
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
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
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
        int qhs;
        int qlw0;
        int qlw1;
        int qhw0;
        int qhw1;
        int raw0;
        int raw1;
        int qsub;
        int qsm;
        float xsc0 = scaleBatch.get(TornadoMath.min(b0 + vr + 0, batch - 1));
        float xsc1 = scaleBatch.get(TornadoMath.min(b0 + vr + 32, batch - 1));
        Float4 rv0 = rmsWeights.getFloat4(0 + vc);
        Float4 fx00 = x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + 0 + vc);
        px0_0 = rv0.getX() * xsc0 * fx00.getX();
        px0_1 = rv0.getY() * xsc0 * fx00.getY();
        px0_2 = rv0.getZ() * xsc0 * fx00.getZ();
        px0_3 = rv0.getW() * xsc0 * fx00.getW();
        Float4 fx01 = x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + 0 + vc);
        px1_0 = rv0.getX() * xsc1 * fx01.getX();
        px1_1 = rv0.getY() * xsc1 * fx01.getY();
        px1_2 = rv0.getZ() * xsc1 * fx01.getZ();
        px1_3 = rv0.getW() * xsc1 * fx01.getW();
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 210;
            qsub = ((0) >> 5) & 7;
            qsm = w1.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
            psc = w1.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
            qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
            qh = qb + 128 + (qsub >> 2) * 32 + wk0;
            qs = ((qsub & 3) >> 1) << 2;
            qhs = (qsub & 3) << 1;
            qlw0 =
                    (w1.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qlw1 =
                    (w1.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
            qhw0 =
                    (w1.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qhw1 =
                    (w1.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w1.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
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
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8) + ((0) >> 8)) * 210;
            qsub = ((0) >> 5) & 7;
            qsm = w3.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
            psc = w3.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
            qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
            qh = qb + 128 + (qsub >> 2) * 32 + wk0;
            qs = ((qsub & 3) >> 1) << 2;
            qhs = (qsub & 3) << 1;
            qlw0 =
                    (w3.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qlw1 =
                    (w3.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF) << 16);
            qhw0 =
                    (w3.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF) << 16);
            qhw1 =
                    (w3.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w3.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF) << 16);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (k0 + 32 < dim) {
                Float4 rvk032 = rmsWeights.getFloat4(k0 + 32 + vc);
                Float4 fxk0320 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 0, batch - 1) * dim + k0 + 32 + vc);
                px0_0 = rvk032.getX() * xsc0 * fxk0320.getX();
                px0_1 = rvk032.getY() * xsc0 * fxk0320.getY();
                px0_2 = rvk032.getZ() * xsc0 * fxk0320.getZ();
                px0_3 = rvk032.getW() * xsc0 * fxk0320.getW();
                Float4 fxk0321 =
                        x.getFloat4(TornadoMath.min(b0 + vr + 32, batch - 1) * dim + k0 + 32 + vc);
                px1_0 = rvk032.getX() * xsc1 * fxk0321.getX();
                px1_1 = rvk032.getY() * xsc1 * fxk0321.getY();
                px1_2 = rvk032.getZ() * xsc1 * fxk0321.getZ();
                px1_3 = rvk032.getW() * xsc1 * fxk0321.getW();
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 210;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = w1.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
                    psc = w1.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
                    qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
                    qh = qb + 128 + (qsub >> 2) * 32 + wk0;
                    qs = ((qsub & 3) >> 1) << 2;
                    qhs = (qsub & 3) << 1;
                    qlw0 =
                            (w1.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qlw1 =
                            (w1.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw0 =
                            (w1.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw1 =
                            (w1.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w1.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
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
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 8)
                                            + ((k0 + 32) >> 8))
                                    * 210;
                    qsub = ((k0 + 32) >> 5) & 7;
                    qsm = w3.get(qb + 192 + (qsub >> 2) * 8 + (wk0 >> 4) + (qsub & 3) * 2) & 0xFF;
                    psc = w3.getHalfFloat(qb + 208).getFloat32() * (qsm - ((qsm >> 7) << 8));
                    qn = qb + (qsub >> 2) * 64 + (qsub & 1) * 32 + wk0;
                    qh = qb + 128 + (qsub >> 2) * 32 + wk0;
                    qs = ((qsub & 3) >> 1) << 2;
                    qhs = (qsub & 3) << 1;
                    qlw0 =
                            (w3.getHalfFloat(qn + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qn + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qlw1 =
                            (w3.getHalfFloat(qn + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qn + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw0 =
                            (w3.getHalfFloat(qh + 0).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qh + 2).getHalfFloatValue() & 0xFFFF)
                                            << 16);
                    qhw1 =
                            (w3.getHalfFloat(qh + 4).getHalfFloatValue() & 0xFFFF)
                                    | ((w3.getHalfFloat(qh + 6).getHalfFloatValue() & 0xFFFF)
                                            << 16);
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
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, xs, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(u00, ws, (sg * 4 + 0) << 6, 8);
        context.simdgroupMatrixStore(g10, xs, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(u10, ws, (sg * 4 + 1) << 6, 8);
        context.simdgroupMatrixStore(g20, xs, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(u20, ws, (sg * 4 + 2) << 6, 8);
        context.simdgroupMatrixStore(g30, xs, (sg * 4 + 3) << 6, 8);
        context.simdgroupMatrixStore(u30, ws, (sg * 4 + 3) << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 256; e += 32) {
            int f = e >> 6;
            int t = b0 + sgRow + ((f / 1) << 3) + ((e & 63) >> 3);
            int r = r0 + sgCol + ((f % 1) << 3) + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = xs[sg * 256 + e];
                hb.set(t * hiddenDim + r, (gv / (1.0f + TornadoMath.exp(-gv))) * ws[sg * 256 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@code out[b * ldo + off + r] = sum_k x[b, k] * w[r, k]} on SIMD-group matrices, with FP16
     * activations and Q8_0 weights dequantized to {@code float} as they are staged: one projection
     * written into its columns of a wider packed row. {@code n} must be a multiple of 64. Grid:
     * {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ8Strided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 34;
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
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 34;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} for Q4_0 weights: each 32-wide slice one 18-byte block, its
     * nibbles recentred by 8: the staging of {@link #batchedGemmQ4_0}. Grid: {@code ceil(d / 64) *
     * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ4_0Strided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 18;
        psc = w.getHalfFloat(qb).getFloat32();
        qn = qb + 2 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
        pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
        pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
        pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
        pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
        pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
        pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
        pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
        pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 18;
                psc = w.getHalfFloat(qb).getFloat32();
                qn = qb + 2 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
                pw0 = (float) (((w.get(qn + 0) & 0xFF) >> qs) & 15) - 8.0f;
                pw1 = (float) (((w.get(qn + 1) & 0xFF) >> qs) & 15) - 8.0f;
                pw2 = (float) (((w.get(qn + 2) & 0xFF) >> qs) & 15) - 8.0f;
                pw3 = (float) (((w.get(qn + 3) & 0xFF) >> qs) & 15) - 8.0f;
                pw4 = (float) (((w.get(qn + 4) & 0xFF) >> qs) & 15) - 8.0f;
                pw5 = (float) (((w.get(qn + 5) & 0xFF) >> qs) & 15) - 8.0f;
                pw6 = (float) (((w.get(qn + 6) & 0xFF) >> qs) & 15) - 8.0f;
                pw7 = (float) (((w.get(qn + 7) & 0xFF) >> qs) & 15) - 8.0f;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} for Q4_1 weights: each 32-wide slice one 20-byte block,
     * dequantized to {@code d * q + m}: the staging of {@link #batchedGemmQ4_1}. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ4_1Strided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        float pmn;
        int qb;
        int qn;
        int qs;
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 20;
        psc = w.getHalfFloat(qb).getFloat32();
        pmn = w.getHalfFloat(qb + 2).getFloat32();
        qn = qb + 4 + (wk0 & 15);
        qs = (wk0 & 16) >> 2;
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
            ws[wrow * LDS + wk0 + 0] = pw0 * psc + pmn;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc + pmn;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc + pmn;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc + pmn;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc + pmn;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc + pmn;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc + pmn;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc + pmn;
            context.localBarrier();
            if (k0 + 32 < n) {
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 20;
                psc = w.getHalfFloat(qb).getFloat32();
                pmn = w.getHalfFloat(qb + 2).getFloat32();
                qn = qb + 4 + (wk0 & 15);
                qs = (wk0 & 16) >> 2;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} for Q4_K weights: each 32-wide slice one sub-block of a
     * 144-byte super-block, its six-bit scale and minimum decoded as it is staged: the staging of
     * {@link #batchedGemmQ4_K}. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ4_KStrided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 144;
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
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 144;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} for Q5_K weights: each 32-wide slice one sub-block of a
     * 176-byte super-block, the fifth bits read from {@code qh}: the staging of {@link
     * #batchedGemmQ5_K}. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ5_KStrided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 176;
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
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 176;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} for Q6_K weights: each 32-wide slice half a group of a
     * 210-byte super-block, recentred by 32, with the sixteen-wide sub-block's scale: the staging
     * of {@link #batchedGemmQ6_K}. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of
     * {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfQ6_KStrided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
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
        int b0 = (grp / rowTiles) << 6;
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
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((0) >> 8)) * 210;
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
                Half4 hxk0320 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
                px0_0 = hxk0320.getX().getFloat32();
                px0_1 = hxk0320.getY().getFloat32();
                px0_2 = hxk0320.getZ().getFloat32();
                px0_3 = hxk0320.getW().getFloat32();
                Half4 hxk0321 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
                px1_0 = hxk0321.getX().getFloat32();
                px1_1 = hxk0321.getY().getFloat32();
                px1_2 = hxk0321.getZ().getFloat32();
                px1_3 = hxk0321.getW().getFloat32();
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 8) + ((k0 + 32) >> 8)) * 210;
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
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmHalfQ8Strided} with FP16 weights. Grid: {@code ceil(d / 64) * ceil(batch /
     * 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmHalfFP16Strided(
            KernelContext context,
            HalfFloatArray inputBatch,
            FloatArray outputBatch,
            HalfFloatArray w,
            int n,
            int d,
            int batch,
            int ldo,
            int off) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] xs1 = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws1 = context.allocateHalfFloatLocalArray(64 * LDS);
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
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        float px0_0;
        float px0_1;
        float px0_2;
        float px0_3;
        float px1_0;
        float px1_1;
        float px1_2;
        float px1_3;
        float pw0_0;
        float pw0_1;
        float pw0_2;
        float pw0_3;
        float pw1_0;
        float pw1_1;
        float pw1_2;
        float pw1_3;
        Half4 hx00 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + 0 + vc);
        px0_0 = hx00.getX().getFloat32();
        px0_1 = hx00.getY().getFloat32();
        px0_2 = hx00.getZ().getFloat32();
        px0_3 = hx00.getW().getFloat32();
        Half4 hx01 = inputBatch.getHalf4(TornadoMath.min(b0 + vr + 32, batch - 1) * n + 0 + vc);
        px1_0 = hx01.getX().getFloat32();
        px1_1 = hx01.getY().getFloat32();
        px1_2 = hx01.getZ().getFloat32();
        px1_3 = hx01.getW().getFloat32();
        Half4 hw00 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + 0 + vc);
        pw0_0 = hw00.getX().getFloat32();
        pw0_1 = hw00.getY().getFloat32();
        pw0_2 = hw00.getZ().getFloat32();
        pw0_3 = hw00.getW().getFloat32();
        Half4 hw01 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + 0 + vc);
        pw1_0 = hw01.getX().getFloat32();
        pw1_1 = hw01.getY().getFloat32();
        pw1_2 = hw01.getZ().getFloat32();
        pw1_3 = hw01.getW().getFloat32();
        xs[(vr + 0) * LDS + vc + 0] = px0_0;
        xs[(vr + 0) * LDS + vc + 1] = px0_1;
        xs[(vr + 0) * LDS + vc + 2] = px0_2;
        xs[(vr + 0) * LDS + vc + 3] = px0_3;
        xs[(vr + 32) * LDS + vc + 0] = px1_0;
        xs[(vr + 32) * LDS + vc + 1] = px1_1;
        xs[(vr + 32) * LDS + vc + 2] = px1_2;
        xs[(vr + 32) * LDS + vc + 3] = px1_3;
        ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
        ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
        ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
        ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
        ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
        ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
        ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
        ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
        context.localBarrier();
        for (int k0 = 0; k0 < n; k0 += 64) {
            Half4 hxk0320 =
                    inputBatch.getHalf4(TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 32 + vc);
            px0_0 = hxk0320.getX().getFloat32();
            px0_1 = hxk0320.getY().getFloat32();
            px0_2 = hxk0320.getZ().getFloat32();
            px0_3 = hxk0320.getW().getFloat32();
            Half4 hxk0321 =
                    inputBatch.getHalf4(
                            TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 32 + vc);
            px1_0 = hxk0321.getX().getFloat32();
            px1_1 = hxk0321.getY().getFloat32();
            px1_2 = hxk0321.getZ().getFloat32();
            px1_3 = hxk0321.getW().getFloat32();
            Half4 hwk0320 = w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 32 + vc);
            pw0_0 = hwk0320.getX().getFloat32();
            pw0_1 = hwk0320.getY().getFloat32();
            pw0_2 = hwk0320.getZ().getFloat32();
            pw0_3 = hwk0320.getW().getFloat32();
            Half4 hwk0321 = w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 32 + vc);
            pw1_0 = hwk0321.getX().getFloat32();
            pw1_1 = hwk0321.getY().getFloat32();
            pw1_2 = hwk0321.getZ().getFloat32();
            pw1_3 = hwk0321.getW().getFloat32();
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
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
            xs1[(vr + 0) * LDS + vc + 0] = px0_0;
            xs1[(vr + 0) * LDS + vc + 1] = px0_1;
            xs1[(vr + 0) * LDS + vc + 2] = px0_2;
            xs1[(vr + 0) * LDS + vc + 3] = px0_3;
            xs1[(vr + 32) * LDS + vc + 0] = px1_0;
            xs1[(vr + 32) * LDS + vc + 1] = px1_1;
            xs1[(vr + 32) * LDS + vc + 2] = px1_2;
            xs1[(vr + 32) * LDS + vc + 3] = px1_3;
            ws1[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
            ws1[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
            ws1[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
            ws1[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
            ws1[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
            ws1[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
            ws1[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
            ws1[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            context.localBarrier();
            if (k0 + 64 < n) {
                Half4 hxk0640 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 0, batch - 1) * n + k0 + 64 + vc);
                px0_0 = hxk0640.getX().getFloat32();
                px0_1 = hxk0640.getY().getFloat32();
                px0_2 = hxk0640.getZ().getFloat32();
                px0_3 = hxk0640.getW().getFloat32();
                Half4 hxk0641 =
                        inputBatch.getHalf4(
                                TornadoMath.min(b0 + vr + 32, batch - 1) * n + k0 + 64 + vc);
                px1_0 = hxk0641.getX().getFloat32();
                px1_1 = hxk0641.getY().getFloat32();
                px1_2 = hxk0641.getZ().getFloat32();
                px1_3 = hxk0641.getW().getFloat32();
                Half4 hwk0640 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 0, d - 1)) * n + k0 + 64 + vc);
                pw0_0 = hwk0640.getX().getFloat32();
                pw0_1 = hwk0640.getY().getFloat32();
                pw0_2 = hwk0640.getZ().getFloat32();
                pw0_3 = hwk0640.getW().getFloat32();
                Half4 hwk0641 =
                        w.getHalf4((TornadoMath.min(r0 + vr + 32, d - 1)) * n + k0 + 64 + vc);
                pw1_0 = hwk0641.getX().getFloat32();
                pw1_1 = hwk0641.getY().getFloat32();
                pw1_2 = hwk0641.getZ().getFloat32();
                pw1_3 = hwk0641.getW().getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs1, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs1, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs1, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs1, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws1, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            if (k0 + 64 < n) {
                xs[(vr + 0) * LDS + vc + 0] = px0_0;
                xs[(vr + 0) * LDS + vc + 1] = px0_1;
                xs[(vr + 0) * LDS + vc + 2] = px0_2;
                xs[(vr + 0) * LDS + vc + 3] = px0_3;
                xs[(vr + 32) * LDS + vc + 0] = px1_0;
                xs[(vr + 32) * LDS + vc + 1] = px1_1;
                xs[(vr + 32) * LDS + vc + 2] = px1_2;
                xs[(vr + 32) * LDS + vc + 3] = px1_3;
                ws[(vr + 0) * LDS + vc + 0] = new HalfFloat(pw0_0);
                ws[(vr + 0) * LDS + vc + 1] = new HalfFloat(pw0_1);
                ws[(vr + 0) * LDS + vc + 2] = new HalfFloat(pw0_2);
                ws[(vr + 0) * LDS + vc + 3] = new HalfFloat(pw0_3);
                ws[(vr + 32) * LDS + vc + 0] = new HalfFloat(pw1_0);
                ws[(vr + 32) * LDS + vc + 1] = new HalfFloat(pw1_1);
                ws[(vr + 32) * LDS + vc + 2] = new HalfFloat(pw1_2);
                ws[(vr + 32) * LDS + vc + 3] = new HalfFloat(pw1_3);
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * ldo + off + r0 + sgCol + 8, ldo);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 0, ldo);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * ldo + off + r0 + sgCol + 8, ldo);
        } else {
            context.simdgroupMatrixStore(c00, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, xs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * ldo + off + r, xs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ8WithResidual} with the weights staged as FP16 (the activations stay
     * FP32, the sums FP32), 128 threads per workgroup and indices carried across the loop. {@code
     * n} must be a multiple of 64. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of 128
     * threads.
     */
    // @formatter:on
    public static void batchedGemmQ8WithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int blk = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 34;
        int q = blk + 2 + kh;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
        for (int k0 = 0; k0 < n; k0 += 32) {
            float sc = w.getHalfFloat(blk).getFloat32();
            float a0 = w.get(q + 0) * sc;
            float a1 = w.get(q + 1) * sc;
            float a2 = w.get(q + 2) * sc;
            float a3 = w.get(q + 3) * sc;
            float a4 = w.get(q + 4) * sc;
            float a5 = w.get(q + 5) * sc;
            float a6 = w.get(q + 6) * sc;
            float a7 = w.get(q + 7) * sc;
            float a8 = w.get(q + 8) * sc;
            float a9 = w.get(q + 9) * sc;
            float a10 = w.get(q + 10) * sc;
            float a11 = w.get(q + 11) * sc;
            float a12 = w.get(q + 12) * sc;
            float a13 = w.get(q + 13) * sc;
            float a14 = w.get(q + 14) * sc;
            float a15 = w.get(q + 15) * sc;
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
            blk += 34;
            q += 34;
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQKVQ8} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch /
     * 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQKVQ8HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (qDim + 2 * kvDim) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        int width = which == 0 ? qDim : kvDim;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int blk = (local0 + lr) * (dim >> 5) * 34;
        int qi = blk + 2 + kh;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * dim + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * width + local0 + ((sg & 1) << 5);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            float sc;
            if (which == 0) {
                sc = wq.getHalfFloat(blk).getFloat32();
                a0 = wq.get(qi + 0) * sc;
                a1 = wq.get(qi + 1) * sc;
                a2 = wq.get(qi + 2) * sc;
                a3 = wq.get(qi + 3) * sc;
                a4 = wq.get(qi + 4) * sc;
                a5 = wq.get(qi + 5) * sc;
                a6 = wq.get(qi + 6) * sc;
                a7 = wq.get(qi + 7) * sc;
                a8 = wq.get(qi + 8) * sc;
                a9 = wq.get(qi + 9) * sc;
                a10 = wq.get(qi + 10) * sc;
                a11 = wq.get(qi + 11) * sc;
                a12 = wq.get(qi + 12) * sc;
                a13 = wq.get(qi + 13) * sc;
                a14 = wq.get(qi + 14) * sc;
                a15 = wq.get(qi + 15) * sc;
            } else if (which == 1) {
                sc = wk.getHalfFloat(blk).getFloat32();
                a0 = wk.get(qi + 0) * sc;
                a1 = wk.get(qi + 1) * sc;
                a2 = wk.get(qi + 2) * sc;
                a3 = wk.get(qi + 3) * sc;
                a4 = wk.get(qi + 4) * sc;
                a5 = wk.get(qi + 5) * sc;
                a6 = wk.get(qi + 6) * sc;
                a7 = wk.get(qi + 7) * sc;
                a8 = wk.get(qi + 8) * sc;
                a9 = wk.get(qi + 9) * sc;
                a10 = wk.get(qi + 10) * sc;
                a11 = wk.get(qi + 11) * sc;
                a12 = wk.get(qi + 12) * sc;
                a13 = wk.get(qi + 13) * sc;
                a14 = wk.get(qi + 14) * sc;
                a15 = wk.get(qi + 15) * sc;
            } else {
                sc = wv.getHalfFloat(blk).getFloat32();
                a0 = wv.get(qi + 0) * sc;
                a1 = wv.get(qi + 1) * sc;
                a2 = wv.get(qi + 2) * sc;
                a3 = wv.get(qi + 3) * sc;
                a4 = wv.get(qi + 4) * sc;
                a5 = wv.get(qi + 5) * sc;
                a6 = wv.get(qi + 6) * sc;
                a7 = wv.get(qi + 7) * sc;
                a8 = wv.get(qi + 8) * sc;
                a9 = wv.get(qi + 9) * sc;
                a10 = wv.get(qi + 10) * sc;
                a11 = wv.get(qi + 11) * sc;
                a12 = wv.get(qi + 12) * sc;
                a13 = wv.get(qi + 13) * sc;
                a14 = wv.get(qi + 14) * sc;
                a15 = wv.get(qi + 15) * sc;
            }
            float y0 = x.get(xi + 0);
            float y1 = x.get(xi + 1);
            float y2 = x.get(xi + 2);
            float y3 = x.get(xi + 3);
            float y4 = x.get(xi + 4);
            float y5 = x.get(xi + 5);
            float y6 = x.get(xi + 6);
            float y7 = x.get(xi + 7);
            float y8 = x.get(xi + 8);
            float y9 = x.get(xi + 9);
            float y10 = x.get(xi + 10);
            float y11 = x.get(xi + 11);
            float y12 = x.get(xi + 12);
            float y13 = x.get(xi + 13);
            float y14 = x.get(xi + 14);
            float y15 = x.get(xi + 15);
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
            blk += 34;
            qi += 34;
            xi += 32;
        }
        if (full) {
            if (which == 0) {
                context.simdgroupMatrixStore(c0, q, cbase + 0 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c1, q, cbase + 0 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c2, q, cbase + 0 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c3, q, cbase + 0 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c4, q, cbase + 8 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c5, q, cbase + 8 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c6, q, cbase + 8 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c7, q, cbase + 8 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c8, q, cbase + 16 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c9, q, cbase + 16 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c10, q, cbase + 16 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c11, q, cbase + 16 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c12, q, cbase + 24 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c13, q, cbase + 24 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c14, q, cbase + 24 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c15, q, cbase + 24 * qDim + 24, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(c0, k, cbase + 0 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c1, k, cbase + 0 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c2, k, cbase + 0 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c3, k, cbase + 0 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c4, k, cbase + 8 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c5, k, cbase + 8 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c6, k, cbase + 8 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c7, k, cbase + 8 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c8, k, cbase + 16 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c9, k, cbase + 16 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c10, k, cbase + 16 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c11, k, cbase + 16 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c12, k, cbase + 24 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c13, k, cbase + 24 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c14, k, cbase + 24 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c15, k, cbase + 24 * kvDim + 24, kvDim);
            } else {
                context.simdgroupMatrixStore(c0, v, cbase + 0 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c1, v, cbase + 0 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c2, v, cbase + 0 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c3, v, cbase + 0 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c4, v, cbase + 8 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c5, v, cbase + 8 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c6, v, cbase + 8 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c7, v, cbase + 8 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c8, v, cbase + 16 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c9, v, cbase + 16 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c10, v, cbase + 16 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c11, v, cbase + 16 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c12, v, cbase + 24 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c13, v, cbase + 24 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c14, v, cbase + 24 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c15, v, cbase + 24 * kvDim + 24, kvDim);
            }
        } else {
            int t0 = b0 + ((sg >> 1) << 5);
            int rr = local0 + ((sg & 1) << 5);
            context.localBarrier();
            context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ8} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. A tile holds 32 gate rows and the 32 matching up rows.
     * Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ8HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int blk = row * (dim >> 5) * 34;
        int qi = blk + 2 + kh;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            float sc;
            if (!up) {
                sc = w1.getHalfFloat(blk).getFloat32();
                a0 = w1.get(qi + 0) * sc;
                a1 = w1.get(qi + 1) * sc;
                a2 = w1.get(qi + 2) * sc;
                a3 = w1.get(qi + 3) * sc;
                a4 = w1.get(qi + 4) * sc;
                a5 = w1.get(qi + 5) * sc;
                a6 = w1.get(qi + 6) * sc;
                a7 = w1.get(qi + 7) * sc;
                a8 = w1.get(qi + 8) * sc;
                a9 = w1.get(qi + 9) * sc;
                a10 = w1.get(qi + 10) * sc;
                a11 = w1.get(qi + 11) * sc;
                a12 = w1.get(qi + 12) * sc;
                a13 = w1.get(qi + 13) * sc;
                a14 = w1.get(qi + 14) * sc;
                a15 = w1.get(qi + 15) * sc;
            } else {
                sc = w3.getHalfFloat(blk).getFloat32();
                a0 = w3.get(qi + 0) * sc;
                a1 = w3.get(qi + 1) * sc;
                a2 = w3.get(qi + 2) * sc;
                a3 = w3.get(qi + 3) * sc;
                a4 = w3.get(qi + 4) * sc;
                a5 = w3.get(qi + 5) * sc;
                a6 = w3.get(qi + 6) * sc;
                a7 = w3.get(qi + 7) * sc;
                a8 = w3.get(qi + 8) * sc;
                a9 = w3.get(qi + 9) * sc;
                a10 = w3.get(qi + 10) * sc;
                a11 = w3.get(qi + 11) * sc;
                a12 = w3.get(qi + 12) * sc;
                a13 = w3.get(qi + 13) * sc;
                a14 = w3.get(qi + 14) * sc;
                a15 = w3.get(qi + 15) * sc;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            blk += 34;
            qi += 34;
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 144;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_KWithResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_KWithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 144;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code dim} must be a multiple of 256. Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_KHalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int rb = row * (dim >> 8) * 144;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (!up) {
                int qb = rb + (k0 >> 8) * 144;
                int sub = (k0 >> 5) & 7;
                int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, sub);
                float sc = w1.getHalfFloat(qb).getFloat32() * (sm >> 8);
                float mn = w1.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
                int qn = qb + 16 + (sub >> 1) * 32 + kh;
                int sh = (sub & 1) << 2;
                a0 = (float) (((w1.get(qn + 0) & 0xFF) >> sh) & 15) * sc - mn;
                a1 = (float) (((w1.get(qn + 1) & 0xFF) >> sh) & 15) * sc - mn;
                a2 = (float) (((w1.get(qn + 2) & 0xFF) >> sh) & 15) * sc - mn;
                a3 = (float) (((w1.get(qn + 3) & 0xFF) >> sh) & 15) * sc - mn;
                a4 = (float) (((w1.get(qn + 4) & 0xFF) >> sh) & 15) * sc - mn;
                a5 = (float) (((w1.get(qn + 5) & 0xFF) >> sh) & 15) * sc - mn;
                a6 = (float) (((w1.get(qn + 6) & 0xFF) >> sh) & 15) * sc - mn;
                a7 = (float) (((w1.get(qn + 7) & 0xFF) >> sh) & 15) * sc - mn;
                a8 = (float) (((w1.get(qn + 8) & 0xFF) >> sh) & 15) * sc - mn;
                a9 = (float) (((w1.get(qn + 9) & 0xFF) >> sh) & 15) * sc - mn;
                a10 = (float) (((w1.get(qn + 10) & 0xFF) >> sh) & 15) * sc - mn;
                a11 = (float) (((w1.get(qn + 11) & 0xFF) >> sh) & 15) * sc - mn;
                a12 = (float) (((w1.get(qn + 12) & 0xFF) >> sh) & 15) * sc - mn;
                a13 = (float) (((w1.get(qn + 13) & 0xFF) >> sh) & 15) * sc - mn;
                a14 = (float) (((w1.get(qn + 14) & 0xFF) >> sh) & 15) * sc - mn;
                a15 = (float) (((w1.get(qn + 15) & 0xFF) >> sh) & 15) * sc - mn;
            } else {
                int qb = rb + (k0 >> 8) * 144;
                int sub = (k0 >> 5) & 7;
                int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, sub);
                float sc = w3.getHalfFloat(qb).getFloat32() * (sm >> 8);
                float mn = w3.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
                int qn = qb + 16 + (sub >> 1) * 32 + kh;
                int sh = (sub & 1) << 2;
                a0 = (float) (((w3.get(qn + 0) & 0xFF) >> sh) & 15) * sc - mn;
                a1 = (float) (((w3.get(qn + 1) & 0xFF) >> sh) & 15) * sc - mn;
                a2 = (float) (((w3.get(qn + 2) & 0xFF) >> sh) & 15) * sc - mn;
                a3 = (float) (((w3.get(qn + 3) & 0xFF) >> sh) & 15) * sc - mn;
                a4 = (float) (((w3.get(qn + 4) & 0xFF) >> sh) & 15) * sc - mn;
                a5 = (float) (((w3.get(qn + 5) & 0xFF) >> sh) & 15) * sc - mn;
                a6 = (float) (((w3.get(qn + 6) & 0xFF) >> sh) & 15) * sc - mn;
                a7 = (float) (((w3.get(qn + 7) & 0xFF) >> sh) & 15) * sc - mn;
                a8 = (float) (((w3.get(qn + 8) & 0xFF) >> sh) & 15) * sc - mn;
                a9 = (float) (((w3.get(qn + 9) & 0xFF) >> sh) & 15) * sc - mn;
                a10 = (float) (((w3.get(qn + 10) & 0xFF) >> sh) & 15) * sc - mn;
                a11 = (float) (((w3.get(qn + 11) & 0xFF) >> sh) & 15) * sc - mn;
                a12 = (float) (((w3.get(qn + 12) & 0xFF) >> sh) & 15) * sc - mn;
                a13 = (float) (((w3.get(qn + 13) & 0xFF) >> sh) & 15) * sc - mn;
                a14 = (float) (((w3.get(qn + 14) & 0xFF) >> sh) & 15) * sc - mn;
                a15 = (float) (((w3.get(qn + 15) & 0xFF) >> sh) & 15) * sc - mn;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ6_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ6_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 210;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ6_KWithResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ6_KWithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 210;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ6_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code dim} must be a multiple of 256. Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ6_KHalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int rb = row * (dim >> 8) * 210;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (!up) {
                int qb = rb + (k0 >> 8) * 210;
                int sub = (k0 >> 5) & 7;
                int sv = w1.get(qb + 192 + (sub >> 2) * 8 + (kh >> 4) + (sub & 3) * 2) & 0xFF;
                float sc = w1.getHalfFloat(qb + 208).getFloat32() * (sv - ((sv >> 7) << 8));
                int qn = qb + (sub >> 2) * 64 + (sub & 1) * 32 + kh;
                int qh = qb + 128 + (sub >> 2) * 32 + kh;
                int sl = ((sub & 3) >> 1) << 2;
                int shh = (sub & 3) << 1;
                a0 =
                        ((float)
                                                ((((w1.get(qn + 0) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 0) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a1 =
                        ((float)
                                                ((((w1.get(qn + 1) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 1) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a2 =
                        ((float)
                                                ((((w1.get(qn + 2) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 2) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a3 =
                        ((float)
                                                ((((w1.get(qn + 3) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 3) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a4 =
                        ((float)
                                                ((((w1.get(qn + 4) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 4) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a5 =
                        ((float)
                                                ((((w1.get(qn + 5) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 5) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a6 =
                        ((float)
                                                ((((w1.get(qn + 6) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 6) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a7 =
                        ((float)
                                                ((((w1.get(qn + 7) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 7) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a8 =
                        ((float)
                                                ((((w1.get(qn + 8) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 8) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a9 =
                        ((float)
                                                ((((w1.get(qn + 9) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 9) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a10 =
                        ((float)
                                                ((((w1.get(qn + 10) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 10) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a11 =
                        ((float)
                                                ((((w1.get(qn + 11) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 11) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a12 =
                        ((float)
                                                ((((w1.get(qn + 12) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 12) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a13 =
                        ((float)
                                                ((((w1.get(qn + 13) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 13) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a14 =
                        ((float)
                                                ((((w1.get(qn + 14) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 14) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a15 =
                        ((float)
                                                ((((w1.get(qn + 15) & 0xFF) >> sl) & 15)
                                                        | ((((w1.get(qh + 15) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
            } else {
                int qb = rb + (k0 >> 8) * 210;
                int sub = (k0 >> 5) & 7;
                int sv = w3.get(qb + 192 + (sub >> 2) * 8 + (kh >> 4) + (sub & 3) * 2) & 0xFF;
                float sc = w3.getHalfFloat(qb + 208).getFloat32() * (sv - ((sv >> 7) << 8));
                int qn = qb + (sub >> 2) * 64 + (sub & 1) * 32 + kh;
                int qh = qb + 128 + (sub >> 2) * 32 + kh;
                int sl = ((sub & 3) >> 1) << 2;
                int shh = (sub & 3) << 1;
                a0 =
                        ((float)
                                                ((((w3.get(qn + 0) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 0) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a1 =
                        ((float)
                                                ((((w3.get(qn + 1) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 1) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a2 =
                        ((float)
                                                ((((w3.get(qn + 2) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 2) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a3 =
                        ((float)
                                                ((((w3.get(qn + 3) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 3) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a4 =
                        ((float)
                                                ((((w3.get(qn + 4) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 4) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a5 =
                        ((float)
                                                ((((w3.get(qn + 5) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 5) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a6 =
                        ((float)
                                                ((((w3.get(qn + 6) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 6) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a7 =
                        ((float)
                                                ((((w3.get(qn + 7) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 7) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a8 =
                        ((float)
                                                ((((w3.get(qn + 8) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 8) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a9 =
                        ((float)
                                                ((((w3.get(qn + 9) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 9) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a10 =
                        ((float)
                                                ((((w3.get(qn + 10) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 10) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a11 =
                        ((float)
                                                ((((w3.get(qn + 11) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 11) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a12 =
                        ((float)
                                                ((((w3.get(qn + 12) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 12) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a13 =
                        ((float)
                                                ((((w3.get(qn + 13) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 13) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a14 =
                        ((float)
                                                ((((w3.get(qn + 14) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 14) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
                a15 =
                        ((float)
                                                ((((w3.get(qn + 15) & 0xFF) >> sl) & 15)
                                                        | ((((w3.get(qh + 15) & 0xFF) >> shh) & 3)
                                                                << 4))
                                        - 32.0f)
                                * sc;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ5_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ5_KHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 176;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ5_KWithResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 256. Grid: {@code
     * ceil(d / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ5_KWithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 8) * 176;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ5_K} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code dim} must be a multiple of 256. Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ5_KHalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int rb = row * (dim >> 8) * 176;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (!up) {
                int qb = rb + (k0 >> 8) * 176;
                int sub = (k0 >> 5) & 7;
                int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w1, qb + 4, sub);
                float sc = w1.getHalfFloat(qb).getFloat32() * (sm >> 8);
                float mn = w1.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
                int qn = qb + 48 + (sub >> 1) * 32 + kh;
                int qh = qb + 16 + kh;
                int sh = (sub & 1) << 2;
                a0 =
                        (float)
                                                ((((w1.get(qn + 0) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 0) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a1 =
                        (float)
                                                ((((w1.get(qn + 1) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 1) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a2 =
                        (float)
                                                ((((w1.get(qn + 2) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 2) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a3 =
                        (float)
                                                ((((w1.get(qn + 3) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 3) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a4 =
                        (float)
                                                ((((w1.get(qn + 4) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 4) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a5 =
                        (float)
                                                ((((w1.get(qn + 5) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 5) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a6 =
                        (float)
                                                ((((w1.get(qn + 6) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 6) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a7 =
                        (float)
                                                ((((w1.get(qn + 7) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 7) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a8 =
                        (float)
                                                ((((w1.get(qn + 8) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 8) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a9 =
                        (float)
                                                ((((w1.get(qn + 9) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 9) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a10 =
                        (float)
                                                ((((w1.get(qn + 10) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 10) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a11 =
                        (float)
                                                ((((w1.get(qn + 11) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 11) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a12 =
                        (float)
                                                ((((w1.get(qn + 12) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 12) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a13 =
                        (float)
                                                ((((w1.get(qn + 13) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 13) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a14 =
                        (float)
                                                ((((w1.get(qn + 14) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 14) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a15 =
                        (float)
                                                ((((w1.get(qn + 15) & 0xFF) >> sh) & 15)
                                                        | ((((w1.get(qh + 15) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
            } else {
                int qb = rb + (k0 >> 8) * 176;
                int sub = (k0 >> 5) & 7;
                int sm = TransformerComputeKernelsQ4_K.scaleAndMin(w3, qb + 4, sub);
                float sc = w3.getHalfFloat(qb).getFloat32() * (sm >> 8);
                float mn = w3.getHalfFloat(qb + 2).getFloat32() * (sm & 0xFF);
                int qn = qb + 48 + (sub >> 1) * 32 + kh;
                int qh = qb + 16 + kh;
                int sh = (sub & 1) << 2;
                a0 =
                        (float)
                                                ((((w3.get(qn + 0) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 0) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a1 =
                        (float)
                                                ((((w3.get(qn + 1) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 1) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a2 =
                        (float)
                                                ((((w3.get(qn + 2) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 2) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a3 =
                        (float)
                                                ((((w3.get(qn + 3) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 3) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a4 =
                        (float)
                                                ((((w3.get(qn + 4) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 4) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a5 =
                        (float)
                                                ((((w3.get(qn + 5) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 5) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a6 =
                        (float)
                                                ((((w3.get(qn + 6) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 6) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a7 =
                        (float)
                                                ((((w3.get(qn + 7) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 7) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a8 =
                        (float)
                                                ((((w3.get(qn + 8) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 8) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a9 =
                        (float)
                                                ((((w3.get(qn + 9) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 9) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a10 =
                        (float)
                                                ((((w3.get(qn + 10) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 10) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a11 =
                        (float)
                                                ((((w3.get(qn + 11) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 11) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a12 =
                        (float)
                                                ((((w3.get(qn + 12) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 12) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a13 =
                        (float)
                                                ((((w3.get(qn + 13) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 13) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a14 =
                        (float)
                                                ((((w3.get(qn + 14) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 14) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
                a15 =
                        (float)
                                                ((((w3.get(qn + 15) & 0xFF) >> sh) & 15)
                                                        | ((((w3.get(qh + 15) & 0xFF) >> sub) & 1)
                                                                << 4))
                                        * sc
                                - mn;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 32. Grid: {@code ceil(d
     * / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_0HalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 18;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
            int qb = rb + (k0 >> 5) * 18;
            float sc = w.getHalfFloat(qb).getFloat32();
            int qn = qb + 2;
            int sh = kh >> 2;
            a0 = ((float) (((w.get(qn + 0) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a1 = ((float) (((w.get(qn + 1) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a2 = ((float) (((w.get(qn + 2) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a3 = ((float) (((w.get(qn + 3) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a4 = ((float) (((w.get(qn + 4) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a5 = ((float) (((w.get(qn + 5) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a6 = ((float) (((w.get(qn + 6) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a7 = ((float) (((w.get(qn + 7) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a8 = ((float) (((w.get(qn + 8) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a9 = ((float) (((w.get(qn + 9) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a10 = ((float) (((w.get(qn + 10) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a11 = ((float) (((w.get(qn + 11) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a12 = ((float) (((w.get(qn + 12) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a13 = ((float) (((w.get(qn + 13) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a14 = ((float) (((w.get(qn + 14) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a15 = ((float) (((w.get(qn + 15) & 0xFF) >> sh) & 15) - 8.0f) * sc;
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
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_0WithResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 32. Grid: {@code ceil(d
     * / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_0WithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 18;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
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
            int qb = rb + (k0 >> 5) * 18;
            float sc = w.getHalfFloat(qb).getFloat32();
            int qn = qb + 2;
            int sh = kh >> 2;
            a0 = ((float) (((w.get(qn + 0) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a1 = ((float) (((w.get(qn + 1) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a2 = ((float) (((w.get(qn + 2) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a3 = ((float) (((w.get(qn + 3) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a4 = ((float) (((w.get(qn + 4) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a5 = ((float) (((w.get(qn + 5) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a6 = ((float) (((w.get(qn + 6) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a7 = ((float) (((w.get(qn + 7) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a8 = ((float) (((w.get(qn + 8) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a9 = ((float) (((w.get(qn + 9) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a10 = ((float) (((w.get(qn + 10) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a11 = ((float) (((w.get(qn + 11) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a12 = ((float) (((w.get(qn + 12) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a13 = ((float) (((w.get(qn + 13) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a14 = ((float) (((w.get(qn + 14) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            a15 = ((float) (((w.get(qn + 15) & 0xFF) >> sh) & 15) - 8.0f) * sc;
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_0} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code dim} must be a multiple of 32. Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_0HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int rb = row * (dim >> 5) * 18;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (!up) {
                int qb = rb + (k0 >> 5) * 18;
                float sc = w1.getHalfFloat(qb).getFloat32();
                int qn = qb + 2;
                int sh = kh >> 2;
                a0 = ((float) (((w1.get(qn + 0) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a1 = ((float) (((w1.get(qn + 1) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a2 = ((float) (((w1.get(qn + 2) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a3 = ((float) (((w1.get(qn + 3) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a4 = ((float) (((w1.get(qn + 4) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a5 = ((float) (((w1.get(qn + 5) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a6 = ((float) (((w1.get(qn + 6) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a7 = ((float) (((w1.get(qn + 7) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a8 = ((float) (((w1.get(qn + 8) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a9 = ((float) (((w1.get(qn + 9) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a10 = ((float) (((w1.get(qn + 10) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a11 = ((float) (((w1.get(qn + 11) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a12 = ((float) (((w1.get(qn + 12) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a13 = ((float) (((w1.get(qn + 13) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a14 = ((float) (((w1.get(qn + 14) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a15 = ((float) (((w1.get(qn + 15) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            } else {
                int qb = rb + (k0 >> 5) * 18;
                float sc = w3.getHalfFloat(qb).getFloat32();
                int qn = qb + 2;
                int sh = kh >> 2;
                a0 = ((float) (((w3.get(qn + 0) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a1 = ((float) (((w3.get(qn + 1) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a2 = ((float) (((w3.get(qn + 2) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a3 = ((float) (((w3.get(qn + 3) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a4 = ((float) (((w3.get(qn + 4) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a5 = ((float) (((w3.get(qn + 5) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a6 = ((float) (((w3.get(qn + 6) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a7 = ((float) (((w3.get(qn + 7) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a8 = ((float) (((w3.get(qn + 8) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a9 = ((float) (((w3.get(qn + 9) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a10 = ((float) (((w3.get(qn + 10) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a11 = ((float) (((w3.get(qn + 11) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a12 = ((float) (((w3.get(qn + 12) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a13 = ((float) (((w3.get(qn + 13) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a14 = ((float) (((w3.get(qn + 14) & 0xFF) >> sh) & 15) - 8.0f) * sc;
                a15 = ((float) (((w3.get(qn + 15) & 0xFF) >> sh) & 15) - 8.0f) * sc;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_1} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 32. Grid: {@code ceil(d
     * / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_1HalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 20;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
            int qb = rb + (k0 >> 5) * 20;
            float sc = w.getHalfFloat(qb).getFloat32();
            float mn = w.getHalfFloat(qb + 2).getFloat32();
            int qn = qb + 4;
            int sh = kh >> 2;
            a0 = (float) (((w.get(qn + 0) & 0xFF) >> sh) & 15) * sc + mn;
            a1 = (float) (((w.get(qn + 1) & 0xFF) >> sh) & 15) * sc + mn;
            a2 = (float) (((w.get(qn + 2) & 0xFF) >> sh) & 15) * sc + mn;
            a3 = (float) (((w.get(qn + 3) & 0xFF) >> sh) & 15) * sc + mn;
            a4 = (float) (((w.get(qn + 4) & 0xFF) >> sh) & 15) * sc + mn;
            a5 = (float) (((w.get(qn + 5) & 0xFF) >> sh) & 15) * sc + mn;
            a6 = (float) (((w.get(qn + 6) & 0xFF) >> sh) & 15) * sc + mn;
            a7 = (float) (((w.get(qn + 7) & 0xFF) >> sh) & 15) * sc + mn;
            a8 = (float) (((w.get(qn + 8) & 0xFF) >> sh) & 15) * sc + mn;
            a9 = (float) (((w.get(qn + 9) & 0xFF) >> sh) & 15) * sc + mn;
            a10 = (float) (((w.get(qn + 10) & 0xFF) >> sh) & 15) * sc + mn;
            a11 = (float) (((w.get(qn + 11) & 0xFF) >> sh) & 15) * sc + mn;
            a12 = (float) (((w.get(qn + 12) & 0xFF) >> sh) & 15) * sc + mn;
            a13 = (float) (((w.get(qn + 13) & 0xFF) >> sh) & 15) * sc + mn;
            a14 = (float) (((w.get(qn + 14) & 0xFF) >> sh) & 15) * sc + mn;
            a15 = (float) (((w.get(qn + 15) & 0xFF) >> sh) & 15) * sc + mn;
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
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ4_1WithResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code n} must be a multiple of 32. Grid: {@code ceil(d
     * / 64) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ4_1WithResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int rb = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 20;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
        if (full) {
            c0 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 0, d);
            c1 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 8, d);
            c2 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 16, d);
            c3 = context.simdgroupMatrixLoad(outputBatch, cbase + 0 * d + 24, d);
            c4 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 0, d);
            c5 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 8, d);
            c6 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 16, d);
            c7 = context.simdgroupMatrixLoad(outputBatch, cbase + 8 * d + 24, d);
            c8 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 0, d);
            c9 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 8, d);
            c10 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 16, d);
            c11 = context.simdgroupMatrixLoad(outputBatch, cbase + 16 * d + 24, d);
            c12 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 0, d);
            c13 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 8, d);
            c14 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 16, d);
            c15 = context.simdgroupMatrixLoad(outputBatch, cbase + 24 * d + 24, d);
        }
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
            int qb = rb + (k0 >> 5) * 20;
            float sc = w.getHalfFloat(qb).getFloat32();
            float mn = w.getHalfFloat(qb + 2).getFloat32();
            int qn = qb + 4;
            int sh = kh >> 2;
            a0 = (float) (((w.get(qn + 0) & 0xFF) >> sh) & 15) * sc + mn;
            a1 = (float) (((w.get(qn + 1) & 0xFF) >> sh) & 15) * sc + mn;
            a2 = (float) (((w.get(qn + 2) & 0xFF) >> sh) & 15) * sc + mn;
            a3 = (float) (((w.get(qn + 3) & 0xFF) >> sh) & 15) * sc + mn;
            a4 = (float) (((w.get(qn + 4) & 0xFF) >> sh) & 15) * sc + mn;
            a5 = (float) (((w.get(qn + 5) & 0xFF) >> sh) & 15) * sc + mn;
            a6 = (float) (((w.get(qn + 6) & 0xFF) >> sh) & 15) * sc + mn;
            a7 = (float) (((w.get(qn + 7) & 0xFF) >> sh) & 15) * sc + mn;
            a8 = (float) (((w.get(qn + 8) & 0xFF) >> sh) & 15) * sc + mn;
            a9 = (float) (((w.get(qn + 9) & 0xFF) >> sh) & 15) * sc + mn;
            a10 = (float) (((w.get(qn + 10) & 0xFF) >> sh) & 15) * sc + mn;
            a11 = (float) (((w.get(qn + 11) & 0xFF) >> sh) & 15) * sc + mn;
            a12 = (float) (((w.get(qn + 12) & 0xFF) >> sh) & 15) * sc + mn;
            a13 = (float) (((w.get(qn + 13) & 0xFF) >> sh) & 15) * sc + mn;
            a14 = (float) (((w.get(qn + 14) & 0xFF) >> sh) & 15) * sc + mn;
            a15 = (float) (((w.get(qn + 15) & 0xFF) >> sh) & 15) * sc + mn;
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
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpQ4_1} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. {@code dim} must be a multiple of 32. Grid: {@code
     * ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ4_1HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int rb = row * (dim >> 5) * 20;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            if (!up) {
                int qb = rb + (k0 >> 5) * 20;
                float sc = w1.getHalfFloat(qb).getFloat32();
                float mn = w1.getHalfFloat(qb + 2).getFloat32();
                int qn = qb + 4;
                int sh = kh >> 2;
                a0 = (float) (((w1.get(qn + 0) & 0xFF) >> sh) & 15) * sc + mn;
                a1 = (float) (((w1.get(qn + 1) & 0xFF) >> sh) & 15) * sc + mn;
                a2 = (float) (((w1.get(qn + 2) & 0xFF) >> sh) & 15) * sc + mn;
                a3 = (float) (((w1.get(qn + 3) & 0xFF) >> sh) & 15) * sc + mn;
                a4 = (float) (((w1.get(qn + 4) & 0xFF) >> sh) & 15) * sc + mn;
                a5 = (float) (((w1.get(qn + 5) & 0xFF) >> sh) & 15) * sc + mn;
                a6 = (float) (((w1.get(qn + 6) & 0xFF) >> sh) & 15) * sc + mn;
                a7 = (float) (((w1.get(qn + 7) & 0xFF) >> sh) & 15) * sc + mn;
                a8 = (float) (((w1.get(qn + 8) & 0xFF) >> sh) & 15) * sc + mn;
                a9 = (float) (((w1.get(qn + 9) & 0xFF) >> sh) & 15) * sc + mn;
                a10 = (float) (((w1.get(qn + 10) & 0xFF) >> sh) & 15) * sc + mn;
                a11 = (float) (((w1.get(qn + 11) & 0xFF) >> sh) & 15) * sc + mn;
                a12 = (float) (((w1.get(qn + 12) & 0xFF) >> sh) & 15) * sc + mn;
                a13 = (float) (((w1.get(qn + 13) & 0xFF) >> sh) & 15) * sc + mn;
                a14 = (float) (((w1.get(qn + 14) & 0xFF) >> sh) & 15) * sc + mn;
                a15 = (float) (((w1.get(qn + 15) & 0xFF) >> sh) & 15) * sc + mn;
            } else {
                int qb = rb + (k0 >> 5) * 20;
                float sc = w3.getHalfFloat(qb).getFloat32();
                float mn = w3.getHalfFloat(qb + 2).getFloat32();
                int qn = qb + 4;
                int sh = kh >> 2;
                a0 = (float) (((w3.get(qn + 0) & 0xFF) >> sh) & 15) * sc + mn;
                a1 = (float) (((w3.get(qn + 1) & 0xFF) >> sh) & 15) * sc + mn;
                a2 = (float) (((w3.get(qn + 2) & 0xFF) >> sh) & 15) * sc + mn;
                a3 = (float) (((w3.get(qn + 3) & 0xFF) >> sh) & 15) * sc + mn;
                a4 = (float) (((w3.get(qn + 4) & 0xFF) >> sh) & 15) * sc + mn;
                a5 = (float) (((w3.get(qn + 5) & 0xFF) >> sh) & 15) * sc + mn;
                a6 = (float) (((w3.get(qn + 6) & 0xFF) >> sh) & 15) * sc + mn;
                a7 = (float) (((w3.get(qn + 7) & 0xFF) >> sh) & 15) * sc + mn;
                a8 = (float) (((w3.get(qn + 8) & 0xFF) >> sh) & 15) * sc + mn;
                a9 = (float) (((w3.get(qn + 9) & 0xFF) >> sh) & 15) * sc + mn;
                a10 = (float) (((w3.get(qn + 10) & 0xFF) >> sh) & 15) * sc + mn;
                a11 = (float) (((w3.get(qn + 11) & 0xFF) >> sh) & 15) * sc + mn;
                a12 = (float) (((w3.get(qn + 12) & 0xFF) >> sh) & 15) * sc + mn;
                a13 = (float) (((w3.get(qn + 13) & 0xFF) >> sh) & 15) * sc + mn;
                a14 = (float) (((w3.get(qn + 14) & 0xFF) >> sh) & 15) * sc + mn;
                a15 = (float) (((w3.get(qn + 15) & 0xFF) >> sh) & 15) * sc + mn;
            }
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQ8WithScaledResidual} with the weights staged as FP16, as {@link
     * #batchedGemmQ8WithResidualHalfTiles}. Grid: {@code ceil(d / 64) * ceil(batch / 64)}
     * workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQ8WithScaledResidualHalfTiles(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch,
            float residualScale) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (d + 63) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int blk = TornadoMath.min(r0 + lr, d - 1) * (n >> 5) * 34;
        int q = blk + 2 + kh;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * n + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * d + r0 + ((sg & 1) << 5);
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
            float sc = w.getHalfFloat(blk).getFloat32();
            float a0 = w.get(q + 0) * sc;
            float a1 = w.get(q + 1) * sc;
            float a2 = w.get(q + 2) * sc;
            float a3 = w.get(q + 3) * sc;
            float a4 = w.get(q + 4) * sc;
            float a5 = w.get(q + 5) * sc;
            float a6 = w.get(q + 6) * sc;
            float a7 = w.get(q + 7) * sc;
            float a8 = w.get(q + 8) * sc;
            float a9 = w.get(q + 9) * sc;
            float a10 = w.get(q + 10) * sc;
            float a11 = w.get(q + 11) * sc;
            float a12 = w.get(q + 12) * sc;
            float a13 = w.get(q + 13) * sc;
            float a14 = w.get(q + 14) * sc;
            float a15 = w.get(q + 15) * sc;
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
            blk += 34;
            q += 34;
            xi += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 5);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 16 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 24 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 16 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 24 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 16 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 24 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 16 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 24 + (e & 7);
            if (t < batch && r < d) {
                outputBatch.set(
                        t * d + r, outputBatch.get(t * d + r) + residualScale * cs[(sg << 6) + e]);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmQKVFusedQ8} with the weights staged as FP16, as {@link
     * #batchedGemmQKVQ8HalfTiles}. Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)}
     * workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmQKVFusedQ8HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wqkv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(256);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (qDim + 2 * kvDim) >> 6;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        int width = which == 0 ? qDim : kvDim;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        int blk = (r0 + lr) * (dim >> 5) * 34;
        int qi = blk + 2 + kh;
        int xi = TornadoMath.min(b0 + lr, batch - 1) * dim + kh;
        int cbase = (b0 + ((sg >> 1) << 5)) * width + local0 + ((sg & 1) << 5);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            float sc = wqkv.getHalfFloat(blk).getFloat32();
            a0 = wqkv.get(qi + 0) * sc;
            a1 = wqkv.get(qi + 1) * sc;
            a2 = wqkv.get(qi + 2) * sc;
            a3 = wqkv.get(qi + 3) * sc;
            a4 = wqkv.get(qi + 4) * sc;
            a5 = wqkv.get(qi + 5) * sc;
            a6 = wqkv.get(qi + 6) * sc;
            a7 = wqkv.get(qi + 7) * sc;
            a8 = wqkv.get(qi + 8) * sc;
            a9 = wqkv.get(qi + 9) * sc;
            a10 = wqkv.get(qi + 10) * sc;
            a11 = wqkv.get(qi + 11) * sc;
            a12 = wqkv.get(qi + 12) * sc;
            a13 = wqkv.get(qi + 13) * sc;
            a14 = wqkv.get(qi + 14) * sc;
            a15 = wqkv.get(qi + 15) * sc;
            float y0 = x.get(xi + 0);
            float y1 = x.get(xi + 1);
            float y2 = x.get(xi + 2);
            float y3 = x.get(xi + 3);
            float y4 = x.get(xi + 4);
            float y5 = x.get(xi + 5);
            float y6 = x.get(xi + 6);
            float y7 = x.get(xi + 7);
            float y8 = x.get(xi + 8);
            float y9 = x.get(xi + 9);
            float y10 = x.get(xi + 10);
            float y11 = x.get(xi + 11);
            float y12 = x.get(xi + 12);
            float y13 = x.get(xi + 13);
            float y14 = x.get(xi + 14);
            float y15 = x.get(xi + 15);
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
            blk += 34;
            qi += 34;
            xi += 32;
        }
        if (full) {
            if (which == 0) {
                context.simdgroupMatrixStore(c0, q, cbase + 0 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c1, q, cbase + 0 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c2, q, cbase + 0 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c3, q, cbase + 0 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c4, q, cbase + 8 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c5, q, cbase + 8 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c6, q, cbase + 8 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c7, q, cbase + 8 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c8, q, cbase + 16 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c9, q, cbase + 16 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c10, q, cbase + 16 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c11, q, cbase + 16 * qDim + 24, qDim);
                context.simdgroupMatrixStore(c12, q, cbase + 24 * qDim + 0, qDim);
                context.simdgroupMatrixStore(c13, q, cbase + 24 * qDim + 8, qDim);
                context.simdgroupMatrixStore(c14, q, cbase + 24 * qDim + 16, qDim);
                context.simdgroupMatrixStore(c15, q, cbase + 24 * qDim + 24, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(c0, k, cbase + 0 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c1, k, cbase + 0 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c2, k, cbase + 0 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c3, k, cbase + 0 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c4, k, cbase + 8 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c5, k, cbase + 8 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c6, k, cbase + 8 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c7, k, cbase + 8 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c8, k, cbase + 16 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c9, k, cbase + 16 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c10, k, cbase + 16 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c11, k, cbase + 16 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c12, k, cbase + 24 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c13, k, cbase + 24 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c14, k, cbase + 24 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c15, k, cbase + 24 * kvDim + 24, kvDim);
            } else {
                context.simdgroupMatrixStore(c0, v, cbase + 0 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c1, v, cbase + 0 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c2, v, cbase + 0 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c3, v, cbase + 0 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c4, v, cbase + 8 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c5, v, cbase + 8 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c6, v, cbase + 8 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c7, v, cbase + 8 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c8, v, cbase + 16 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c9, v, cbase + 16 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c10, v, cbase + 16 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c11, v, cbase + 16 * kvDim + 24, kvDim);
                context.simdgroupMatrixStore(c12, v, cbase + 24 * kvDim + 0, kvDim);
                context.simdgroupMatrixStore(c13, v, cbase + 24 * kvDim + 8, kvDim);
                context.simdgroupMatrixStore(c14, v, cbase + 24 * kvDim + 16, kvDim);
                context.simdgroupMatrixStore(c15, v, cbase + 24 * kvDim + 24, kvDim);
            }
        } else {
            int t0 = b0 + ((sg >> 1) << 5);
            int rr = local0 + ((sg & 1) << 5);
            context.localBarrier();
            context.simdgroupMatrixStore(c0, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c1, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c2, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c3, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 0 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c4, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c5, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c6, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c7, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 8 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c8, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c9, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 16 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c12, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c13, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c14, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 16 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c15, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = t0 + 24 + (e >> 3);
                int r = rr + 24 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    if (which == 0) {
                        q.set(t * qDim + r, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + r, val);
                    } else {
                        v.set(t * kvDim + r, val);
                    }
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedGemmRmsNormFFNGateUpFusedQ8} with the weights staged as FP16, as {@link
     * #batchedGemmRmsNormFFNGateUpQ8HalfTiles}. Grid: {@code ceil(hiddenDim / 32) * ceil(batch /
     * 64)} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpFusedQ8HalfTiles(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray wUp,
            int dim,
            int hiddenDim,
            int batch) {
        HalfFloat[] sa = context.allocateHalfFloatLocalArray(2048);
        float[] sb = context.allocateFloatLocalArray(2048);
        float[] cs = context.allocateFloatLocalArray(512);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int rowTiles = (hiddenDim + 31) >> 5;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        int lr = tid >> 1;
        int kh = (tid & 1) << 4;
        int aBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + (lr & 7);
        int bBase = ((((tid & 1) << 4) + (lr >> 3)) << 6) + ((lr & 7) << 3);
        int ca = (sg & 1) << 8;
        int cb = (sg >> 1) << 8;
        boolean up = ((lr >> 4) & 1) == 1;
        int row = TornadoMath.min(r0 + ((lr >> 5) << 4) + (lr & 15), hiddenDim - 1);
        int blk = (up ? hiddenDim + row : row) * (dim >> 5) * 34;
        int qi = blk + 2 + kh;
        int tok = TornadoMath.min(b0 + lr, batch - 1);
        int xi = tok * dim + kh;
        int ri = kh;
        float xsc = scaleBatch.get(tok);
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
        for (int k0 = 0; k0 < dim; k0 += 32) {
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
            float sc = wUp.getHalfFloat(blk).getFloat32();
            a0 = wUp.get(qi + 0) * sc;
            a1 = wUp.get(qi + 1) * sc;
            a2 = wUp.get(qi + 2) * sc;
            a3 = wUp.get(qi + 3) * sc;
            a4 = wUp.get(qi + 4) * sc;
            a5 = wUp.get(qi + 5) * sc;
            a6 = wUp.get(qi + 6) * sc;
            a7 = wUp.get(qi + 7) * sc;
            a8 = wUp.get(qi + 8) * sc;
            a9 = wUp.get(qi + 9) * sc;
            a10 = wUp.get(qi + 10) * sc;
            a11 = wUp.get(qi + 11) * sc;
            a12 = wUp.get(qi + 12) * sc;
            a13 = wUp.get(qi + 13) * sc;
            a14 = wUp.get(qi + 14) * sc;
            a15 = wUp.get(qi + 15) * sc;
            float y0 = x.get(xi + 0) * rmsWeights.get(ri + 0) * xsc;
            float y1 = x.get(xi + 1) * rmsWeights.get(ri + 1) * xsc;
            float y2 = x.get(xi + 2) * rmsWeights.get(ri + 2) * xsc;
            float y3 = x.get(xi + 3) * rmsWeights.get(ri + 3) * xsc;
            float y4 = x.get(xi + 4) * rmsWeights.get(ri + 4) * xsc;
            float y5 = x.get(xi + 5) * rmsWeights.get(ri + 5) * xsc;
            float y6 = x.get(xi + 6) * rmsWeights.get(ri + 6) * xsc;
            float y7 = x.get(xi + 7) * rmsWeights.get(ri + 7) * xsc;
            float y8 = x.get(xi + 8) * rmsWeights.get(ri + 8) * xsc;
            float y9 = x.get(xi + 9) * rmsWeights.get(ri + 9) * xsc;
            float y10 = x.get(xi + 10) * rmsWeights.get(ri + 10) * xsc;
            float y11 = x.get(xi + 11) * rmsWeights.get(ri + 11) * xsc;
            float y12 = x.get(xi + 12) * rmsWeights.get(ri + 12) * xsc;
            float y13 = x.get(xi + 13) * rmsWeights.get(ri + 13) * xsc;
            float y14 = x.get(xi + 14) * rmsWeights.get(ri + 14) * xsc;
            float y15 = x.get(xi + 15) * rmsWeights.get(ri + 15) * xsc;
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
            blk += 34;
            qi += 34;
            xi += 32;
            ri += 32;
        }
        int t0 = b0 + ((sg >> 1) << 5);
        int rr = r0 + ((sg & 1) << 4);
        context.localBarrier();
        context.simdgroupMatrixStore(c0, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c2, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c1, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c3, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 0 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c4, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c6, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c5, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c7, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 8 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c8, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c9, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c11, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 16 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c12, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c14, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(c13, cs, sg << 7, 8);
        context.simdgroupMatrixStore(c15, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = t0 + 24 + (e >> 3);
            int r = rr + 8 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
    }
}
