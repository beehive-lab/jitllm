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
}
