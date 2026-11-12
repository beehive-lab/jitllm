package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;

// @formatter:off
/**
 * Batched-prefill attention over the paged FP16 key/value cache on Apple SIMD-group matrices, one
 * kernel per head width (64, 96 and 128): the kernels of {@link
 * TransformerPagedKvBatchPrefillSimdgroupKernels} with the cache read as {@code half}.
 *
 * <p>Key fragments are read from the cache by transposed {@code simdgroup_half8x8} loads and value
 * fragments by plain ones; each is multiplied with a {@code float} fragment (the staged queries,
 * the probabilities) and accumulated in {@code float}, so the only rounding the FP16 cache adds is
 * the one made when the value was stored. Everything else is as in the FP32 kernels: the online
 * softmax in threadgroup memory, the causal mask applied there, the diagonal rescaling, and the key
 * fragment that reaches past the last position written by this batch reading its values through a
 * zero-filled {@code float} buffer, so unwritten cache memory never meets a zero weight.
 */
// @formatter:on
public final class TransformerPagedKvBatchPrefillSimdgroupFP16Kernels {

    /** Threads per workgroup: four SIMD groups. */
    public static final int THREADS = TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS;

    /** Query tokens per workgroup. */
    public static final int QUERY_TILE = TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE;

    private TransformerPagedKvBatchPrefillSimdgroupFP16Kernels() {}

    // @formatter:off
    /**
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 64-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 64.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedHead64(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        float[] qs = context.allocateFloatLocalArray(2048);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(512);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = 1.0f / TornadoMath.sqrt(headSize);
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 2048; e += 128) {
            int r = e / 64;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 64);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 64; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 64 + d, 64);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp0 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp1 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp2 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp3 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 96-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 96.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedHead96(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        float[] qs = context.allocateFloatLocalArray(3072);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(768);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = 1.0f / TornadoMath.sqrt(headSize);
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 3072; e += 128) {
            int r = e / 96;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 96);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        Matrix8x8Float o8 = context.simdgroupMatrixZero();
        Matrix8x8Float o9 = context.simdgroupMatrixZero();
        Matrix8x8Float o10 = context.simdgroupMatrixZero();
        Matrix8x8Float o11 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 96; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 96 + d, 96);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            o8 = context.simdgroupMatrixMultiplyAccumulate(da, o8, zero);
            o9 = context.simdgroupMatrixMultiplyAccumulate(da, o9, zero);
            o10 = context.simdgroupMatrixMultiplyAccumulate(da, o10, zero);
            o11 = context.simdgroupMatrixMultiplyAccumulate(da, o11, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp0 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp1 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp2 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp3 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 88, kvDim),
                                o11);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        o8 = context.simdgroupMatrixMultiplyAccumulate(dn, o8, zero);
        o9 = context.simdgroupMatrixMultiplyAccumulate(dn, o9, zero);
        o10 = context.simdgroupMatrixMultiplyAccumulate(dn, o10, zero);
        o11 = context.simdgroupMatrixMultiplyAccumulate(dn, o11, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
            context.simdgroupMatrixStore(o8, wrapXbBatch, outBase + 64, dim);
            context.simdgroupMatrixStore(o9, wrapXbBatch, outBase + 72, dim);
            context.simdgroupMatrixStore(o10, wrapXbBatch, outBase + 80, dim);
            context.simdgroupMatrixStore(o11, wrapXbBatch, outBase + 88, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o8, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 64 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o9, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 72 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o10, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 80 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o11, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 88 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 128-wide head,
     * on SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 128.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedHead128(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        float[] qs = context.allocateFloatLocalArray(4096);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(1024);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = 1.0f / TornadoMath.sqrt(headSize);
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 4096; e += 128) {
            int r = e / 128;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 128);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        Matrix8x8Float o8 = context.simdgroupMatrixZero();
        Matrix8x8Float o9 = context.simdgroupMatrixZero();
        Matrix8x8Float o10 = context.simdgroupMatrixZero();
        Matrix8x8Float o11 = context.simdgroupMatrixZero();
        Matrix8x8Float o12 = context.simdgroupMatrixZero();
        Matrix8x8Float o13 = context.simdgroupMatrixZero();
        Matrix8x8Float o14 = context.simdgroupMatrixZero();
        Matrix8x8Float o15 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 128; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 128 + d, 128);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            o8 = context.simdgroupMatrixMultiplyAccumulate(da, o8, zero);
            o9 = context.simdgroupMatrixMultiplyAccumulate(da, o9, zero);
            o10 = context.simdgroupMatrixMultiplyAccumulate(da, o10, zero);
            o11 = context.simdgroupMatrixMultiplyAccumulate(da, o11, zero);
            o12 = context.simdgroupMatrixMultiplyAccumulate(da, o12, zero);
            o13 = context.simdgroupMatrixMultiplyAccumulate(da, o13, zero);
            o14 = context.simdgroupMatrixMultiplyAccumulate(da, o14, zero);
            o15 = context.simdgroupMatrixMultiplyAccumulate(da, o15, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp0 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp1 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp2 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp3 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 120, kvDim),
                                o15);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        o8 = context.simdgroupMatrixMultiplyAccumulate(dn, o8, zero);
        o9 = context.simdgroupMatrixMultiplyAccumulate(dn, o9, zero);
        o10 = context.simdgroupMatrixMultiplyAccumulate(dn, o10, zero);
        o11 = context.simdgroupMatrixMultiplyAccumulate(dn, o11, zero);
        o12 = context.simdgroupMatrixMultiplyAccumulate(dn, o12, zero);
        o13 = context.simdgroupMatrixMultiplyAccumulate(dn, o13, zero);
        o14 = context.simdgroupMatrixMultiplyAccumulate(dn, o14, zero);
        o15 = context.simdgroupMatrixMultiplyAccumulate(dn, o15, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
            context.simdgroupMatrixStore(o8, wrapXbBatch, outBase + 64, dim);
            context.simdgroupMatrixStore(o9, wrapXbBatch, outBase + 72, dim);
            context.simdgroupMatrixStore(o10, wrapXbBatch, outBase + 80, dim);
            context.simdgroupMatrixStore(o11, wrapXbBatch, outBase + 88, dim);
            context.simdgroupMatrixStore(o12, wrapXbBatch, outBase + 96, dim);
            context.simdgroupMatrixStore(o13, wrapXbBatch, outBase + 104, dim);
            context.simdgroupMatrixStore(o14, wrapXbBatch, outBase + 112, dim);
            context.simdgroupMatrixStore(o15, wrapXbBatch, outBase + 120, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o8, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 64 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o9, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 72 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o10, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 80 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o11, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 88 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o12, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 96 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o13, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 104 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o14, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 112 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o15, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 120 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedFlashAttentionKVFP16PagedHead64} with the scores scaled by {@code
     * attentionScale} instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier).
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 64-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 64.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedScaledHead64(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim,
            float attentionScale) {
        float[] qs = context.allocateFloatLocalArray(2048);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(512);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = attentionScale;
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 2048; e += 128) {
            int r = e / 64;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 64);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 64; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 64 + d, 64);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp0 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp1 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp2 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 512; e += 128) {
                    int kp = kp3 + e / 64;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 64)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 64), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 64), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 64), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 64), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 64), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 64), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 64), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 64), o7);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedFlashAttentionKVFP16PagedHead96} with the scores scaled by {@code
     * attentionScale} instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier).
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 96-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 96.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedScaledHead96(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim,
            float attentionScale) {
        float[] qs = context.allocateFloatLocalArray(3072);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(768);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = attentionScale;
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 3072; e += 128) {
            int r = e / 96;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 96);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        Matrix8x8Float o8 = context.simdgroupMatrixZero();
        Matrix8x8Float o9 = context.simdgroupMatrixZero();
        Matrix8x8Float o10 = context.simdgroupMatrixZero();
        Matrix8x8Float o11 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 96; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 96 + d, 96);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            o8 = context.simdgroupMatrixMultiplyAccumulate(da, o8, zero);
            o9 = context.simdgroupMatrixMultiplyAccumulate(da, o9, zero);
            o10 = context.simdgroupMatrixMultiplyAccumulate(da, o10, zero);
            o11 = context.simdgroupMatrixMultiplyAccumulate(da, o11, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp0 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp1 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp2 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 88, kvDim),
                                o11);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 768; e += 128) {
                    int kp = kp3 + e / 96;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 96)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 96), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 96), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 96), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 96), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 96), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 96), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 96), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 96), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 64, 96), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 72, 96), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 80, 96), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 88, 96), o11);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 88, kvDim),
                                o11);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        o8 = context.simdgroupMatrixMultiplyAccumulate(dn, o8, zero);
        o9 = context.simdgroupMatrixMultiplyAccumulate(dn, o9, zero);
        o10 = context.simdgroupMatrixMultiplyAccumulate(dn, o10, zero);
        o11 = context.simdgroupMatrixMultiplyAccumulate(dn, o11, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
            context.simdgroupMatrixStore(o8, wrapXbBatch, outBase + 64, dim);
            context.simdgroupMatrixStore(o9, wrapXbBatch, outBase + 72, dim);
            context.simdgroupMatrixStore(o10, wrapXbBatch, outBase + 80, dim);
            context.simdgroupMatrixStore(o11, wrapXbBatch, outBase + 88, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o8, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 64 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o9, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 72 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o10, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 80 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o11, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 88 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link #batchedFlashAttentionKVFP16PagedHead128} with the scores scaled by {@code
     * attentionScale} instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier).
     * Causal attention of a batch of query tokens over the paged FP16 cache, for a 128-wide head,
     * on SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 128.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedScaledHead128(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim,
            float attentionScale) {
        float[] qs = context.allocateFloatLocalArray(4096);
        float[] ss = context.allocateFloatLocalArray(4 * 256);
        float[] ds = context.allocateFloatLocalArray(4 * 64);
        float[] vt = context.allocateFloatLocalArray(1024);
        float[] rmax = context.allocateFloatLocalArray(4 * 32);
        float[] rsum = context.allocateFloatLocalArray(4 * 32);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 5;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = attentionScale;
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 31, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 4096; e += 128) {
            int r = e / 128;
            qs[e] =
                    wrapQBatch.get(
                            TornadoMath.min(t0 + r, active - 1) * dim + h * headSize + e % 128);
        }
        for (int e = tid; e < 256; e += 128) {
            ds[e] = 0.0f;
        }
        context.localBarrier();
        int sgq = sg << 3;
        // Rows past the batch take the last active row's limit: the diagonal rescaling multiplies
        // every
        // row by zero for the others, so a row that read unwritten cache would poison the valid
        // ones.
        int row = lane >> 2;
        int sub = lane & 3;
        int myPos = start + TornadoMath.min(t0 + sgq + row, active - 1);
        int sOff = sg << 8;
        int dOff = sg << 6;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        Matrix8x8Float o8 = context.simdgroupMatrixZero();
        Matrix8x8Float o9 = context.simdgroupMatrixZero();
        Matrix8x8Float o10 = context.simdgroupMatrixZero();
        Matrix8x8Float o11 = context.simdgroupMatrixZero();
        Matrix8x8Float o12 = context.simdgroupMatrixZero();
        Matrix8x8Float o13 = context.simdgroupMatrixZero();
        Matrix8x8Float o14 = context.simdgroupMatrixZero();
        Matrix8x8Float o15 = context.simdgroupMatrixZero();
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 32) {
            int kp0 = kb0 + 0;
            int base0 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp0, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp1 = kb0 + 8;
            int base1 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp1, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp2 = kb0 + 16;
            int base2 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp2, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int kp3 = kb0 + 24;
            int base3 =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kp3, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            Matrix8x8Float s1 = context.simdgroupMatrixZero();
            Matrix8x8Float s2 = context.simdgroupMatrixZero();
            Matrix8x8Float s3 = context.simdgroupMatrixZero();
            for (int d = 0; d < 128; d += 8) {
                Matrix8x8Float qf = context.simdgroupMatrixLoad(qs, sgq * 128 + d, 128);
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base0 + d, kvDim),
                                s0);
                s1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base1 + d, kvDim),
                                s1);
                s2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base2 + d, kvDim),
                                s2);
                s3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                qf,
                                context.simdgroupMatrixLoadTransposed(
                                        wrapKeyCache, base3 + d, kvDim),
                                s3);
            }
            context.simdgroupMatrixStore(s0, ss, sOff + 0, 32);
            context.simdgroupMatrixStore(s1, ss, sOff + 8, 32);
            context.simdgroupMatrixStore(s2, ss, sOff + 16, 32);
            context.simdgroupMatrixStore(s3, ss, sOff + 24, 32);
            context.localBarrier();
            int rowOff = sOff + (row << 5) + (sub << 3);
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 8; j++) {
                if (kb0 + (sub << 3) + j <= myPos) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j] * scale);
                }
            }
            rmax[(sg << 5) + lane] = pmx;
            context.localBarrier();
            int rq = (sg << 5) + (row << 2);
            float mx =
                    TornadoMath.max(
                            TornadoMath.max(m, TornadoMath.max(rmax[rq], rmax[rq + 1])),
                            TornadoMath.max(rmax[rq + 2], rmax[rq + 3]));
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float p = 0.0f;
                if (kb0 + (sub << 3) + j <= myPos) {
                    p = TornadoMath.exp(ss[rowOff + j] * scale - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[(sg << 5) + lane] = psum;
            context.localBarrier();
            l = l * alpha + ((rsum[rq] + rsum[rq + 1]) + (rsum[rq + 2] + rsum[rq + 3]));
            m = mx;
            if (sub == 0) {
                ds[dOff + row * 9] = alpha;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, dOff, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            o8 = context.simdgroupMatrixMultiplyAccumulate(da, o8, zero);
            o9 = context.simdgroupMatrixMultiplyAccumulate(da, o9, zero);
            o10 = context.simdgroupMatrixMultiplyAccumulate(da, o10, zero);
            o11 = context.simdgroupMatrixMultiplyAccumulate(da, o11, zero);
            o12 = context.simdgroupMatrixMultiplyAccumulate(da, o12, zero);
            o13 = context.simdgroupMatrixMultiplyAccumulate(da, o13, zero);
            o14 = context.simdgroupMatrixMultiplyAccumulate(da, o14, zero);
            o15 = context.simdgroupMatrixMultiplyAccumulate(da, o15, zero);
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, sOff + 0, 32);
            if (kp0 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp0 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(wrapValueCache, base0 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, sOff + 8, 32);
            if (kp1 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp1 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(wrapValueCache, base1 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, sOff + 16, 32);
            if (kp2 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp2 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(wrapValueCache, base2 + 120, kvDim),
                                o15);
            }
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, sOff + 24, 32);
            if (kp3 + 7 > lastValid) {
                for (int e = tid; e < 1024; e += 128) {
                    int kp = kp3 + e / 128;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv =
                                wrapValueCache
                                        .get(
                                                KvBlockAddress.offset(
                                                                blockTable,
                                                                slot,
                                                                kp,
                                                                layerOff,
                                                                kvDim,
                                                                blockCfg,
                                                                blockStride)
                                                        + headOff
                                                        + e % 128)
                                        .getFloat32();
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 0, 128), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 8, 128), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 16, 128), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 24, 128), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 32, 128), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 40, 128), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 48, 128), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 56, 128), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 64, 128), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 72, 128), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 80, 128), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 88, 128), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 96, 128), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 104, 128), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 112, 128), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, 120, 128), o15);
                context.localBarrier();
            } else {
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 0, kvDim),
                                o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 8, kvDim),
                                o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(wrapValueCache, base3 + 120, kvDim),
                                o15);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[dOff + row * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, dOff, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        o8 = context.simdgroupMatrixMultiplyAccumulate(dn, o8, zero);
        o9 = context.simdgroupMatrixMultiplyAccumulate(dn, o9, zero);
        o10 = context.simdgroupMatrixMultiplyAccumulate(dn, o10, zero);
        o11 = context.simdgroupMatrixMultiplyAccumulate(dn, o11, zero);
        o12 = context.simdgroupMatrixMultiplyAccumulate(dn, o12, zero);
        o13 = context.simdgroupMatrixMultiplyAccumulate(dn, o13, zero);
        o14 = context.simdgroupMatrixMultiplyAccumulate(dn, o14, zero);
        o15 = context.simdgroupMatrixMultiplyAccumulate(dn, o15, zero);
        int outBase = (t0 + sgq) * dim + h * headSize;
        if (t0 + 32 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
            context.simdgroupMatrixStore(o4, wrapXbBatch, outBase + 32, dim);
            context.simdgroupMatrixStore(o5, wrapXbBatch, outBase + 40, dim);
            context.simdgroupMatrixStore(o6, wrapXbBatch, outBase + 48, dim);
            context.simdgroupMatrixStore(o7, wrapXbBatch, outBase + 56, dim);
            context.simdgroupMatrixStore(o8, wrapXbBatch, outBase + 64, dim);
            context.simdgroupMatrixStore(o9, wrapXbBatch, outBase + 72, dim);
            context.simdgroupMatrixStore(o10, wrapXbBatch, outBase + 80, dim);
            context.simdgroupMatrixStore(o11, wrapXbBatch, outBase + 88, dim);
            context.simdgroupMatrixStore(o12, wrapXbBatch, outBase + 96, dim);
            context.simdgroupMatrixStore(o13, wrapXbBatch, outBase + 104, dim);
            context.simdgroupMatrixStore(o14, wrapXbBatch, outBase + 112, dim);
            context.simdgroupMatrixStore(o15, wrapXbBatch, outBase + 120, dim);
        } else {
            context.simdgroupMatrixStore(o0, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 0 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o1, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 8 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o2, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 16 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o3, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 24 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o4, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 32 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o5, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 40 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o6, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 48 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o7, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 56 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o8, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 64 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o9, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 72 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o10, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 80 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o11, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 88 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o12, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 96 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o13, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 104 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o14, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 112 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(o15, ss, sOff, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                if (t0 + sgq + (e >> 3) < active) {
                    wrapXbBatch.set(outBase + (e >> 3) * dim + 120 + (e & 7), ss[sOff + e]);
                }
            }
            context.localBarrier();
        }
    }

    /** Queries per workgroup of {@link #batchedFlashAttentionKVFP16PagedHead128Rows8}. */
    public static final int ROWS8_QUERY_TILE = 8;

    // @formatter:off
    /**
     * {@link #batchedFlashAttentionKVFP16PagedHead128} with 8 queries per workgroup instead of 32.
     * Each step takes 64 keys: every SIMD group scores 16 of them, the softmax runs in SIMD-group
     * sums and shuffles, and every SIMD group adds 32 of the output dimensions. Three barriers per
     * step. Grid: {@code ceil(batch / ROWS8_QUERY_TILE) * nHeads} workgroups of 128 threads.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedHead128Rows8(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        float[] qs = context.allocateFloatLocalArray(1024);
        float[] ss = context.allocateFloatLocalArray(512);
        float[] dg = context.allocateFloatLocalArray(64);
        float[] vt = context.allocateFloatLocalArray(1024);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 3;
        if (t0 >= active) {
            return;
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = 1.0f / TornadoMath.sqrt(headSize);
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 7, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 1024; e += 128) {
            qs[e] =
                    wrapQBatch.get(
                                    TornadoMath.min(t0 + (e >> 7), active - 1) * dim
                                            + h * headSize
                                            + (e & 127))
                            * scale;
        }
        if (tid < 64) {
            dg[tid] = 0.0f;
        }
        context.localBarrier();
        int ra = sg;
        int rb = sg + 4;
        int posA = start + TornadoMath.min(t0 + ra, active - 1);
        int posB = start + TornadoMath.min(t0 + rb, active - 1);
        float mA = Float.NEGATIVE_INFINITY;
        float lA = 0.0f;
        float mB = Float.NEGATIVE_INFINITY;
        float lB = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        int dOff = sg << 5;
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 64) {
            int kpA = kb0 + (sg << 4);
            int kpB = kpA + 8;
            int baseA =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kpA, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int baseB =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kpB, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float sA = context.simdgroupMatrixZero();
            Matrix8x8Float sB = context.simdgroupMatrixZero();
            Matrix8x8Float q0 = context.simdgroupMatrixLoad(qs, 0, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q0,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 0, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q0,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 0, kvDim),
                            sB);
            Matrix8x8Float q8 = context.simdgroupMatrixLoad(qs, 8, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q8,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 8, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q8,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 8, kvDim),
                            sB);
            Matrix8x8Float q16 = context.simdgroupMatrixLoad(qs, 16, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q16,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 16, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q16,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 16, kvDim),
                            sB);
            Matrix8x8Float q24 = context.simdgroupMatrixLoad(qs, 24, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q24,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 24, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q24,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 24, kvDim),
                            sB);
            Matrix8x8Float q32 = context.simdgroupMatrixLoad(qs, 32, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q32,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 32, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q32,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 32, kvDim),
                            sB);
            Matrix8x8Float q40 = context.simdgroupMatrixLoad(qs, 40, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q40,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 40, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q40,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 40, kvDim),
                            sB);
            Matrix8x8Float q48 = context.simdgroupMatrixLoad(qs, 48, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q48,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 48, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q48,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 48, kvDim),
                            sB);
            Matrix8x8Float q56 = context.simdgroupMatrixLoad(qs, 56, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q56,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 56, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q56,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 56, kvDim),
                            sB);
            Matrix8x8Float q64 = context.simdgroupMatrixLoad(qs, 64, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q64,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 64, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q64,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 64, kvDim),
                            sB);
            Matrix8x8Float q72 = context.simdgroupMatrixLoad(qs, 72, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q72,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 72, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q72,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 72, kvDim),
                            sB);
            Matrix8x8Float q80 = context.simdgroupMatrixLoad(qs, 80, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q80,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 80, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q80,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 80, kvDim),
                            sB);
            Matrix8x8Float q88 = context.simdgroupMatrixLoad(qs, 88, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q88,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 88, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q88,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 88, kvDim),
                            sB);
            Matrix8x8Float q96 = context.simdgroupMatrixLoad(qs, 96, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q96,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 96, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q96,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 96, kvDim),
                            sB);
            Matrix8x8Float q104 = context.simdgroupMatrixLoad(qs, 104, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q104,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 104, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q104,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 104, kvDim),
                            sB);
            Matrix8x8Float q112 = context.simdgroupMatrixLoad(qs, 112, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q112,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 112, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q112,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 112, kvDim),
                            sB);
            Matrix8x8Float q120 = context.simdgroupMatrixLoad(qs, 120, 128);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q120,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 120, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q120,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 120, kvDim),
                            sB);
            context.simdgroupMatrixStore(sA, ss, sg << 4, 64);
            context.simdgroupMatrixStore(sB, ss, (sg << 4) + 8, 64);
            context.localBarrier();
            float x1A = kb0 + lane <= posA ? ss[(ra << 6) + lane] : Float.NEGATIVE_INFINITY;
            float x2A =
                    kb0 + lane + 32 <= posA ? ss[(ra << 6) + lane + 32] : Float.NEGATIVE_INFINITY;
            float mxA = TornadoMath.max(x1A, x2A);
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 16));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 8));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 4));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 2));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 1));
            mxA = TornadoMath.max(mA, context.simdBroadcastFirst(mxA));
            float alA = mA == Float.NEGATIVE_INFINITY ? 0.0f : TornadoMath.exp(mA - mxA);
            float p1A = kb0 + lane <= posA ? TornadoMath.exp(x1A - mxA) : 0.0f;
            float p2A = kb0 + lane + 32 <= posA ? TornadoMath.exp(x2A - mxA) : 0.0f;
            lA = lA * alA + context.simdSum(p1A + p2A);
            mA = mxA;
            ss[(ra << 6) + lane] = p1A;
            ss[(ra << 6) + lane + 32] = p2A;
            float x1B = kb0 + lane <= posB ? ss[(rb << 6) + lane] : Float.NEGATIVE_INFINITY;
            float x2B =
                    kb0 + lane + 32 <= posB ? ss[(rb << 6) + lane + 32] : Float.NEGATIVE_INFINITY;
            float mxB = TornadoMath.max(x1B, x2B);
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 16));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 8));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 4));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 2));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 1));
            mxB = TornadoMath.max(mB, context.simdBroadcastFirst(mxB));
            float alB = mB == Float.NEGATIVE_INFINITY ? 0.0f : TornadoMath.exp(mB - mxB);
            float p1B = kb0 + lane <= posB ? TornadoMath.exp(x1B - mxB) : 0.0f;
            float p2B = kb0 + lane + 32 <= posB ? TornadoMath.exp(x2B - mxB) : 0.0f;
            lB = lB * alB + context.simdSum(p1B + p2B);
            mB = mxB;
            ss[(rb << 6) + lane] = p1B;
            ss[(rb << 6) + lane + 32] = p2B;
            if (lane == 0) {
                dg[ra * 9] = alA;
                dg[rb * 9] = alB;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(dg, 0, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            int kp0 = kb0 + 0;
            if (kp0 <= maxPos) {
                Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, 0, 64);
                if (kp0 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp0 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb0 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp0,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 24, kvDim),
                                    o3);
                }
            }
            int kp1 = kb0 + 8;
            if (kp1 <= maxPos) {
                Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, 8, 64);
                if (kp1 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp1 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb1 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp1,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 24, kvDim),
                                    o3);
                }
            }
            int kp2 = kb0 + 16;
            if (kp2 <= maxPos) {
                Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, 16, 64);
                if (kp2 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp2 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb2 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp2,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 24, kvDim),
                                    o3);
                }
            }
            int kp3 = kb0 + 24;
            if (kp3 <= maxPos) {
                Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, 24, 64);
                if (kp3 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp3 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb3 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp3,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 24, kvDim),
                                    o3);
                }
            }
            int kp4 = kb0 + 32;
            if (kp4 <= maxPos) {
                Matrix8x8Float p4 = context.simdgroupMatrixLoad(ss, 32, 64);
                if (kp4 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp4 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb4 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp4,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 24, kvDim),
                                    o3);
                }
            }
            int kp5 = kb0 + 40;
            if (kp5 <= maxPos) {
                Matrix8x8Float p5 = context.simdgroupMatrixLoad(ss, 40, 64);
                if (kp5 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp5 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb5 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp5,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 24, kvDim),
                                    o3);
                }
            }
            int kp6 = kb0 + 48;
            if (kp6 <= maxPos) {
                Matrix8x8Float p6 = context.simdgroupMatrixLoad(ss, 48, 64);
                if (kp6 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp6 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb6 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp6,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 24, kvDim),
                                    o3);
                }
            }
            int kp7 = kb0 + 56;
            if (kp7 <= maxPos) {
                Matrix8x8Float p7 = context.simdgroupMatrixLoad(ss, 56, 64);
                if (kp7 + 7 > lastValid) {
                    for (int e = lane; e < 256; e += 32) {
                        int kp = kp7 + (e >> 5);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e & 31))
                                            .getFloat32();
                        }
                        vt[(sg << 8) + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, (sg << 8) + 0, 32), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, (sg << 8) + 8, 32), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, (sg << 8) + 16, 32), o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, (sg << 8) + 24, 32), o3);
                    context.localBarrier();
                } else {
                    int vb7 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp7,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 16, kvDim),
                                    o2);
                    o3 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 24, kvDim),
                                    o3);
                }
            }
            context.localBarrier();
        }
        if (lane == 0) {
            dg[ra * 9] = lA > 0.0f ? 1.0f / lA : 0.0f;
            dg[rb * 9] = lB > 0.0f ? 1.0f / lB : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(dg, 0, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        int outBase = t0 * dim + h * headSize + dOff;
        if (t0 + 8 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
            context.simdgroupMatrixStore(o3, wrapXbBatch, outBase + 24, dim);
        } else {
            context.simdgroupMatrixStore(o0, vt, (sg << 8) + 0, 8);
            context.simdgroupMatrixStore(o1, vt, (sg << 8) + 64, 8);
            context.simdgroupMatrixStore(o2, vt, (sg << 8) + 128, 8);
            context.simdgroupMatrixStore(o3, vt, (sg << 8) + 192, 8);
            context.localBarrier();
            for (int e = lane; e < 256; e += 32) {
                int r = (e & 63) >> 3;
                if (t0 + r < active) {
                    wrapXbBatch.set(
                            outBase + r * dim + ((e >> 6) << 3) + (e & 7), vt[(sg << 8) + e]);
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link #batchedFlashAttentionKVFP16PagedHead128Rows8} for heads 96 wide: each SIMD group adds
     * 24 of the output dimensions. Grid: {@code ceil(batch / ROWS8_QUERY_TILE) * nHeads} workgroups
     * of 128 threads.
     */
    // @formatter:on
    public static void batchedFlashAttentionKVFP16PagedHead96Rows8(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        float[] qs = context.allocateFloatLocalArray(768);
        float[] ss = context.allocateFloatLocalArray(512);
        float[] dg = context.allocateFloatLocalArray(64);
        float[] vt = context.allocateFloatLocalArray(768);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = batchStartPosHolder.get(1);
        int t0 = tile << 3;
        if (t0 >= active) {
            return;
        }
        int start = batchStartPosHolder.get(0);
        int slot = batchStartPosHolder.get(2);
        int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
        int headOff = (h / kvMul) * headSize;
        float scale = 1.0f / TornadoMath.sqrt(headSize);
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 7, active - 1);
        int lastGroup = maxPos & ~7;
        for (int e = tid; e < 768; e += 128) {
            qs[e] =
                    wrapQBatch.get(
                                    TornadoMath.min(t0 + (e / 96), active - 1) * dim
                                            + h * headSize
                                            + (e % 96))
                            * scale;
        }
        if (tid < 64) {
            dg[tid] = 0.0f;
        }
        context.localBarrier();
        int ra = sg;
        int rb = sg + 4;
        int posA = start + TornadoMath.min(t0 + ra, active - 1);
        int posB = start + TornadoMath.min(t0 + rb, active - 1);
        float mA = Float.NEGATIVE_INFINITY;
        float lA = 0.0f;
        float mB = Float.NEGATIVE_INFINITY;
        float lB = 0.0f;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        int dOff = sg * 24;
        for (int kb0 = 0; kb0 <= maxPos; kb0 += 64) {
            int kpA = kb0 + (sg << 4);
            int kpB = kpA + 8;
            int baseA =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kpA, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            int baseB =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    TornadoMath.min(kpB, lastGroup),
                                    layerOff,
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + headOff;
            Matrix8x8Float sA = context.simdgroupMatrixZero();
            Matrix8x8Float sB = context.simdgroupMatrixZero();
            Matrix8x8Float q0 = context.simdgroupMatrixLoad(qs, 0, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q0,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 0, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q0,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 0, kvDim),
                            sB);
            Matrix8x8Float q8 = context.simdgroupMatrixLoad(qs, 8, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q8,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 8, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q8,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 8, kvDim),
                            sB);
            Matrix8x8Float q16 = context.simdgroupMatrixLoad(qs, 16, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q16,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 16, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q16,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 16, kvDim),
                            sB);
            Matrix8x8Float q24 = context.simdgroupMatrixLoad(qs, 24, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q24,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 24, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q24,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 24, kvDim),
                            sB);
            Matrix8x8Float q32 = context.simdgroupMatrixLoad(qs, 32, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q32,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 32, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q32,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 32, kvDim),
                            sB);
            Matrix8x8Float q40 = context.simdgroupMatrixLoad(qs, 40, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q40,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 40, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q40,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 40, kvDim),
                            sB);
            Matrix8x8Float q48 = context.simdgroupMatrixLoad(qs, 48, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q48,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 48, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q48,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 48, kvDim),
                            sB);
            Matrix8x8Float q56 = context.simdgroupMatrixLoad(qs, 56, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q56,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 56, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q56,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 56, kvDim),
                            sB);
            Matrix8x8Float q64 = context.simdgroupMatrixLoad(qs, 64, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q64,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 64, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q64,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 64, kvDim),
                            sB);
            Matrix8x8Float q72 = context.simdgroupMatrixLoad(qs, 72, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q72,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 72, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q72,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 72, kvDim),
                            sB);
            Matrix8x8Float q80 = context.simdgroupMatrixLoad(qs, 80, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q80,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 80, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q80,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 80, kvDim),
                            sB);
            Matrix8x8Float q88 = context.simdgroupMatrixLoad(qs, 88, 96);
            sA =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q88,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseA + 88, kvDim),
                            sA);
            sB =
                    context.simdgroupMatrixMultiplyAccumulate(
                            q88,
                            context.simdgroupMatrixLoadTransposed(wrapKeyCache, baseB + 88, kvDim),
                            sB);
            context.simdgroupMatrixStore(sA, ss, sg << 4, 64);
            context.simdgroupMatrixStore(sB, ss, (sg << 4) + 8, 64);
            context.localBarrier();
            float x1A = kb0 + lane <= posA ? ss[(ra << 6) + lane] : Float.NEGATIVE_INFINITY;
            float x2A =
                    kb0 + lane + 32 <= posA ? ss[(ra << 6) + lane + 32] : Float.NEGATIVE_INFINITY;
            float mxA = TornadoMath.max(x1A, x2A);
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 16));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 8));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 4));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 2));
            mxA = TornadoMath.max(mxA, context.simdShuffleDown(mxA, 1));
            mxA = TornadoMath.max(mA, context.simdBroadcastFirst(mxA));
            float alA = mA == Float.NEGATIVE_INFINITY ? 0.0f : TornadoMath.exp(mA - mxA);
            float p1A = kb0 + lane <= posA ? TornadoMath.exp(x1A - mxA) : 0.0f;
            float p2A = kb0 + lane + 32 <= posA ? TornadoMath.exp(x2A - mxA) : 0.0f;
            lA = lA * alA + context.simdSum(p1A + p2A);
            mA = mxA;
            ss[(ra << 6) + lane] = p1A;
            ss[(ra << 6) + lane + 32] = p2A;
            float x1B = kb0 + lane <= posB ? ss[(rb << 6) + lane] : Float.NEGATIVE_INFINITY;
            float x2B =
                    kb0 + lane + 32 <= posB ? ss[(rb << 6) + lane + 32] : Float.NEGATIVE_INFINITY;
            float mxB = TornadoMath.max(x1B, x2B);
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 16));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 8));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 4));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 2));
            mxB = TornadoMath.max(mxB, context.simdShuffleDown(mxB, 1));
            mxB = TornadoMath.max(mB, context.simdBroadcastFirst(mxB));
            float alB = mB == Float.NEGATIVE_INFINITY ? 0.0f : TornadoMath.exp(mB - mxB);
            float p1B = kb0 + lane <= posB ? TornadoMath.exp(x1B - mxB) : 0.0f;
            float p2B = kb0 + lane + 32 <= posB ? TornadoMath.exp(x2B - mxB) : 0.0f;
            lB = lB * alB + context.simdSum(p1B + p2B);
            mB = mxB;
            ss[(rb << 6) + lane] = p1B;
            ss[(rb << 6) + lane + 32] = p2B;
            if (lane == 0) {
                dg[ra * 9] = alA;
                dg[rb * 9] = alB;
            }
            context.localBarrier();
            Matrix8x8Float da = context.simdgroupMatrixLoad(dg, 0, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            int kp0 = kb0 + 0;
            if (kp0 <= maxPos) {
                Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, 0, 64);
                if (kp0 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp0 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb0 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp0,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p0,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb0 + 16, kvDim),
                                    o2);
                }
            }
            int kp1 = kb0 + 8;
            if (kp1 <= maxPos) {
                Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, 8, 64);
                if (kp1 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp1 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb1 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp1,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p1,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb1 + 16, kvDim),
                                    o2);
                }
            }
            int kp2 = kb0 + 16;
            if (kp2 <= maxPos) {
                Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, 16, 64);
                if (kp2 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp2 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb2 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp2,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p2,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb2 + 16, kvDim),
                                    o2);
                }
            }
            int kp3 = kb0 + 24;
            if (kp3 <= maxPos) {
                Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, 24, 64);
                if (kp3 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp3 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb3 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp3,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p3,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb3 + 16, kvDim),
                                    o2);
                }
            }
            int kp4 = kb0 + 32;
            if (kp4 <= maxPos) {
                Matrix8x8Float p4 = context.simdgroupMatrixLoad(ss, 32, 64);
                if (kp4 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp4 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb4 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp4,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p4,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb4 + 16, kvDim),
                                    o2);
                }
            }
            int kp5 = kb0 + 40;
            if (kp5 <= maxPos) {
                Matrix8x8Float p5 = context.simdgroupMatrixLoad(ss, 40, 64);
                if (kp5 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp5 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb5 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp5,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p5,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb5 + 16, kvDim),
                                    o2);
                }
            }
            int kp6 = kb0 + 48;
            if (kp6 <= maxPos) {
                Matrix8x8Float p6 = context.simdgroupMatrixLoad(ss, 48, 64);
                if (kp6 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp6 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb6 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp6,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p6,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb6 + 16, kvDim),
                                    o2);
                }
            }
            int kp7 = kb0 + 56;
            if (kp7 <= maxPos) {
                Matrix8x8Float p7 = context.simdgroupMatrixLoad(ss, 56, 64);
                if (kp7 + 7 > lastValid) {
                    for (int e = lane; e < 192; e += 32) {
                        int kp = kp7 + (e / 24);
                        float vv = 0.0f;
                        if (kp <= lastValid) {
                            vv =
                                    wrapValueCache
                                            .get(
                                                    KvBlockAddress.offset(
                                                                    blockTable,
                                                                    slot,
                                                                    kp,
                                                                    layerOff,
                                                                    kvDim,
                                                                    blockCfg,
                                                                    blockStride)
                                                            + headOff
                                                            + dOff
                                                            + (e % 24))
                                            .getFloat32();
                        }
                        vt[sg * 192 + e] = vv;
                    }
                    context.localBarrier();
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, sg * 192 + 0, 24), o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, sg * 192 + 8, 24), o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7, context.simdgroupMatrixLoad(vt, sg * 192 + 16, 24), o2);
                    context.localBarrier();
                } else {
                    int vb7 =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            kp7,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + headOff
                                    + dOff;
                    o0 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 0, kvDim),
                                    o0);
                    o1 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 8, kvDim),
                                    o1);
                    o2 =
                            context.simdgroupMatrixMultiplyAccumulate(
                                    p7,
                                    context.simdgroupMatrixLoad(wrapValueCache, vb7 + 16, kvDim),
                                    o2);
                }
            }
            context.localBarrier();
        }
        if (lane == 0) {
            dg[ra * 9] = lA > 0.0f ? 1.0f / lA : 0.0f;
            dg[rb * 9] = lB > 0.0f ? 1.0f / lB : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(dg, 0, 8);
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        int outBase = t0 * dim + h * headSize + dOff;
        if (t0 + 8 <= active) {
            context.simdgroupMatrixStore(o0, wrapXbBatch, outBase + 0, dim);
            context.simdgroupMatrixStore(o1, wrapXbBatch, outBase + 8, dim);
            context.simdgroupMatrixStore(o2, wrapXbBatch, outBase + 16, dim);
        } else {
            context.simdgroupMatrixStore(o0, vt, sg * 192 + 0, 8);
            context.simdgroupMatrixStore(o1, vt, sg * 192 + 64, 8);
            context.simdgroupMatrixStore(o2, vt, sg * 192 + 128, 8);
            context.localBarrier();
            for (int e = lane; e < 192; e += 32) {
                int r = (e & 63) >> 3;
                if (t0 + r < active) {
                    wrapXbBatch.set(
                            outBase + r * dim + ((e >> 6) << 3) + (e & 7), vt[sg * 192 + e]);
                }
            }
        }
    }
}
