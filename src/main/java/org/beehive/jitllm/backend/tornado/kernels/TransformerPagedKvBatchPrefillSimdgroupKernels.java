package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;

// @formatter:off
/**
 * Batched-prefill attention over the paged FP32 key/value cache on Apple SIMD-group matrices
 * ({@code simdgroup_float8x8}), one kernel per head width (64, 96 and 128).
 *
 * <p>A workgroup of {@value #THREADS} threads handles 32 consecutive query tokens of one head; each
 * of its four SIMD groups owns eight of the tokens. The queries are staged in threadgroup memory
 * once. Keys are visited in blocks of 32 up to the last position of the workgroup's tokens: the
 * scores of a block are one {@code simdgroup_multiply_accumulate} chain per 8x8 fragment, with key
 * fragments read from the cache by transposed loads (eight consecutive positions lie in one cache
 * block, a row of {@code kvDim} floats apart). The online softmax runs per query row on the scores
 * in threadgroup memory, with the causal mask applied there; the output accumulator is rescaled by
 * multiplying it with a diagonal fragment of the correction factors, and the probabilities are
 * multiplied with value fragments read from the cache. A key fragment that reaches past the last
 * position written by this batch reads its values through a zero-filled threadgroup buffer instead,
 * so unwritten cache memory never meets a zero weight. Query rows past the batch are computed as
 * copies of the last active row for the same reason. The result is normalised by a second diagonal
 * multiplication and stored straight to the output when the workgroup's 32 tokens are all active.
 */
// @formatter:on
public final class TransformerPagedKvBatchPrefillSimdgroupKernels {

    /** Threads per workgroup: four SIMD groups. */
    public static final int THREADS = 128;

    /** Query tokens per workgroup. */
    public static final int QUERY_TILE = 32;

    private TransformerPagedKvBatchPrefillSimdgroupKernels() {}

    // @formatter:off
    /**
     * Causal attention of a batch of query tokens over the paged FP32 cache, for a 64-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 64.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedHead64(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
     * Causal attention of a batch of query tokens over the paged FP32 cache, for a 96-wide head, on
     * SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 96.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedHead96(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
     * Causal attention of a batch of query tokens over the paged FP32 cache, for a 128-wide head,
     * on SIMD-group matrices; the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 128.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedHead128(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
     * {@link #batchedFlashAttentionPagedHead64} with the scores scaled by {@code attentionScale}
     * instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier). Causal attention of a
     * batch of query tokens over the paged FP32 cache, for a 64-wide head, on SIMD-group matrices;
     * the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 64.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedScaledHead64(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 64);
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
     * {@link #batchedFlashAttentionPagedHead96} with the scores scaled by {@code attentionScale}
     * instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier). Causal attention of a
     * batch of query tokens over the paged FP32 cache, for a 96-wide head, on SIMD-group matrices;
     * the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 96.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedScaledHead96(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 96);
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
     * {@link #batchedFlashAttentionPagedHead128} with the scores scaled by {@code attentionScale}
     * instead of {@code 1 / sqrt(headSize)} (Granite's attention multiplier). Causal attention of a
     * batch of query tokens over the paged FP32 cache, for a 128-wide head, on SIMD-group matrices;
     * the arguments of {@link
     * TransformerPagedKvBatchPrefillKernels#batchedFlashAttentionPagedLaneHead128}. Grid: {@code
     * ceil(batch / 32) * nHeads} workgroups of {@value #THREADS} threads, workgroup {@code g}
     * taking tokens {@code 32 * (g / nHeads)} onwards of head {@code g % nHeads}. {@code headSize}
     * must be 128.
     */
    // @formatter:on
    public static void batchedFlashAttentionPagedScaledHead128(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
                                wrapValueCache.get(
                                        KvBlockAddress.offset(
                                                        blockTable,
                                                        slot,
                                                        kp,
                                                        layerOff,
                                                        kvDim,
                                                        blockCfg,
                                                        blockStride)
                                                + headOff
                                                + e % 128);
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
}
