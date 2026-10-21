package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;

// @formatter:off
/**
 * Gemma 4's batched-prefill attention on Apple SIMD-group matrices ({@code simdgroup_float8x8}),
 * for its 256-wide (sliding-window) and 512-wide (full) heads.
 *
 * <p>A workgroup of {@value #THREADS} threads takes 8 query tokens of one head. Keys are visited in
 * blocks of 32 from the first position any of its tokens can see: the four SIMD groups compute the
 * block's scores together, one 8-key fragment each, reading query fragments from the packed QKV
 * rows and key fragments from the cache by transposed loads. The online softmax runs on the scores
 * in threadgroup memory with the causal window applied there. Each SIMD group then owns a quarter
 * of the head's output columns: it rescales its accumulators by multiplying them with a diagonal
 * fragment of the correction factors and adds the probabilities times its value columns. Wide heads
 * are why the work is split by columns: a whole 512-wide row of accumulators per SIMD group would
 * not fit in registers, and staging 32 such query rows would not fit in threadgroup memory.
 *
 * <p>Unwritten memory never meets a nonzero product: a key fragment reaching past the last position
 * this chunk wrote reads its values through a zero-filled threadgroup buffer, and a query row past
 * the chunk gets probability zero everywhere and a zero rescale factor. The output is FP16, as the
 * output projection reads it.
 */
// @formatter:on
public final class Gemma4BatchPrefillSimdgroupAttentionKernels {

    /** Threads per workgroup: four SIMD groups. */
    public static final int THREADS = 128;

    /** Query tokens per workgroup. */
    public static final int QUERY_TILE = 8;

    private Gemma4BatchPrefillSimdgroupAttentionKernels() {}

    // @formatter:off
    /**
     * Sliding-window causal attention of a chunk of query tokens over the FP32 cache for a 256-wide
     * head, on SIMD-group matrices; the arguments of {@link
     * Gemma4BatchPrefillKernels#batchedSlidingWindowAttentionStaged} without its score scratch.
     * Unscaled scores, as Gemma 4 has them. Grid: {@code ceil(batch / 8) * nHeads} workgroups of
     * {@value #THREADS} threads, workgroup {@code g} taking tokens {@code 8 * (g / nHeads)} onwards
     * of head {@code g % nHeads}. {@code headDim} must be 256.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttentionHead256(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray out,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize) {
        float[] ss = context.allocateFloatLocalArray(256);
        float[] ds = context.allocateFloatLocalArray(64);
        float[] rmax = context.allocateFloatLocalArray(128);
        float[] rsum = context.allocateFloatLocalArray(128);
        float[] vt = context.allocateFloatLocalArray(2048);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = startPosHolder.get(1);
        int t0 = tile * 8;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = startPosHolder.get(0);
        int kvBase = cacheBaseOffset + (h / kvMul) * 256;
        int qBase = t0 * qkvStride + h * 256;
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 7, active - 1);
        int lastGroup = maxPos & ~7;
        int kbStart = TornadoMath.max(0, start + t0 - windowSize + 1) & ~7;
        for (int e = tid; e < 64; e += 128) {
            ds[e] = 0.0f;
        }
        // Softmax: lanes 16 to a query row, 2 scores each. A row past the batch, or a key outside
        // the
        // row's causal window, gets probability zero; such a row keeps m = -inf, so its rescale
        // factor is
        // zero too, and garbage in its query never reaches the valid rows through the diagonal
        // products.
        int row = tid / 16;
        int sub = tid % 16;
        boolean rowActive = t0 + row < active;
        int myPos = start + t0 + row;
        int lo = myPos - windowSize + 1;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        // Output: SIMD group sg owns query block 0, columns 64 * (sg % 4) onwards.
        int oqr = 0;
        int oc = (sg % 4) * 64;
        Matrix8x8Float zero = context.simdgroupMatrixZero();
        Matrix8x8Float o0 = context.simdgroupMatrixZero();
        Matrix8x8Float o1 = context.simdgroupMatrixZero();
        Matrix8x8Float o2 = context.simdgroupMatrixZero();
        Matrix8x8Float o3 = context.simdgroupMatrixZero();
        Matrix8x8Float o4 = context.simdgroupMatrixZero();
        Matrix8x8Float o5 = context.simdgroupMatrixZero();
        Matrix8x8Float o6 = context.simdgroupMatrixZero();
        Matrix8x8Float o7 = context.simdgroupMatrixZero();
        context.localBarrier();
        for (int kb0 = kbStart; kb0 <= maxPos; kb0 += 32) {
            int f0 = sg * 1 + 0;
            int sq0 = f0 >> 2;
            int kq0 = kb0 + ((f0 & 3) << 3);
            int kAddr0 = kvBase + TornadoMath.min(kq0, lastGroup) * kvDim;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            for (int d = 0; d < 256; d += 8) {
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                context.simdgroupMatrixLoad(
                                        qkv, qBase + sq0 * 8 * qkvStride + d, qkvStride),
                                context.simdgroupMatrixLoadTransposed(keyCache, kAddr0 + d, kvDim),
                                s0);
            }
            context.simdgroupMatrixStore(s0, ss, sq0 * 256 + ((f0 & 3) << 3), 32);
            context.localBarrier();
            int rowOff = (row << 5) + sub * 2;
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 2; j++) {
                int kp = kb0 + sub * 2 + j;
                if (rowActive && kp <= myPos && kp >= lo) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j]);
                }
            }
            rmax[tid] = pmx;
            context.localBarrier();
            float mx = m;
            for (int j = 0; j < 16; j++) {
                mx = TornadoMath.max(mx, rmax[row * 16 + j]);
            }
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 2; j++) {
                int kp = kb0 + sub * 2 + j;
                float p = 0.0f;
                if (rowActive && kp <= myPos && kp >= lo) {
                    p = TornadoMath.exp(ss[rowOff + j] - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[tid] = psum;
            if (sub == 0) {
                ds[((row >> 3) << 6) + (row & 7) * 9] = alpha;
            }
            context.localBarrier();
            float bsum = 0.0f;
            for (int j = 0; j < 16; j++) {
                bsum += rsum[row * 16 + j];
            }
            l = l * alpha + bsum;
            m = mx;
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, oqr << 6, 8);
            o0 = context.simdgroupMatrixMultiplyAccumulate(da, o0, zero);
            o1 = context.simdgroupMatrixMultiplyAccumulate(da, o1, zero);
            o2 = context.simdgroupMatrixMultiplyAccumulate(da, o2, zero);
            o3 = context.simdgroupMatrixMultiplyAccumulate(da, o3, zero);
            o4 = context.simdgroupMatrixMultiplyAccumulate(da, o4, zero);
            o5 = context.simdgroupMatrixMultiplyAccumulate(da, o5, zero);
            o6 = context.simdgroupMatrixMultiplyAccumulate(da, o6, zero);
            o7 = context.simdgroupMatrixMultiplyAccumulate(da, o7, zero);
            int kv0 = kb0 + 0;
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 0, 32);
            if (kv0 + 7 > lastValid) {
                for (int e = tid; e < 2048; e += 128) {
                    int kp = kv0 + e / 256;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 256);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 0, 256), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 8, 256), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 16, 256), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 24, 256), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 32, 256), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 40, 256), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 48, 256), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 56, 256), o7);
                context.localBarrier();
            } else {
                int vAddr0 = kvBase + kv0 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(valueCache, vAddr0 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(valueCache, vAddr0 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 56, kvDim),
                                o7);
            }
            int kv1 = kb0 + 8;
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 8, 32);
            if (kv1 + 7 > lastValid) {
                for (int e = tid; e < 2048; e += 128) {
                    int kp = kv1 + e / 256;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 256);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 0, 256), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 8, 256), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 16, 256), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 24, 256), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 32, 256), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 40, 256), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 48, 256), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 56, 256), o7);
                context.localBarrier();
            } else {
                int vAddr1 = kvBase + kv1 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(valueCache, vAddr1 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(valueCache, vAddr1 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 56, kvDim),
                                o7);
            }
            int kv2 = kb0 + 16;
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 16, 32);
            if (kv2 + 7 > lastValid) {
                for (int e = tid; e < 2048; e += 128) {
                    int kp = kv2 + e / 256;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 256);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 0, 256), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 8, 256), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 16, 256), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 24, 256), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 32, 256), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 40, 256), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 48, 256), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 56, 256), o7);
                context.localBarrier();
            } else {
                int vAddr2 = kvBase + kv2 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(valueCache, vAddr2 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(valueCache, vAddr2 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 56, kvDim),
                                o7);
            }
            int kv3 = kb0 + 24;
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 24, 32);
            if (kv3 + 7 > lastValid) {
                for (int e = tid; e < 2048; e += 128) {
                    int kp = kv3 + e / 256;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 256);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 0, 256), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 8, 256), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 16, 256), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 24, 256), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 32, 256), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 40, 256), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 48, 256), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 56, 256), o7);
                context.localBarrier();
            } else {
                int vAddr3 = kvBase + kv3 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(valueCache, vAddr3 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(valueCache, vAddr3 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 56, kvDim),
                                o7);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[((row >> 3) << 6) + (row & 7) * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, oqr << 6, 8);
        int outStride = nHeads * 256;
        int outBase = (t0 + (oqr << 3)) * outStride + h * 256 + oc;
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        context.simdgroupMatrixStore(o0, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 0 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        context.simdgroupMatrixStore(o1, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 8 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        context.simdgroupMatrixStore(o2, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 16 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        context.simdgroupMatrixStore(o3, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 24 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        context.simdgroupMatrixStore(o4, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 32 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        context.simdgroupMatrixStore(o5, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 40 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        context.simdgroupMatrixStore(o6, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 48 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        context.simdgroupMatrixStore(o7, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 56 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
    }

    // @formatter:off
    /**
     * Sliding-window causal attention of a chunk of query tokens over the FP32 cache for a 512-wide
     * head, on SIMD-group matrices; the arguments of {@link
     * Gemma4BatchPrefillKernels#batchedSlidingWindowAttentionStaged} without its score scratch.
     * Unscaled scores, as Gemma 4 has them. Grid: {@code ceil(batch / 8) * nHeads} workgroups of
     * {@value #THREADS} threads, workgroup {@code g} taking tokens {@code 8 * (g / nHeads)} onwards
     * of head {@code g % nHeads}. {@code headDim} must be 512.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttentionHead512(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray out,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize) {
        float[] ss = context.allocateFloatLocalArray(256);
        float[] ds = context.allocateFloatLocalArray(64);
        float[] rmax = context.allocateFloatLocalArray(128);
        float[] rsum = context.allocateFloatLocalArray(128);
        float[] vt = context.allocateFloatLocalArray(4096);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int grp = context.groupIdx;
        int tile = grp / nHeads;
        int h = grp % nHeads;
        int active = startPosHolder.get(1);
        int t0 = tile * 8;
        if (t0 >= active) {
            return; // padding tile: the whole workgroup leaves together
        }
        int start = startPosHolder.get(0);
        int kvBase = cacheBaseOffset + (h / kvMul) * 512;
        int qBase = t0 * qkvStride + h * 512;
        int lastValid = start + active - 1;
        int maxPos = start + TornadoMath.min(t0 + 7, active - 1);
        int lastGroup = maxPos & ~7;
        int kbStart = TornadoMath.max(0, start + t0 - windowSize + 1) & ~7;
        for (int e = tid; e < 64; e += 128) {
            ds[e] = 0.0f;
        }
        // Softmax: lanes 16 to a query row, 2 scores each. A row past the batch, or a key outside
        // the
        // row's causal window, gets probability zero; such a row keeps m = -inf, so its rescale
        // factor is
        // zero too, and garbage in its query never reaches the valid rows through the diagonal
        // products.
        int row = tid / 16;
        int sub = tid % 16;
        boolean rowActive = t0 + row < active;
        int myPos = start + t0 + row;
        int lo = myPos - windowSize + 1;
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        // Output: SIMD group sg owns query block 0, columns 128 * (sg % 4) onwards.
        int oqr = 0;
        int oc = (sg % 4) * 128;
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
        context.localBarrier();
        for (int kb0 = kbStart; kb0 <= maxPos; kb0 += 32) {
            int f0 = sg * 1 + 0;
            int sq0 = f0 >> 2;
            int kq0 = kb0 + ((f0 & 3) << 3);
            int kAddr0 = kvBase + TornadoMath.min(kq0, lastGroup) * kvDim;
            Matrix8x8Float s0 = context.simdgroupMatrixZero();
            for (int d = 0; d < 512; d += 8) {
                s0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                context.simdgroupMatrixLoad(
                                        qkv, qBase + sq0 * 8 * qkvStride + d, qkvStride),
                                context.simdgroupMatrixLoadTransposed(keyCache, kAddr0 + d, kvDim),
                                s0);
            }
            context.simdgroupMatrixStore(s0, ss, sq0 * 256 + ((f0 & 3) << 3), 32);
            context.localBarrier();
            int rowOff = (row << 5) + sub * 2;
            float pmx = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < 2; j++) {
                int kp = kb0 + sub * 2 + j;
                if (rowActive && kp <= myPos && kp >= lo) {
                    pmx = TornadoMath.max(pmx, ss[rowOff + j]);
                }
            }
            rmax[tid] = pmx;
            context.localBarrier();
            float mx = m;
            for (int j = 0; j < 16; j++) {
                mx = TornadoMath.max(mx, rmax[row * 16 + j]);
            }
            float alpha = (m == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(m - mx);
            float psum = 0.0f;
            for (int j = 0; j < 2; j++) {
                int kp = kb0 + sub * 2 + j;
                float p = 0.0f;
                if (rowActive && kp <= myPos && kp >= lo) {
                    p = TornadoMath.exp(ss[rowOff + j] - mx);
                }
                ss[rowOff + j] = p;
                psum += p;
            }
            rsum[tid] = psum;
            if (sub == 0) {
                ds[((row >> 3) << 6) + (row & 7) * 9] = alpha;
            }
            context.localBarrier();
            float bsum = 0.0f;
            for (int j = 0; j < 16; j++) {
                bsum += rsum[row * 16 + j];
            }
            l = l * alpha + bsum;
            m = mx;
            Matrix8x8Float da = context.simdgroupMatrixLoad(ds, oqr << 6, 8);
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
            int kv0 = kb0 + 0;
            Matrix8x8Float p0 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 0, 32);
            if (kv0 + 7 > lastValid) {
                for (int e = tid; e < 4096; e += 128) {
                    int kp = kv0 + e / 512;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 512);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 0, 512), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 8, 512), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 16, 512), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 24, 512), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 32, 512), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 40, 512), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 48, 512), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 56, 512), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 64, 512), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 72, 512), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 80, 512), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 88, 512), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 96, 512), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 104, 512), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 112, 512), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(vt, oc + 120, 512), o15);
                context.localBarrier();
            } else {
                int vAddr0 = kvBase + kv0 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(valueCache, vAddr0 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0, context.simdgroupMatrixLoad(valueCache, vAddr0 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p0,
                                context.simdgroupMatrixLoad(valueCache, vAddr0 + 120, kvDim),
                                o15);
            }
            int kv1 = kb0 + 8;
            Matrix8x8Float p1 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 8, 32);
            if (kv1 + 7 > lastValid) {
                for (int e = tid; e < 4096; e += 128) {
                    int kp = kv1 + e / 512;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 512);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 0, 512), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 8, 512), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 16, 512), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 24, 512), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 32, 512), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 40, 512), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 48, 512), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 56, 512), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 64, 512), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 72, 512), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 80, 512), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 88, 512), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 96, 512), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 104, 512), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 112, 512), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(vt, oc + 120, 512), o15);
                context.localBarrier();
            } else {
                int vAddr1 = kvBase + kv1 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(valueCache, vAddr1 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1, context.simdgroupMatrixLoad(valueCache, vAddr1 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p1,
                                context.simdgroupMatrixLoad(valueCache, vAddr1 + 120, kvDim),
                                o15);
            }
            int kv2 = kb0 + 16;
            Matrix8x8Float p2 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 16, 32);
            if (kv2 + 7 > lastValid) {
                for (int e = tid; e < 4096; e += 128) {
                    int kp = kv2 + e / 512;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 512);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 0, 512), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 8, 512), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 16, 512), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 24, 512), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 32, 512), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 40, 512), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 48, 512), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 56, 512), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 64, 512), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 72, 512), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 80, 512), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 88, 512), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 96, 512), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 104, 512), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 112, 512), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(vt, oc + 120, 512), o15);
                context.localBarrier();
            } else {
                int vAddr2 = kvBase + kv2 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(valueCache, vAddr2 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2, context.simdgroupMatrixLoad(valueCache, vAddr2 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p2,
                                context.simdgroupMatrixLoad(valueCache, vAddr2 + 120, kvDim),
                                o15);
            }
            int kv3 = kb0 + 24;
            Matrix8x8Float p3 = context.simdgroupMatrixLoad(ss, (oqr << 8) + 24, 32);
            if (kv3 + 7 > lastValid) {
                for (int e = tid; e < 4096; e += 128) {
                    int kp = kv3 + e / 512;
                    float vv = 0.0f;
                    if (kp <= lastValid) {
                        vv = valueCache.get(kvBase + kp * kvDim + e % 512);
                    }
                    vt[e] = vv;
                }
                context.localBarrier();
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 0, 512), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 8, 512), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 16, 512), o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 24, 512), o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 32, 512), o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 40, 512), o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 48, 512), o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 56, 512), o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 64, 512), o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 72, 512), o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 80, 512), o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 88, 512), o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 96, 512), o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 104, 512), o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 112, 512), o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(vt, oc + 120, 512), o15);
                context.localBarrier();
            } else {
                int vAddr3 = kvBase + kv3 * kvDim + oc;
                o0 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(valueCache, vAddr3 + 0, kvDim), o0);
                o1 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3, context.simdgroupMatrixLoad(valueCache, vAddr3 + 8, kvDim), o1);
                o2 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 16, kvDim),
                                o2);
                o3 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 24, kvDim),
                                o3);
                o4 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 32, kvDim),
                                o4);
                o5 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 40, kvDim),
                                o5);
                o6 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 48, kvDim),
                                o6);
                o7 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 56, kvDim),
                                o7);
                o8 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 64, kvDim),
                                o8);
                o9 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 72, kvDim),
                                o9);
                o10 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 80, kvDim),
                                o10);
                o11 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 88, kvDim),
                                o11);
                o12 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 96, kvDim),
                                o12);
                o13 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 104, kvDim),
                                o13);
                o14 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 112, kvDim),
                                o14);
                o15 =
                        context.simdgroupMatrixMultiplyAccumulate(
                                p3,
                                context.simdgroupMatrixLoad(valueCache, vAddr3 + 120, kvDim),
                                o15);
            }
            context.localBarrier();
        }
        if (sub == 0) {
            ds[((row >> 3) << 6) + (row & 7) * 9] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        context.localBarrier();
        Matrix8x8Float dn = context.simdgroupMatrixLoad(ds, oqr << 6, 8);
        int outStride = nHeads * 512;
        int outBase = (t0 + (oqr << 3)) * outStride + h * 512 + oc;
        o0 = context.simdgroupMatrixMultiplyAccumulate(dn, o0, zero);
        context.simdgroupMatrixStore(o0, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 0 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o1 = context.simdgroupMatrixMultiplyAccumulate(dn, o1, zero);
        context.simdgroupMatrixStore(o1, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 8 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o2 = context.simdgroupMatrixMultiplyAccumulate(dn, o2, zero);
        context.simdgroupMatrixStore(o2, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 16 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o3 = context.simdgroupMatrixMultiplyAccumulate(dn, o3, zero);
        context.simdgroupMatrixStore(o3, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 24 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o4 = context.simdgroupMatrixMultiplyAccumulate(dn, o4, zero);
        context.simdgroupMatrixStore(o4, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 32 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o5 = context.simdgroupMatrixMultiplyAccumulate(dn, o5, zero);
        context.simdgroupMatrixStore(o5, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 40 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o6 = context.simdgroupMatrixMultiplyAccumulate(dn, o6, zero);
        context.simdgroupMatrixStore(o6, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 48 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o7 = context.simdgroupMatrixMultiplyAccumulate(dn, o7, zero);
        context.simdgroupMatrixStore(o7, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 56 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o8 = context.simdgroupMatrixMultiplyAccumulate(dn, o8, zero);
        context.simdgroupMatrixStore(o8, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 64 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o9 = context.simdgroupMatrixMultiplyAccumulate(dn, o9, zero);
        context.simdgroupMatrixStore(o9, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 72 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o10 = context.simdgroupMatrixMultiplyAccumulate(dn, o10, zero);
        context.simdgroupMatrixStore(o10, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 80 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o11 = context.simdgroupMatrixMultiplyAccumulate(dn, o11, zero);
        context.simdgroupMatrixStore(o11, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 88 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o12 = context.simdgroupMatrixMultiplyAccumulate(dn, o12, zero);
        context.simdgroupMatrixStore(o12, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 96 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o13 = context.simdgroupMatrixMultiplyAccumulate(dn, o13, zero);
        context.simdgroupMatrixStore(o13, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 104 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o14 = context.simdgroupMatrixMultiplyAccumulate(dn, o14, zero);
        context.simdgroupMatrixStore(o14, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 112 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
        o15 = context.simdgroupMatrixMultiplyAccumulate(dn, o15, zero);
        context.simdgroupMatrixStore(o15, ss, sg << 6, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            if (t0 + (oqr << 3) + (e >> 3) < active) {
                out.set(
                        outBase + (e >> 3) * outStride + 120 + (e & 7),
                        new HalfFloat(ss[(sg << 6) + e]));
            }
        }
        context.localBarrier();
    }
}
