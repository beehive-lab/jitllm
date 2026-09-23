package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Gemma 4's attention over a half-precision key/value cache.
 *
 * <p>The cache is the flat one {@link org.beehive.jitllm.inference.state.Gemma4State} lays out:
 * back-to-back slots for the layers that own their entries, addressed through a per-layer base
 * offset that a reusing layer shares with its source. Entries are stored as FP16 and every read
 * widens them to FP32, so the arithmetic downstream of the cache is FP32 as before; what changes is
 * the precision of the stored key and value, which is the tradeoff the FP16 cache makes in every
 * family that has one.
 *
 * <p>This family has <b>one key/value head for eight query heads</b>. The decode kernel here is
 * built around that: a workgroup owns one slice of the window for one key/value head and all of its
 * query heads, so each key and value row is read from memory once and used eight times, instead of
 * once per query head.
 */
// @formatter:on
public final class Gemma4AttentionKernels {

    private Gemma4AttentionKernels() {}

    /** Positions one decode workgroup attends: one per lane of a warp. */
    public static final int DECODE_SLICE = 32;

    /** Lanes of a decode workgroup: eight warps, one per query head in phase two. */
    public static final int DECODE_LANES = 256;

    /** Query heads per key/value head the decode kernel is written for. */
    public static final int DECODE_GROUP = 8;

    /** The widest head the decode kernel's staged queries hold. */
    public static final int DECODE_MAX_HEAD = 512;

    /** Floats of one slice's partial state: the group's numerators, then its maxima and sums. */
    public static int decodePartialStride(int headDim) {
        return DECODE_GROUP * headDim + 2 * DECODE_GROUP;
    }

    /** Whether a model's attention shape is one the grouped decode kernel is written for. */
    public static boolean decodeGroupFits(int kvMul, int headDimSwa, int headDimFull) {
        return kvMul == DECODE_GROUP && fitsHead(headDimSwa) && fitsHead(headDimFull);
    }

    private static boolean fitsHead(int headDim) {
        return headDim % 32 == 0 && headDim <= DECODE_MAX_HEAD;
    }

    /** Slices a window of {@code span} positions needs. */
    public static int decodeSlices(int span) {
        return (span + DECODE_SLICE - 1) / DECODE_SLICE;
    }

    // @formatter:off
    /**
     * NeoX RoPE for the query and key, the rotated key and the value written to the FP16 cache.
     *
     * <p>{@link Gemma4Kernels#ropeNeoxRotateAndCacheCopy} with the cache in half precision: the
     * rotation is the same FP32 arithmetic, and the entry is narrowed once on the way in. Worker: a
     * 2D grid of {@code (nHeads, headDim / 2)}.
     */
    // @formatter:on
    public static void ropeNeoxRotateAndCacheCopyFP16(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeadKv,
            int headDim,
            int kvDim,
            int cacheBaseOffset) {
        int h = context.globalIdx;
        int ic = context.globalIdy;
        int half = headDim / 2;
        int pos = positionHolder.get(0);

        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int qBase = h * headDim;
        float v0q = q.get(qBase + ic);
        float v1q = q.get(qBase + ic + half);
        q.set(qBase + ic, v0q * fcr - v1q * fci);
        q.set(qBase + ic + half, v0q * fci + v1q * fcr);

        if (h < nHeadKv) {
            int kBase = h * headDim;
            float v0k = k.get(kBase + ic);
            float v1k = k.get(kBase + ic + half);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;
            k.set(kBase + ic, rotatedK0);
            k.set(kBase + ic + half, rotatedK1);

            int cacheOffset = cacheBaseOffset + pos * kvDim + h * headDim;
            keyCache.set(cacheOffset + ic, new HalfFloat(rotatedK0));
            keyCache.set(cacheOffset + ic + half, new HalfFloat(rotatedK1));
            valueCache.set(cacheOffset + ic, new HalfFloat(v.get(kBase + ic)));
            valueCache.set(cacheOffset + ic + half, new HalfFloat(v.get(kBase + ic + half)));
        }
    }

    // @formatter:off
    /**
     * {@link Gemma4BatchPrefillKernels#batchedRopeAndCache} with the cache in half precision.
     * Worker: {@code B * nHeads * (headDim / 2)} threads.
     */
    // @formatter:on
    public static void batchedRopeAndCacheFP16(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeads,
            int nHeadKv,
            int headDim,
            int kvDim,
            int qkvStride,
            int cacheBaseOffset) {
        int gid = context.globalIdx;
        int half = headDim / 2;
        int perRow = nHeads * half;
        int b = gid / perRow;
        int rem = gid - b * perRow;
        int h = rem / half;
        int ic = rem - h * half;

        if (b >= startPosHolder.get(1)) {
            return;
        }
        int pos = startPosHolder.get(0) + b;
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int rowBase = b * qkvStride;
        int qBase = rowBase + h * headDim;
        float v0q = qkv.get(qBase + ic);
        float v1q = qkv.get(qBase + ic + half);
        qkv.set(qBase + ic, v0q * fcr - v1q * fci);
        qkv.set(qBase + ic + half, v0q * fci + v1q * fcr);

        if (h < nHeadKv) {
            int kBase = rowBase + nHeads * headDim + h * headDim;
            int vBase = kBase + nHeadKv * headDim;
            float v0k = qkv.get(kBase + ic);
            float v1k = qkv.get(kBase + ic + half);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            int cacheOffset = cacheBaseOffset + pos * kvDim + h * headDim;
            keyCache.set(cacheOffset + ic, new HalfFloat(rotatedK0));
            keyCache.set(cacheOffset + ic + half, new HalfFloat(rotatedK1));
            valueCache.set(cacheOffset + ic, new HalfFloat(qkv.get(vBase + ic)));
            valueCache.set(cacheOffset + ic + half, new HalfFloat(qkv.get(vBase + ic + half)));
        }
    }

    // @formatter:off
    /**
     * Decode attention, phase one of two: one workgroup per (key/value head, slice of {@value
     * #DECODE_SLICE} positions of the window), for all {@value #DECODE_GROUP} query heads of that
     * key/value head at once.
     *
     * <p><b>Scores.</b> Warp {@code w} takes positions {@code w, w + 8, ..} of the slice. For each,
     * the 32 lanes read the key row coalesced — lane {@code l} dimensions {@code l, l + 32, ..} —
     * and each lane accumulates its share of the dot product with every query head of the group; a
     * shuffle tree sums the lanes. <b>Softmax.</b> Warp {@code h} takes query head {@code h}, a
     * lane per position: the slice's maximum and sum of exponentials by shuffle, the exponentials
     * kept in shared memory. <b>Values.</b> A lane per dimension (two for a 512-wide head) reads
     * the value rows coalesced and accumulates the group's numerators.
     *
     * <p>Each slice writes its unnormalized state at {@code (kvHead * maxSlices + slice) *} {@link
     * #decodePartialStride(int)}: {@value #DECODE_GROUP} numerators of {@code headDim}, then
     * {@value #DECODE_GROUP} maxima, then {@value #DECODE_GROUP} sums. A slice past the end of the
     * window writes nothing; the combine only reads the slices the window reaches.
     *
     * <p><b>Arithmetic.</b> FP32 throughout after the cache read, with the attention scale of 1.0
     * this family uses. Each dot product is summed as 32 lane partials folded by the shuffle tree,
     * not left to right, and the softmax is merged across slices — a reassociation of the
     * reference's sums, the same kind every split-KV kernel here makes.
     *
     * <p>Requires {@code kvMul == } {@value #DECODE_GROUP}, {@code headDim} a multiple of 32 and at
     * most {@value #DECODE_MAX_HEAD}. Worker: {@code nKvHeads * maxSlices} workgroups of {@value
     * #DECODE_LANES} lanes.
     */
    // @formatter:on
    public static void attentionDecodeGroupFP16(
            KernelContext context,
            FloatArray q,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray partial,
            IntArray positionHolder,
            int headDim,
            int kvDim,
            int cacheBaseOffset,
            int windowSize,
            int maxSlices) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int kvHead = group / maxSlices;
        int slice = group - kvHead * maxSlices;

        int pos = positionHolder.get(0);
        int windowStart = pos - windowSize + 1;
        if (windowStart < 0) {
            windowStart = 0;
        }
        int from = windowStart + slice * DECODE_SLICE;
        if (from > pos) {
            return;
        }
        int valid = pos - from + 1;
        if (valid > DECODE_SLICE) {
            valid = DECODE_SLICE;
        }

        float[] qs = context.allocateFloatLocalArray(DECODE_GROUP * DECODE_MAX_HEAD);
        float[] sc = context.allocateFloatLocalArray(DECODE_GROUP * DECODE_SLICE);

        int groupWidth = DECODE_GROUP * headDim;
        int qBase = kvHead * groupWidth;
        for (int i = tid; i < groupWidth; i += DECODE_LANES) {
            qs[i] = q.get(qBase + i);
        }
        context.localBarrier();

        int headOff = cacheBaseOffset + kvHead * headDim;
        for (int t = warp; t < DECODE_SLICE; t += 8) {
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
            float a4 = 0.0f;
            float a5 = 0.0f;
            float a6 = 0.0f;
            float a7 = 0.0f;
            if (t < valid) {
                int keyOff = headOff + (from + t) * kvDim;
                for (int d = lane; d < headDim; d += 32) {
                    float kv = keyCache.get(keyOff + d).getFloat32();
                    a0 += qs[d] * kv;
                    a1 += qs[headDim + d] * kv;
                    a2 += qs[2 * headDim + d] * kv;
                    a3 += qs[3 * headDim + d] * kv;
                    a4 += qs[4 * headDim + d] * kv;
                    a5 += qs[5 * headDim + d] * kv;
                    a6 += qs[6 * headDim + d] * kv;
                    a7 += qs[7 * headDim + d] * kv;
                }
            }
            a0 = warpSum(context, a0);
            a1 = warpSum(context, a1);
            a2 = warpSum(context, a2);
            a3 = warpSum(context, a3);
            a4 = warpSum(context, a4);
            a5 = warpSum(context, a5);
            a6 = warpSum(context, a6);
            a7 = warpSum(context, a7);
            if (lane == 0) {
                sc[t] = a0;
                sc[DECODE_SLICE + t] = a1;
                sc[2 * DECODE_SLICE + t] = a2;
                sc[3 * DECODE_SLICE + t] = a3;
                sc[4 * DECODE_SLICE + t] = a4;
                sc[5 * DECODE_SLICE + t] = a5;
                sc[6 * DECODE_SLICE + t] = a6;
                sc[7 * DECODE_SLICE + t] = a7;
            }
        }
        context.localBarrier();

        // Warp h: query head h of the group, a lane per position.
        int partBase = group * decodePartialStride(headDim);
        float s = lane < valid ? sc[warp * DECODE_SLICE + lane] : Float.NEGATIVE_INFINITY;
        float m = s;
        m = TornadoMath.max(m, context.simdShuffleDown(m, 16));
        m = TornadoMath.max(m, context.simdShuffleDown(m, 8));
        m = TornadoMath.max(m, context.simdShuffleDown(m, 4));
        m = TornadoMath.max(m, context.simdShuffleDown(m, 2));
        m = TornadoMath.max(m, context.simdShuffleDown(m, 1));
        m = context.simdBroadcastFirst(m);
        float e = lane < valid ? TornadoMath.exp(s - m) : 0.0f;
        float l = warpSum(context, e);
        sc[warp * DECODE_SLICE + lane] = e;
        if (lane == 0) {
            partial.set(partBase + groupWidth + warp, m);
            partial.set(partBase + groupWidth + DECODE_GROUP + warp, l);
        }
        context.localBarrier();

        int valueBase = headOff + from * kvDim;
        for (int d = tid; d < headDim; d += DECODE_LANES) {
            float o0 = 0.0f;
            float o1 = 0.0f;
            float o2 = 0.0f;
            float o3 = 0.0f;
            float o4 = 0.0f;
            float o5 = 0.0f;
            float o6 = 0.0f;
            float o7 = 0.0f;
            for (int t = 0; t < valid; t++) {
                float vv = valueCache.get(valueBase + t * kvDim + d).getFloat32();
                o0 += sc[t] * vv;
                o1 += sc[DECODE_SLICE + t] * vv;
                o2 += sc[2 * DECODE_SLICE + t] * vv;
                o3 += sc[3 * DECODE_SLICE + t] * vv;
                o4 += sc[4 * DECODE_SLICE + t] * vv;
                o5 += sc[5 * DECODE_SLICE + t] * vv;
                o6 += sc[6 * DECODE_SLICE + t] * vv;
                o7 += sc[7 * DECODE_SLICE + t] * vv;
            }
            partial.set(partBase + d, o0);
            partial.set(partBase + headDim + d, o1);
            partial.set(partBase + 2 * headDim + d, o2);
            partial.set(partBase + 3 * headDim + d, o3);
            partial.set(partBase + 4 * headDim + d, o4);
            partial.set(partBase + 5 * headDim + d, o5);
            partial.set(partBase + 6 * headDim + d, o6);
            partial.set(partBase + 7 * headDim + d, o7);
        }
    }

    // @formatter:off
    /**
     * Decode attention, phase two: the slices merged, one thread per output element.
     *
     * <p>Reads only the slices the window reaches at this position, in slice order: the maximum
     * over their maxima, then the rescaled sums and numerators. Writes the normalized attention
     * output for query head {@code h} at {@code h * headDim}. Worker: {@code nHeads * headDim}
     * threads.
     */
    // @formatter:on
    public static void combineDecodeGroup(
            KernelContext context,
            FloatArray partial,
            FloatArray out,
            IntArray positionHolder,
            int nHeads,
            int headDim,
            int windowSize,
            int maxSlices) {
        int gid = context.globalIdx;
        if (gid >= nHeads * headDim) {
            return;
        }
        int h = gid / headDim;
        int d = gid - h * headDim;
        int kvHead = h / DECODE_GROUP;
        int member = h - kvHead * DECODE_GROUP;

        int pos = positionHolder.get(0);
        int windowStart = pos - windowSize + 1;
        if (windowStart < 0) {
            windowStart = 0;
        }
        int slices = (pos - windowStart + DECODE_SLICE) / DECODE_SLICE;
        int stride = decodePartialStride(headDim);
        int groupWidth = DECODE_GROUP * headDim;
        int first = kvHead * maxSlices;

        float gMax = Float.NEGATIVE_INFINITY;
        for (int s = 0; s < slices; s++) {
            gMax = TornadoMath.max(gMax, partial.get((first + s) * stride + groupWidth + member));
        }
        float denom = 0.0f;
        float acc = 0.0f;
        for (int s = 0; s < slices; s++) {
            int base = (first + s) * stride;
            float f = TornadoMath.exp(partial.get(base + groupWidth + member) - gMax);
            denom += f * partial.get(base + groupWidth + DECODE_GROUP + member);
            acc += f * partial.get(base + member * headDim + d);
        }
        out.set(h * headDim + d, acc / denom);
    }

    /** Dimensions the staged prefill attention loads per key tile. */
    private static final int STAGED_DIM_TILE = 16;

    // @formatter:off
    /**
     * {@link Gemma4BatchPrefillKernels#batchedSlidingWindowAttentionStaged} over the FP16 cache:
     * the same kernel with each key and value widened as it is read, and nothing else changed.
     * Worker: {@code B * nHeads} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttentionStagedFP16(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray out,
            FloatArray scores,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize,
            int scoreStride,
            int localMemSize) {
        int tid = context.localIdx;
        int group = context.groupIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int outBase = b * (nHeads * headDim) + h * headDim;

        if (b >= startPosHolder.get(1)) {
            for (int i = tid; i < headDim; i += localSize) {
                out.set(outBase + i, new HalfFloat(0.0f));
            }
            return;
        }

        int pos = startPosHolder.get(0) + b;
        int windowStart = Math.max(0, pos - windowSize + 1);
        int scoreBase = group * scoreStride;
        int kvHeadIdx = h / kvMul;
        int qOffset = b * qkvStride + h * headDim;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);
        float[] keyTile = context.allocateFloatLocalArray(STAGED_DIM_TILE * (localMemSize + 1));

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = qkv.get(qOffset + i);
        }
        context.localBarrier();

        int tileStride = localSize + 1;
        int tileElements = STAGED_DIM_TILE * localSize;
        for (int tBase = windowStart; tBase <= pos; tBase += localSize) {
            int t = tBase + tid;
            float score = 0.0f;
            for (int d0 = 0; d0 < headDim; d0 += STAGED_DIM_TILE) {
                context.localBarrier();
                for (int idx = tid; idx < tileElements; idx += localSize) {
                    int d = idx % STAGED_DIM_TILE;
                    int p = idx / STAGED_DIM_TILE;
                    int tt = tBase + p;
                    float v =
                            (tt <= pos)
                                    ? keyCache.get(
                                                    cacheBaseOffset
                                                            + tt * kvDim
                                                            + kvHeadIdx * headDim
                                                            + d0
                                                            + d)
                                            .getFloat32()
                                    : 0.0f;
                    keyTile[d * tileStride + p] = v;
                }
                context.localBarrier();
                if (t <= pos) {
                    for (int d = 0; d < STAGED_DIM_TILE; d++) {
                        score += qShared[d0 + d] * keyTile[d * tileStride + tid];
                    }
                }
            }
            if (t <= pos) {
                scores.set(scoreBase + (t - windowStart), score);
            }
        }
        context.localBarrier();

        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float v = scores.get(scoreBase + (t - windowStart));
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float maxScore = reduce[0];
        context.localBarrier();

        float localSum = 0.0f;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float e = TornadoMath.exp(scores.get(scoreBase + (t - windowStart)) - maxScore);
            scores.set(scoreBase + (t - windowStart), e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sum = reduce[0];
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            scores.set(
                    scoreBase + (t - windowStart),
                    scores.get(scoreBase + (t - windowStart)) * normFactor);
        }
        context.localBarrier();

        for (int i = tid; i < headDim; i += localSize) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum +=
                        scores.get(scoreBase + (t - windowStart))
                                * valueCache.get(valueOffset + i).getFloat32();
            }
            out.set(outBase + i, new HalfFloat(weightedSum));
        }
    }

    private static float warpSum(KernelContext context, float v) {
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return context.simdBroadcastFirst(v);
    }
}
