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

    // ---- tensor-core prefill attention --------------------------------------------------------

    /** Queries of one head a tensor-core prefill workgroup owns: two MMA row tiles. */
    public static final int TC_QUERIES = 32;

    /** Keys one staged tile holds. */
    public static final int TC_KEYS = 32;

    /** The head width one pass of the tensor-core prefill attention covers. */
    public static final int TC_HEAD = 256;

    /** Lanes of a tensor-core prefill workgroup: eight warps. */
    public static final int TC_LANES = 256;

    /** Halves of FP16 staging scratch one workgroup owns: its two probability tiles, hi and lo. */
    public static final int TC_STAGE_HALVES = 2 * TC_QUERIES * TC_KEYS;

    /**
     * The exact power of two the probabilities are scaled by before they are split into FP16 parts,
     * so that the small ones stay out of FP16's subnormal range: at most 4096, for a probability of
     * one.
     */
    private static final float TC_PROB_SCALE = 4096.0f;

    private static final int TC_KV_TILE_INTS = TC_KEYS * TC_HEAD / 2;

    /**
     * Keys one score region holds for a layer: the most keys a 32-query tile can attend, rounded up
     * to whole key tiles. A sliding window bounds it by the window, not by the context.
     */
    public static int tcScoreKeys(int windowSize, int capacity) {
        int span = Math.min(windowSize + TC_QUERIES - 1, capacity);
        return (span + TC_KEYS - 1) / TC_KEYS * TC_KEYS;
    }

    // @formatter:off
    /**
     * Batched prefill attention over the FP16 cache on the tensor cores.
     *
     * <p>Workgroup {@code g}: query tile {@code g / nHeads} (chunk rows {@code 32 (g / nHeads) .. +
     * 31}), head {@code g % nHeads}. Three passes, the structure of {@code
     * Qwen35BatchKernels.attentionBatchFP16PagedTensorCoreT32}, over this family's flat cache and
     * sliding window:
     *
     * <ol>
     *   <li>{@code S^T = K Q^T} a 32-key tile at a time over the keys the tile's rows can attend —
     *       from the first row's window start to the last row's position — stored to the score
     *       region unscaled (this family's attention scale is 1.0);
     *   <li>per query, the maximum and the sum of exponentials over its own window;
     *   <li>{@code O = P V} a 32-key tile at a time, the probabilities outside a row's window zero,
     *       then each output divided by its row's sum and written FP16 for the output projection.
     * </ol>
     *
     * <p><b>Arithmetic: FP32 attention on FP16 tensor cores.</b> The query is carried as two FP16
     * parts, {@code hi = fp16(q)} and {@code lo = fp16(q - hi)}, and each score is {@code K hi + K
     * lo} accumulated in FP32 — the FP32 score to within reassociation, since an FP16 by FP16
     * product is exact in FP32 and the two parts carry the query to about 22 bits. The unnormalized
     * probabilities {@code exp(s - m)}, scaled by 2^12 so the small ones stay normal, are split the
     * same way for the value product. The softmax statistics are FP32; the output is divided by the
     * row's sum and the scale, then narrowed to FP16, as the kernel it replaces narrows it. Rows
     * past the chunk's token count are written as zeros.
     *
     * <p>Score region: {@code (queryTile * nHeads + head) * scoreKeys} rows of 32 floats, one per
     * key relative to the tile's first key; {@code scoreKeys} from {@link #tcScoreKeys}. Staging:
     * {@value #TC_STAGE_HALVES} halves per workgroup. A 512-wide head is taken as two 256-wide
     * halves: the second half's scores are added to the first's in the region, and the values are
     * accumulated one half at a time. Requires {@code headDim} a multiple of 256 and 256-lane
     * workgroups. Worker: {@code (rows / 32) * nHeads * 256} lanes, local 256.
     */
    // @formatter:on
    public static void attentionPrefillTensorCoreFP16(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outScratch,
            HalfFloatArray out,
            FloatArray scores,
            HalfFloatArray stage,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize,
            int capacity,
            int scoreKeys) {
        int tid = context.localIdx;
        int lane = tid & 31;
        int warp = tid >> 5;
        int group = context.groupIdx;
        int queryTile = group / nHeads;
        int head = group - queryTile * nHeads;
        int rowBase = queryTile * TC_QUERIES;
        int count = startPosHolder.get(1);
        int ld = nHeads * headDim;
        int halves = headDim / TC_HEAD;
        if (rowBase >= count) {
            for (int i = tid; i < TC_QUERIES * headDim; i += TC_LANES) {
                int row = i / headDim;
                out.set(
                        (rowBase + row) * ld + head * headDim + (i - row * headDim),
                        new HalfFloat(0.0f));
            }
            return;
        }
        int startPos = startPosHolder.get(0);
        int kvHead = head / kvMul;
        int headOff = cacheBaseOffset + kvHead * headDim;

        HalfFloat[] qTile = context.allocateHalfFloatLocalArray(TC_QUERIES * TC_HEAD);
        int[] kvTile = context.allocateIntLocalArray(TC_KV_TILE_INTS);
        int[] pTile = context.allocateIntLocalArray(TC_QUERIES * TC_KEYS);
        float[] rowStat = context.allocateFloatLocalArray(TC_QUERIES * 2 + TC_LANES);

        // The keys the tile's rows can attend: from the first row's window start to the last row.
        int firstKey = startPos + rowBase - windowSize + 1;
        if (firstKey < 0) {
            firstKey = 0;
        }
        int lastKey = startPos + rowBase + TC_QUERIES - 1;
        if (lastKey > capacity - 1) {
            lastKey = capacity - 1;
        }
        int regionRow = (queryTile * nHeads + head) * scoreKeys;
        int regionBase = regionRow * TC_QUERIES;
        int keyBlock = warp >> 2;
        int qBlock = warp & 3;

        // Pass 1: S^T = K Q^T, 32 keys a tile, stored to the region by key relative to firstKey;
        // a 512-wide head in two 256-wide halves, the second added to the first.
        for (int pass = 0; pass < 2 * halves; pass++) {
            int dOff = (pass >> 1) * TC_HEAD;
            boolean lowPart = (pass & 1) == 1;
            // Q^T half as the swizzled B operand: sub-tile (dims 16 t .., queries 8 g ..) at 4 t +
            // g.
            // The high part is fp16(q), the low part fp16(q - fp16(q)); their sum carries q to
            // about 22 bits, so K q_hi + K q_lo in FP32 is the FP32 score to reassociation.
            for (int i = tid; i < TC_QUERIES * TC_HEAD; i += TC_LANES) {
                int row = i >> 8;
                int d = i & 255;
                float qv = qkv.get((rowBase + row) * qkvStride + head * headDim + dOff + d);
                // One store per branch: a conditional choosing between two HalfFloat objects
                // computed the wrong low part on the 512-wide head (TornadoVM 4c6b5819).
                HalfFloat hi = new HalfFloat(qv);
                int offset = (((d >> 4) << 2) + (row >> 3)) * 256;
                if (lowPart) {
                    context.mmaStoreBSwizzled(
                            qTile, d & 15, row & 7, 8, new HalfFloat(qv - hi.getFloat32()), offset);
                } else {
                    context.mmaStoreBSwizzled(qTile, d & 15, row & 7, 8, hi, offset);
                }
            }
            context.localBarrier();
            for (int tileStart = firstKey; tileStart <= lastKey; tileStart += TC_KEYS) {
                for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC_LANES; i++) {
                    int e = i * TC_LANES + tid;
                    int key = e >> 7;
                    int d = (e & 127) << 1;
                    int p = tileStart + key;
                    if (p > lastKey) {
                        p = lastKey;
                    }
                    int dst =
                            (((key >> 4) << 4) + (d >> 4)) * 128
                                    + ((key & 15) << 3)
                                    + ((d & 15) >> 1);
                    context.asyncCopyToLocal(kvTile, dst, keyCache, headOff + p * kvDim + dOff + d);
                }
                context.asyncCopyCommit();
                context.asyncCopyWaitGroup(0);
                context.localBarrier();
                float[] s0 = context.mmaFragment(0.0f);
                for (int t = 0; t < TC_HEAD / 16; t++) {
                    HalfFloat[] a = context.mmaLoadA(kvTile, 16, ((keyBlock << 4) + t) * 512);
                    HalfFloat[] b0 = context.mmaLoadBSwizzled(qTile, 16, ((t << 2) + qBlock) * 256);
                    s0 = context.mma(a, b0, s0, MMAShape.M16N8K16);
                }
                int keyRow = regionRow + (tileStart - firstKey) + (keyBlock << 4);
                if (pass == 0) {
                    context.mmaStore(s0, scores, keyRow, qBlock << 3, TC_QUERIES);
                } else {
                    // Accumulator element i of lane l: row (l >> 2) + 8 (i >> 1), column
                    // 2 (l & 3) + (i & 1).
                    int r0 =
                            (keyRow + (lane >> 2)) * TC_QUERIES + (qBlock << 3) + ((lane & 3) << 1);
                    int r1 = r0 + 8 * TC_QUERIES;
                    scores.set(r0, scores.get(r0) + s0[0]);
                    scores.set(r0 + 1, scores.get(r0 + 1) + s0[1]);
                    scores.set(r1, scores.get(r1) + s0[2]);
                    scores.set(r1 + 1, scores.get(r1 + 1) + s0[3]);
                }
                context.localBarrier();
            }
        }

        // Pass 2: per query, the maximum and the denominator over its own window; lane = (query
        // tid % 32, part tid / 32), the parts folded in a fixed order.
        int statRow = tid & 31;
        int statPart = tid >> 5;
        int statPos = startPos + rowBase + statRow;
        if (statPos > lastKey) {
            statPos = lastKey;
        }
        int statFrom = statPos - windowSize + 1;
        if (statFrom < firstKey) {
            statFrom = firstKey;
        }
        float rowMax = Float.NEGATIVE_INFINITY;
        for (int p = statFrom + statPart; p <= statPos; p += 8) {
            rowMax =
                    TornadoMath.max(
                            rowMax, scores.get(regionBase + ((p - firstKey) << 5) + statRow));
        }
        rowStat[TC_QUERIES * 2 + tid] = rowMax;
        context.localBarrier();
        if (tid < TC_QUERIES) {
            int b = TC_QUERIES * 2 + tid;
            float m04 = TornadoMath.max(rowStat[b], rowStat[b + 128]);
            float m26 = TornadoMath.max(rowStat[b + 64], rowStat[b + 192]);
            float m15 = TornadoMath.max(rowStat[b + 32], rowStat[b + 160]);
            float m37 = TornadoMath.max(rowStat[b + 96], rowStat[b + 224]);
            rowStat[tid] = TornadoMath.max(TornadoMath.max(m04, m26), TornadoMath.max(m15, m37));
        }
        context.localBarrier();
        float m = rowStat[statRow];
        float rowSum = 0.0f;
        for (int p = statFrom + statPart; p <= statPos; p += 8) {
            rowSum += TornadoMath.exp(scores.get(regionBase + ((p - firstKey) << 5) + statRow) - m);
        }
        context.localBarrier();
        rowStat[TC_QUERIES * 2 + tid] = rowSum;
        context.localBarrier();
        if (tid < TC_QUERIES) {
            int b = TC_QUERIES * 2 + tid;
            float s04 = rowStat[b] + rowStat[b + 128];
            float s26 = rowStat[b + 64] + rowStat[b + 192];
            float s15 = rowStat[b + 32] + rowStat[b + 160];
            float s37 = rowStat[b + 96] + rowStat[b + 224];
            rowStat[TC_QUERIES + tid] = (s04 + s26) + (s15 + s37);
        }
        context.localBarrier();

        // Pass 3: O = P V, 32 keys a tile, one 256-wide half of the head at a time; warp w: query
        // rows 16 (w / 4) .., dims 64 (w % 4) .. of the half.
        int rowTile = warp >> 2;
        int dimBlock = warp & 3;
        int pBase = group * TC_STAGE_HALVES;
        int outRow = rowBase + (rowTile << 4);
        for (int hc = 0; hc < halves; hc++) {
            int dOff = hc * TC_HEAD;
            float[] o0 = context.mmaFragment(0.0f);
            float[] o1 = context.mmaFragment(0.0f);
            float[] o2 = context.mmaFragment(0.0f);
            float[] o3 = context.mmaFragment(0.0f);
            float[] o4 = context.mmaFragment(0.0f);
            float[] o5 = context.mmaFragment(0.0f);
            float[] o6 = context.mmaFragment(0.0f);
            float[] o7 = context.mmaFragment(0.0f);
            for (int tileStart = firstKey; tileStart <= lastKey; tileStart += TC_KEYS) {
                for (int i = 0; i < TC_KEYS * TC_HEAD / 2 / TC_LANES; i++) {
                    int e = i * TC_LANES + tid;
                    int key = e >> 7;
                    int d = (e & 127) << 1;
                    int p = tileStart + key;
                    if (p > lastKey) {
                        p = lastKey;
                    }
                    int dst =
                            (((key >> 4) << 5) + (d >> 3)) * 64
                                    + ((key & 15) << 2)
                                    + ((d & 7) >> 1);
                    context.asyncCopyToLocal(
                            kvTile, dst, valueCache, headOff + p * kvDim + dOff + d);
                }
                context.asyncCopyCommit();
                // P for this tile: lane covers (key = i / 32, query = i % 32), as halves into
                // P[query][key]; zero outside the query's window.
                for (int i = tid; i < TC_QUERIES * TC_KEYS; i += TC_LANES) {
                    int key = i >> 5;
                    int row = i & 31;
                    int p = tileStart + key;
                    int rowPos = startPos + rowBase + row;
                    float prob = 0.0f;
                    if (p <= rowPos && p <= lastKey && p > rowPos - windowSize) {
                        float sc = scores.get(regionBase + ((p - firstKey) << 5) + row);
                        prob = TC_PROB_SCALE * TornadoMath.exp(sc - rowStat[row]);
                    }
                    // The scaled probability as FP16 hi and lo parts, as the query is.
                    HalfFloat hi = new HalfFloat(prob);
                    stage.set(pBase + (row << 5) + key, hi);
                    stage.set(
                            pBase + TC_QUERIES * TC_KEYS + (row << 5) + key,
                            new HalfFloat(prob - hi.getFloat32()));
                }
                context.localBarrier();
                // Each P part as two A row tiles of [16 queries][32 keys]: block (rowTile r, key
                // step t) at (2 r + t) * 128 ints, 8 ints a row; the lo part after the hi one.
                for (int i = tid; i < TC_QUERIES * TC_KEYS; i += TC_LANES) {
                    int lo = i >> 9;
                    int j = i & 511;
                    int blk = j >> 7;
                    int q = ((blk >> 1) << 4) + ((j >> 3) & 15);
                    int kk = ((blk & 1) << 4) + ((j & 7) << 1);
                    context.asyncCopyToLocal(
                            pTile, i, stage, pBase + lo * TC_QUERIES * TC_KEYS + (q << 5) + kk);
                }
                context.asyncCopyCommit();
                context.asyncCopyWaitGroup(0);
                context.localBarrier();
                for (int t = 0; t < TC_KEYS / 16; t++) {
                    HalfFloat[] aHi = context.mmaLoadA(pTile, 16, ((rowTile << 1) + t) * 512);
                    HalfFloat[] aLo =
                            context.mmaLoadA(pTile, 16, ((rowTile << 1) + t) * 512 + 2048);
                    int vBase = (t << 5) + (dimBlock << 3);
                    HalfFloat[] b0 = context.mmaLoadB(kvTile, 16, vBase * 256);
                    o0 = context.mma(aHi, b0, o0, MMAShape.M16N8K16);
                    o0 = context.mma(aLo, b0, o0, MMAShape.M16N8K16);
                    HalfFloat[] b1 = context.mmaLoadB(kvTile, 16, (vBase + 1) * 256);
                    o1 = context.mma(aHi, b1, o1, MMAShape.M16N8K16);
                    o1 = context.mma(aLo, b1, o1, MMAShape.M16N8K16);
                    HalfFloat[] b2 = context.mmaLoadB(kvTile, 16, (vBase + 2) * 256);
                    o2 = context.mma(aHi, b2, o2, MMAShape.M16N8K16);
                    o2 = context.mma(aLo, b2, o2, MMAShape.M16N8K16);
                    HalfFloat[] b3 = context.mmaLoadB(kvTile, 16, (vBase + 3) * 256);
                    o3 = context.mma(aHi, b3, o3, MMAShape.M16N8K16);
                    o3 = context.mma(aLo, b3, o3, MMAShape.M16N8K16);
                    HalfFloat[] b4 = context.mmaLoadB(kvTile, 16, (vBase + 4) * 256);
                    o4 = context.mma(aHi, b4, o4, MMAShape.M16N8K16);
                    o4 = context.mma(aLo, b4, o4, MMAShape.M16N8K16);
                    HalfFloat[] b5 = context.mmaLoadB(kvTile, 16, (vBase + 5) * 256);
                    o5 = context.mma(aHi, b5, o5, MMAShape.M16N8K16);
                    o5 = context.mma(aLo, b5, o5, MMAShape.M16N8K16);
                    HalfFloat[] b6 = context.mmaLoadB(kvTile, 16, (vBase + 6) * 256);
                    o6 = context.mma(aHi, b6, o6, MMAShape.M16N8K16);
                    o6 = context.mma(aLo, b6, o6, MMAShape.M16N8K16);
                    HalfFloat[] b7 = context.mmaLoadB(kvTile, 16, (vBase + 7) * 256);
                    o7 = context.mma(aHi, b7, o7, MMAShape.M16N8K16);
                    o7 = context.mma(aLo, b7, o7, MMAShape.M16N8K16);
                }
                context.localBarrier();
            }
            int colBase = head * headDim + dOff + (dimBlock << 6);
            context.mmaStore(o0, outScratch, outRow, colBase, ld);
            context.mmaStore(o1, outScratch, outRow, colBase + 8, ld);
            context.mmaStore(o2, outScratch, outRow, colBase + 16, ld);
            context.mmaStore(o3, outScratch, outRow, colBase + 24, ld);
            context.mmaStore(o4, outScratch, outRow, colBase + 32, ld);
            context.mmaStore(o5, outScratch, outRow, colBase + 40, ld);
            context.mmaStore(o6, outScratch, outRow, colBase + 48, ld);
            context.mmaStore(o7, outScratch, outRow, colBase + 56, ld);
        }
        context.localBarrier();
        // Divide by the denominator and narrow; lane covers (row = i / headDim, dim).
        for (int i = tid; i < TC_QUERIES * headDim; i += TC_LANES) {
            int row = i / headDim;
            int idx = (rowBase + row) * ld + head * headDim + (i - row * headDim);
            float v =
                    rowBase + row < count
                            ? outScratch.get(idx) / (TC_PROB_SCALE * rowStat[TC_QUERIES + row])
                            : 0.0f;
            out.set(idx, new HalfFloat(v));
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
