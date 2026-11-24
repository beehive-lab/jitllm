package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Single-token device kernels for gpt-oss.
 *
 * <p>Projections run one warp per output row, and a shuffle reduction leaves the sum in lane 0.
 * Each kernel takes its {@link KernelContext} first and checks its row or element bound explicitly.
 *
 * <p>MXFP4 is decoded as llama.cpp decodes it: twice the E2M1 value times {@code 2^(e - 128)}. A
 * row of {@code n} weights is {@code n / 32} blocks of 17 bytes, the exponent byte first, then 16
 * bytes whose low nibbles hold elements 0..15 and high nibbles 16..31. The expert kernels copy each
 * weight row into shared memory with coalesced 16-bit loads (a row of an even number of blocks is
 * 2-byte aligned), then decode it a lane per block through a shared nibble table.
 */
public final class GptOssKernels {

    private GptOssKernels() {}

    static final int WARP = 32;
    static final int MXFP4_BYTES = 17;
    static final float LIMIT = 7.0f;
    static final float ALPHA = 1.702f;

    /** Attention: the head width and the positions one work-group scores per step. */
    public static final int HEAD = 64;

    /** A masked score: finite, so it never meets another infinity in a subtraction. */
    static final float MASKED = -1.0e30f;

    static float warpSum(KernelContext ctx, float v) {
        float s = v;
        s += ctx.simdShuffleDown(s, 16);
        s += ctx.simdShuffleDown(s, 8);
        s += ctx.simdShuffleDown(s, 4);
        s += ctx.simdShuffleDown(s, 2);
        s += ctx.simdShuffleDown(s, 1);
        return s;
    }

    /**
     * Twice the E2M1 value of nibble {@code n}: {@code 0,1,2,3,4,6,8,12}, negated when bit 3 is
     * set.
     */
    static float fp4(int n) {
        int m = n & 7;
        int big = (m >> 2) & 1;
        int small = m * (1 - big);
        int shift = ((m >> 1) - 1) & 3;
        int large = ((2 + (m & 1)) << shift) * big;
        int sign = 1 - ((n >> 2) & 2);
        return (float) ((small + large) * sign);
    }

    /** {@code 2^(e - 128)}, the E8M0 exponent halved, exactly as the host decodes it. */
    static float e8m0Half(int e) {
        // Below 2 the halved scale is subnormal: 2^-128 and 2^-127 as their bit patterns.
        int bits = e < 2 ? 0x00200000 << e : (e - 1) << 23;
        return Float.intBitsToFloat(bits);
    }

    /** gpt-oss's clamped SwiGLU. */
    static float swiglu(float gate, float up) {
        float g = Math.min(gate, LIMIT);
        float u = Math.min(Math.max(up, -LIMIT), LIMIT);
        return g / (1.0f + TornadoMath.exp(-ALPHA * g)) * (u + 1.0f);
    }

    /**
     * The Q, K and V biases, NeoX RoPE from the precomputed (YaRN) tables, and the key and value
     * written to the paged cache. One thread per (head, pair); keys exist for the first {@code
     * kvHeads} heads only.
     */
    public static void ropeAndCache(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray bq,
            FloatArray bk,
            FloatArray bv,
            FloatArray cosTable,
            FloatArray sinTable,
            FloatArray keyCache,
            FloatArray valueCache,
            int heads,
            int kvHeads,
            int headSize,
            int kvDim,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int half = headSize / 2;
        int id = ctx.globalIdx;
        if (id < heads * half) {
            int h = id / half;
            int i = id - h * half;
            int pos = positionHolder.get(0);
            float c = cosTable.get(pos * half + i);
            float s = sinTable.get(pos * half + i);
            int o = h * headSize + i;
            float q0 = q.get(o) + bq.get(o);
            float q1 = q.get(o + half) + bq.get(o + half);
            q.set(o, q0 * c - q1 * s);
            q.set(o + half, q0 * s + q1 * c);
            if (h < kvHeads) {
                float k0 = k.get(o) + bk.get(o);
                float k1 = k.get(o + half) + bk.get(o + half);
                float r0 = k0 * c - k1 * s;
                float r1 = k0 * s + k1 * c;
                int slot = positionHolder.get(1);
                int base =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(base + o, r0);
                keyCache.set(base + o + half, r1);
                valueCache.set(base + o, v.get(o) + bv.get(o));
                valueCache.set(base + o + half, v.get(o + half) + bv.get(o + half));
            }
        }
    }

    /**
     * The top {@code used} router logits and a softmax over just those, serially in one thread: the
     * expert count is at most a few hundred. {@code logits} is consumed (chosen entries masked).
     */
    public static void topKSoftmax(
            KernelContext ctx,
            FloatArray logits,
            IntArray selected,
            FloatArray weights,
            int experts,
            int used) {
        if (ctx.globalIdx == 0) {
            for (int slot = 0; slot < used; slot++) {
                float best = Float.NEGATIVE_INFINITY;
                int index = 0;
                for (int e = 0; e < experts; e++) {
                    if (logits.get(e) > best) {
                        best = logits.get(e);
                        index = e;
                    }
                }
                selected.set(slot, index);
                weights.set(slot, best);
                logits.set(index, Float.NEGATIVE_INFINITY);
            }
            float max = weights.get(0);
            float sum = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                float e = TornadoMath.exp(weights.get(slot) - max);
                weights.set(slot, e);
                sum += e;
            }
            for (int slot = 0; slot < used; slot++) {
                weights.set(slot, weights.get(slot) / sum);
            }
        }
    }

    /** {@link #ropeAndCache} over an FP16 key/value cache. */
    public static void ropeAndCacheFP16(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray bq,
            FloatArray bk,
            FloatArray bv,
            FloatArray cosTable,
            FloatArray sinTable,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            int heads,
            int kvHeads,
            int headSize,
            int kvDim,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int half = headSize / 2;
        int id = ctx.globalIdx;
        if (id < heads * half) {
            int h = id / half;
            int i = id - h * half;
            int pos = positionHolder.get(0);
            float c = cosTable.get(pos * half + i);
            float s = sinTable.get(pos * half + i);
            int o = h * headSize + i;
            float q0 = q.get(o) + bq.get(o);
            float q1 = q.get(o + half) + bq.get(o + half);
            q.set(o, q0 * c - q1 * s);
            q.set(o + half, q0 * s + q1 * c);
            if (h < kvHeads) {
                float k0 = k.get(o) + bk.get(o);
                float k1 = k.get(o + half) + bk.get(o + half);
                float r0 = k0 * c - k1 * s;
                float r1 = k0 * s + k1 * c;
                int slot = positionHolder.get(1);
                int base =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(base + o, new HalfFloat(r0));
                keyCache.set(base + o + half, new HalfFloat(r1));
                valueCache.set(base + o, new HalfFloat(v.get(o) + bv.get(o)));
                valueCache.set(base + o + half, new HalfFloat(v.get(o + half) + bv.get(o + half)));
            }
        }
    }

    /**
     * The most position splits of the split-KV decode attention: the work-groups launched per head
     * (or per key/value head). How many hold positions depends on the position: {@link
     * #activeSplits}.
     */
    public static final int SPLITS = 32;

    /** The fewest positions a split takes before another split is used. */
    public static final int SPLIT_SPAN = 128;

    /**
     * The splits that hold positions for an attended span of {@code length} positions: one per
     * {@link #SPLIT_SPAN} positions, at most {@link #SPLITS}. A short context runs few splits, so
     * the merge reads few partials; a long one spreads over every work-group.
     */
    static int activeSplits(int length) {
        return Math.min(SPLITS, Math.max(1, (length + SPLIT_SPAN - 1) / SPLIT_SPAN));
    }

    /** Floats one split writes: its {@link #HEAD} accumulators, its maximum and its sum. */
    public static final int PARTIAL = HEAD + 2;

    /**
     * Grouped-query attention for one token, split over the attended positions (flash decoding):
     * work-group {@code g} is split {@code g % SPLITS} of head {@code g / SPLITS}, scores its share
     * of the positions in steps of {@link #HEAD} (a thread a position), folds them into an online
     * softmax with each thread accumulating its own output dimension, and writes the unnormalized
     * partial ({@link #PARTIAL} floats). {@code window} is the sliding window in positions, the
     * current one included; 0 attends to the whole context. {@link #combineSplitsQuantize} merges
     * the splits with the sink. A split with no positions writes an empty partial.
     */
    public static void attentionSplit(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray partials,
            int heads,
            int kvDim,
            int kvMul,
            int window,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int g = ctx.groupIdx;
        int h = g / SPLITS;
        int split = g - h * SPLITS;
        int t = ctx.localIdx;
        float[] qs = ctx.allocateFloatLocalArray(HEAD);
        float[] p = ctx.allocateFloatLocalArray(HEAD);
        float[] red = ctx.allocateFloatLocalArray(HEAD);
        if (h < heads) {
            int pos = positionHolder.get(0);
            int slot = positionHolder.get(1);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int length = pos - start + 1;
            int active = activeSplits(length);
            int per = (length + active - 1) / active;
            int first = start + split * per;
            int last = Math.min(first + per, pos + 1) - 1;
            int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
            int kvHead = (h / kvMul) * HEAD;
            float scale = 1.0f / TornadoMath.sqrt((float) HEAD);
            qs[t] = q.get(h * HEAD + t);
            ctx.localBarrier();

            float runMax = MASKED;
            float runSum = 0.0f;
            float acc = 0.0f;
            for (int c = first; c <= last; c += HEAD) {
                int pt = c + t;
                int valid = 1 - ((last - pt) >>> 31);
                int pc = Math.min(pt, last);
                int kb =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        pc,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHead;
                float dot = 0.0f;
                for (int d = 0; d < HEAD; d++) {
                    dot += qs[d] * keyCache.get(kb + d);
                }
                float score = dot * scale * valid + MASKED * (1 - valid);
                red[t] = score;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = Math.max(red[t], red[t + stride]);
                    }
                    ctx.localBarrier();
                }
                float newMax = Math.max(runMax, red[0]);
                ctx.localBarrier();
                float e = TornadoMath.exp(score - newMax);
                p[t] = e;
                red[t] = e;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = red[t] + red[t + stride];
                    }
                    ctx.localBarrier();
                }
                float correction = TornadoMath.exp(runMax - newMax);
                runSum = runSum * correction + red[0];
                acc = acc * correction;
                runMax = newMax;
                int n = Math.min(HEAD, last - c + 1);
                for (int i = 0; i < n; i++) {
                    int vb =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            c + i,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead;
                    acc += p[i] * valueCache.get(vb + t);
                }
                ctx.localBarrier();
            }
            int base = g * PARTIAL;
            partials.set(base + t, acc);
            if (t == 0) {
                partials.set(base + HEAD, runMax);
                partials.set(base + HEAD + 1, runSum);
            }
        }
    }

    /** {@link #attentionSplit} over an FP16 key/value cache. */
    public static void attentionSplitFP16(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray q,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray partials,
            int heads,
            int kvDim,
            int kvMul,
            int window,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int g = ctx.groupIdx;
        int h = g / SPLITS;
        int split = g - h * SPLITS;
        int t = ctx.localIdx;
        float[] qs = ctx.allocateFloatLocalArray(HEAD);
        float[] p = ctx.allocateFloatLocalArray(HEAD);
        float[] red = ctx.allocateFloatLocalArray(HEAD);
        if (h < heads) {
            int pos = positionHolder.get(0);
            int slot = positionHolder.get(1);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int length = pos - start + 1;
            int active = activeSplits(length);
            int per = (length + active - 1) / active;
            int first = start + split * per;
            int last = Math.min(first + per, pos + 1) - 1;
            int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
            int kvHead = (h / kvMul) * HEAD;
            float scale = 1.0f / TornadoMath.sqrt((float) HEAD);
            qs[t] = q.get(h * HEAD + t);
            ctx.localBarrier();

            float runMax = MASKED;
            float runSum = 0.0f;
            float acc = 0.0f;
            for (int c = first; c <= last; c += HEAD) {
                int pt = c + t;
                int valid = 1 - ((last - pt) >>> 31);
                int pc = Math.min(pt, last);
                int kb =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        pc,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHead;
                float dot = 0.0f;
                for (int d = 0; d < HEAD; d++) {
                    dot += qs[d] * keyCache.get(kb + d).getFloat32();
                }
                float score = dot * scale * valid + MASKED * (1 - valid);
                red[t] = score;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = Math.max(red[t], red[t + stride]);
                    }
                    ctx.localBarrier();
                }
                float newMax = Math.max(runMax, red[0]);
                ctx.localBarrier();
                float e = TornadoMath.exp(score - newMax);
                p[t] = e;
                red[t] = e;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = red[t] + red[t + stride];
                    }
                    ctx.localBarrier();
                }
                float correction = TornadoMath.exp(runMax - newMax);
                runSum = runSum * correction + red[0];
                acc = acc * correction;
                runMax = newMax;
                int n = Math.min(HEAD, last - c + 1);
                for (int i = 0; i < n; i++) {
                    int vb =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            c + i,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead;
                    acc += p[i] * valueCache.get(vb + t).getFloat32();
                }
                ctx.localBarrier();
            }
            int base = g * PARTIAL;
            partials.set(base + t, acc);
            if (t == 0) {
                partials.set(base + HEAD, runMax);
                partials.set(base + HEAD + 1, runSum);
            }
        }
    }

    /** Threads of the fused-norm and staged-expert work-groups. */
    public static final int GROUP = 256;

    /** Expert rows one staged work-group computes: eight warps, a row each. */
    public static final int EXPERT_ROWS = 8;

    /** The widest vector the staged kernels hold in shared memory, padded one float per 32. */
    public static final int MAX_STAGED = 3072;

    private static final int STAGED = MAX_STAGED + MAX_STAGED / 32;

    /** Stages {@code n} floats of {@code x} from {@code offset} into {@code xs}, one pad per 32. */
    static void stage(KernelContext ctx, float[] xs, FloatArray x, int offset, int n) {
        for (int i = ctx.localIdx; i < n; i += GROUP) {
            xs[i + (i >> 5)] = x.get(offset + i);
        }
        ctx.localBarrier();
    }

    private static final int WARPS = GROUP / WARP;

    /** Fills the shared nibble table with twice the E2M1 values. */
    static void stageLut(KernelContext ctx, float[] lut) {
        if (ctx.localIdx < 16) {
            lut[ctx.localIdx] = fp4(ctx.localIdx);
        }
    }

    /**
     * Router logits plus bias, F32 weights, one work-group of {@link #GROUP} threads per expert: a
     * few experts' long rows spread over the whole device rather than one warp each.
     */
    public static void routerRows(
            KernelContext ctx,
            FloatArray in,
            FloatArray w,
            FloatArray bias,
            FloatArray out,
            int experts,
            int dim) {
        int e = ctx.groupIdx;
        int lid = ctx.localIdx;
        float[] red = ctx.allocateFloatLocalArray(GROUP);
        float partial = 0.0f;
        if (e < experts) {
            for (int j = lid; j < dim; j += GROUP) {
                partial += w.get(e * dim + j) * in.get(j);
            }
        }
        red[lid] = partial;
        ctx.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                red[lid] = red[lid] + red[lid + stride];
            }
            ctx.localBarrier();
        }
        if (lid == 0 && e < experts) {
            out.set(e, red[0] + bias.get(e));
        }
    }

    /**
     * Query heads one {@link #attentionGroupSplitFP16} work-group serves: a key/value head's group.
     */
    public static final int GROUP_HEADS = 8;

    private static final int KEY_STRIDE = HEAD + 1;

    /**
     * {@link #attentionSplit} for a whole grouped-query group at once: work-group {@code g} is
     * split {@code g % SPLITS} of key/value head {@code g / SPLITS}, and its {@link #GROUP} threads
     * serve that head's {@link #GROUP_HEADS} query heads. Each step loads {@link #HEAD} positions
     * of keys and values into shared memory once, with coalesced row loads, for all the group's
     * heads; a warp per head then folds the step's scores into that head's online softmax, and
     * every thread accumulates two of the group's output elements. Writes the same partials as
     * {@link #attentionSplit}. Requires {@code kvMul == GROUP_HEADS}.
     */
    public static void attentionGroupSplitFP16(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray q,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray partials,
            int kvHeads,
            int kvDim,
            int window,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int g = ctx.groupIdx;
        int kvh = g / SPLITS;
        int split = g - kvh * SPLITS;
        int t = ctx.localIdx;
        int lane = t & (WARP - 1);
        int warp = t / WARP;
        float[] qs = ctx.allocateFloatLocalArray(GROUP_HEADS * HEAD);
        float[] ks = ctx.allocateFloatLocalArray(HEAD * KEY_STRIDE);
        float[] vs = ctx.allocateFloatLocalArray(HEAD * HEAD);
        float[] sc = ctx.allocateFloatLocalArray(GROUP_HEADS * HEAD);
        float[] stat = ctx.allocateFloatLocalArray(3 * GROUP_HEADS);
        if (kvh < kvHeads) {
            int pos = positionHolder.get(0);
            int slot = positionHolder.get(1);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int length = pos - start + 1;
            int active = activeSplits(length);
            int per = (length + active - 1) / active;
            int first = start + split * per;
            int last = Math.min(first + per, pos + 1) - 1;
            int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
            float scale = 1.0f / TornadoMath.sqrt((float) HEAD);
            int qBase = kvh * GROUP_HEADS * HEAD;
            qs[t] = q.get(qBase + t);
            qs[t + GROUP] = q.get(qBase + t + GROUP);
            if (t < GROUP_HEADS) {
                stat[t] = MASKED; // running maximum
                stat[GROUP_HEADS + t] = 0.0f; // running sum
                stat[2 * GROUP_HEADS + t] = 1.0f; // this step's correction
            }
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            int o0 = t;
            int o1 = t + GROUP;
            int h0 = o0 / HEAD;
            int h1 = o1 / HEAD;
            int d0 = o0 - h0 * HEAD;
            int d1 = o1 - h1 * HEAD;
            ctx.localBarrier();
            for (int c = first; c <= last; c += HEAD) {
                // Keys and values of positions c .. c + 63: a thread four rows' sixteen columns.
                for (int e = t; e < HEAD * HEAD / 4; e += GROUP) {
                    int p = e / (HEAD / 4);
                    int col = (e - p * (HEAD / 4)) * 4;
                    int pc = Math.min(c + p, last);
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            pc,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvh * HEAD
                                    + col;
                    for (int i = 0; i < 4; i++) {
                        ks[p * KEY_STRIDE + col + i] = keyCache.get(base + i).getFloat32();
                        vs[p * HEAD + col + i] = valueCache.get(base + i).getFloat32();
                    }
                }
                ctx.localBarrier();
                // Scores: element e is (head e / 64, position e % 64).
                for (int e = t; e < GROUP_HEADS * HEAD; e += GROUP) {
                    int h = e / HEAD;
                    int p = e - h * HEAD;
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD; d++) {
                        dot += qs[h * HEAD + d] * ks[p * KEY_STRIDE + d];
                    }
                    int valid = 1 - ((last - c - p) >>> 31);
                    sc[e] = dot * scale * valid + MASKED * (1 - valid);
                }
                ctx.localBarrier();
                // A warp per head: the step's maximum, its probabilities, their sum.
                float s0 = sc[warp * HEAD + lane];
                float s1 = sc[warp * HEAD + lane + WARP];
                float m = Math.max(s0, s1);
                m = Math.max(m, ctx.simdShuffleDown(m, 16));
                m = Math.max(m, ctx.simdShuffleDown(m, 8));
                m = Math.max(m, ctx.simdShuffleDown(m, 4));
                m = Math.max(m, ctx.simdShuffleDown(m, 2));
                m = Math.max(m, ctx.simdShuffleDown(m, 1));
                m = ctx.simdBroadcastFirst(m);
                float runMax = stat[warp];
                float newMax = Math.max(runMax, m);
                float e0 = TornadoMath.exp(s0 - newMax);
                float e1 = TornadoMath.exp(s1 - newMax);
                sc[warp * HEAD + lane] = e0;
                sc[warp * HEAD + lane + WARP] = e1;
                float sum = warpSum(ctx, e0 + e1);
                if (lane == 0) {
                    float correction = TornadoMath.exp(runMax - newMax);
                    stat[warp] = newMax;
                    stat[GROUP_HEADS + warp] = stat[GROUP_HEADS + warp] * correction + sum;
                    stat[2 * GROUP_HEADS + warp] = correction;
                }
                ctx.localBarrier();
                float c0 = stat[2 * GROUP_HEADS + h0];
                float c1 = stat[2 * GROUP_HEADS + h1];
                float a0 = 0.0f;
                float a1 = 0.0f;
                for (int p = 0; p < HEAD; p++) {
                    a0 += sc[h0 * HEAD + p] * vs[p * HEAD + d0];
                    a1 += sc[h1 * HEAD + p] * vs[p * HEAD + d1];
                }
                acc0 = acc0 * c0 + a0;
                acc1 = acc1 * c1 + a1;
                ctx.localBarrier();
            }
            int head0 = kvh * GROUP_HEADS + h0;
            int head1 = kvh * GROUP_HEADS + h1;
            partials.set((head0 * SPLITS + split) * PARTIAL + d0, acc0);
            partials.set((head1 * SPLITS + split) * PARTIAL + d1, acc1);
            if (t < GROUP_HEADS) {
                int base = ((kvh * GROUP_HEADS + t) * SPLITS + split) * PARTIAL;
                partials.set(base + HEAD, stat[t]);
                partials.set(base + HEAD + 1, stat[GROUP_HEADS + t]);
            }
        }
    }

    /** Four bytes of a little-endian int stream starting {@code shift} bits into {@code lo}. */
    static int funnel(int lo, int hi, int shift) {
        long both = (lo & 0xFFFFFFFFL) | ((long) hi << 32);
        return (int) (both >>> shift);
    }

    /** Two weights a byte, four bytes of a block's quants, against the staged vector. */
    static float wordDot(int word, float[] lut, float[] xs, int xb, int j0) {
        float sum = 0.0f;
        for (int c = 0; c < 4; c++) {
            int by = (word >>> (c << 3)) & 0xFF;
            sum += lut[by & 15] * xs[xb + j0 + c] + lut[by >> 4] * xs[xb + j0 + c + 16];
        }
        return sum;
    }

    /**
     * One lane's whole blocks of an MXFP4 row, the tensor read as ints: a block's 17 bytes are the
     * five ints covering them, the exponent and the four quant words funnelled out of those.
     */
    static float mxfp4IntDot(IntArray w, int row, float[] lut, float[] xs, int cols, int lane) {
        int blocks = cols / 32;
        int rowByte = row * blocks * MXFP4_BYTES;
        float sum = 0.0f;
        for (int b = lane; b < blocks; b += WARP) {
            int start = rowByte + b * MXFP4_BYTES;
            int i = start >> 2;
            int lead = (start & 3) << 3;
            int w0 = w.get(i);
            int w1 = w.get(i + 1);
            int w2 = w.get(i + 2);
            int w3 = w.get(i + 3);
            int w4 = w.get(i + 4);
            int shift = lead + 8;
            int xb = b * 33;
            float block =
                    wordDot(funnel(w0, w1, shift), lut, xs, xb, 0)
                            + wordDot(funnel(w1, w2, shift), lut, xs, xb, 4)
                            + wordDot(funnel(w2, w3, shift), lut, xs, xb, 8)
                            + wordDot(funnel(w3, w4, shift), lut, xs, xb, 12);
            sum += e8m0Half((w0 >>> lead) & 0xFF) * block;
        }
        return sum;
    }

    /**
     * Every chosen expert's gate and up rows plus biases, through the clamped SwiGLU, into {@code
     * hidden[slot * hiddenDim + i]}, from MXFP4 experts read straight from global memory as ints
     * ({@link #mxfp4IntDot}, a lane per block). The normed input and the nibble table are staged in
     * shared memory once per work-group of {@link #EXPERT_ROWS} rows. Worker: {@code ceil(used *
     * hiddenDim / EXPERT_ROWS)} groups of {@link #GROUP}.
     */
    public static void expertsGateUpPacked(
            KernelContext ctx,
            FloatArray xb,
            IntArray selected,
            IntArray gate,
            IntArray up,
            FloatArray gateBias,
            FloatArray upBias,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int used) {
        float[] xs = ctx.allocateFloatLocalArray(STAGED);
        float[] lut = ctx.allocateFloatLocalArray(16);
        stageLut(ctx, lut);
        stage(ctx, xs, xb, 0, dim);
        int lane = ctx.localIdx & (WARP - 1);
        int warp = ctx.localIdx / WARP;
        for (int r = 0; r < EXPERT_ROWS / WARPS; r++) {
            int row = ctx.groupIdx * EXPERT_ROWS + warp * (EXPERT_ROWS / WARPS) + r;
            if (row < used * hiddenDim) {
                int slot = row / hiddenDim;
                int i = row - slot * hiddenDim;
                int wrow = selected.get(slot) * hiddenDim + i;
                float g = warpSum(ctx, mxfp4IntDot(gate, wrow, lut, xs, dim, lane));
                float u = warpSum(ctx, mxfp4IntDot(up, wrow, lut, xs, dim, lane));
                if (lane == 0) {
                    hidden.set(row, swiglu(g + gateBias.get(wrow), u + upBias.get(wrow)));
                }
            }
        }
    }

    /**
     * Each chosen expert's down rows plus bias, unweighted, from MXFP4 experts read as ints: {@code
     * partial[slot * dim + r] = D_slot[r]·hidden_slot + b_slot[r]}; {@link #expertsCombineBias}
     * weighs and adds them. A work-group's {@link #EXPERT_ROWS} rows belong to one slot, whose
     * hidden vector it stages. Requires {@code dim % EXPERT_ROWS == 0}. Worker: {@code used * dim /
     * EXPERT_ROWS} groups of {@link #GROUP}.
     */
    public static void expertsDownPacked(
            KernelContext ctx,
            FloatArray hidden,
            IntArray selected,
            IntArray down,
            FloatArray downBias,
            FloatArray partial,
            int dim,
            int hiddenDim,
            int used) {
        float[] xs = ctx.allocateFloatLocalArray(STAGED);
        float[] lut = ctx.allocateFloatLocalArray(16);
        int first = ctx.groupIdx * EXPERT_ROWS;
        int slot = Math.min(first / dim, used - 1);
        stageLut(ctx, lut);
        stage(ctx, xs, hidden, slot * hiddenDim, hiddenDim);
        int lane = ctx.localIdx & (WARP - 1);
        int warp = ctx.localIdx / WARP;
        int expert = selected.get(slot);
        for (int r = 0; r < EXPERT_ROWS / WARPS; r++) {
            int row = first + warp * (EXPERT_ROWS / WARPS) + r;
            if (row < used * dim) {
                int wrow = expert * dim + row - slot * dim;
                float d = warpSum(ctx, mxfp4IntDot(down, wrow, lut, xs, hiddenDim, lane));
                if (lane == 0) {
                    partial.set(row, d + downBias.get(wrow));
                }
            }
        }
    }

    /**
     * Quantizes this warp's 32 values (one Q8 block, lane {@code j} holding element {@code j}) as
     * {@code TransformerComputeKernelsQ4_0.quantizeActivationQ8Blocks} does: {@code d = amax /
     * 127}, quants rounded half away from zero, four packed an int. Writes block {@code block}'s
     * ints and scale.
     */
    static void quantizeWarp(
            KernelContext ctx, float v, IntArray quants, FloatArray scales, int block, int lane) {
        float amax = TornadoMath.abs(v);
        amax = Math.max(amax, ctx.simdShuffleDown(amax, 16));
        amax = Math.max(amax, ctx.simdShuffleDown(amax, 8));
        amax = Math.max(amax, ctx.simdShuffleDown(amax, 4));
        amax = Math.max(amax, ctx.simdShuffleDown(amax, 2));
        amax = Math.max(amax, ctx.simdShuffleDown(amax, 1));
        amax = ctx.simdBroadcastFirst(amax);
        float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        float scaled = v * inverse;
        int q = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        q = Math.min(127, Math.max(-127, q));
        float qf = (float) q;
        int q1 = (int) ctx.simdShuffleDown(qf, 1);
        int q2 = (int) ctx.simdShuffleDown(qf, 2);
        int q3 = (int) ctx.simdShuffleDown(qf, 3);
        if ((lane & 3) == 0) {
            quants.set(
                    block * 8 + (lane >> 2),
                    (q & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24));
        }
        if (lane == 0) {
            scales.set(block, amax / 127.0f);
        }
    }

    /**
     * The RMS norm of {@code x} times {@code weight}, quantized straight to Q8 blocks for the
     * packed-integer projections: no FP32 normed vector is written. Every work-group reduces the
     * whole vector and quantizes its own {@link #GROUP} elements. Requires {@code size % 32 == 0}.
     */
    public static void normQuantize(
            KernelContext ctx,
            FloatArray x,
            FloatArray weight,
            IntArray quants,
            FloatArray scales,
            int size,
            float eps) {
        int lid = ctx.localIdx;
        float[] red = ctx.allocateFloatLocalArray(GROUP);
        float partial = 0.0f;
        for (int j = lid; j < size; j += GROUP) {
            float v = x.get(j);
            partial += v * v;
        }
        red[lid] = partial;
        ctx.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                red[lid] = red[lid] + red[lid + stride];
            }
            ctx.localBarrier();
        }
        float scale = 1.0f / TornadoMath.sqrt(red[0] / size + eps);
        int i = ctx.globalIdx;
        if (i < size) {
            quantizeWarp(ctx, weight.get(i) * scale * x.get(i), quants, scales, i >> 5, lid & 31);
        }
    }

    /**
     * Merges the {@link #SPLITS} partials of each head with its sink ({@code out = sum_s acc_s
     * e^(m_s - M) / (sum_s l_s e^(m_s - M) + e^(sink - M))}, {@code M} the largest of the splits'
     * maxima and the sink) and quantizes the attention output straight to Q8 blocks for the output
     * projection. A head's 64 outputs are two whole blocks. Reads only the splits that held
     * positions ({@link #activeSplits} of the span the attention covered).
     */
    public static void combineSplitsQuantize(
            KernelContext ctx,
            IntArray positionHolder,
            FloatArray partials,
            FloatArray sinks,
            IntArray quants,
            FloatArray scales,
            int heads,
            int window) {
        int id = ctx.globalIdx;
        if (id < heads * HEAD) {
            int pos = positionHolder.get(0);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int active = activeSplits(pos - start + 1);
            int h = id / HEAD;
            int t = id - h * HEAD;
            float sink = sinks.get(h);
            float max = sink;
            for (int s = 0; s < active; s++) {
                max = Math.max(max, partials.get((h * SPLITS + s) * PARTIAL + HEAD));
            }
            float sum = TornadoMath.exp(sink - max);
            float acc = 0.0f;
            for (int s = 0; s < active; s++) {
                int base = (h * SPLITS + s) * PARTIAL;
                float w = TornadoMath.exp(partials.get(base + HEAD) - max);
                sum += partials.get(base + HEAD + 1) * w;
                acc += partials.get(base + t) * w;
            }
            quantizeWarp(ctx, acc / sum, quants, scales, id >> 5, ctx.localIdx & 31);
        }
    }

    /**
     * {@code out = weight * (x + bias) / rms(x + bias)}: the feed-forward norm reads the residual
     * with the attention output's bias, which {@link #expertsCombineBias} then adds for good. Every
     * work-group reduces the whole vector and writes its own {@link #GROUP} elements.
     */
    public static void normFusedBias(
            KernelContext ctx,
            FloatArray out,
            FloatArray x,
            FloatArray bias,
            FloatArray weight,
            int size,
            float eps) {
        int lid = ctx.localIdx;
        float[] red = ctx.allocateFloatLocalArray(GROUP);
        float partial = 0.0f;
        for (int j = lid; j < size; j += GROUP) {
            float v = x.get(j) + bias.get(j);
            partial += v * v;
        }
        red[lid] = partial;
        ctx.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                red[lid] = red[lid] + red[lid + stride];
            }
            ctx.localBarrier();
        }
        float scale = 1.0f / TornadoMath.sqrt(red[0] / size + eps);
        int i = ctx.globalIdx;
        if (i < size) {
            out.set(i, weight.get(i) * scale * (x.get(i) + bias.get(i)));
        }
    }

    /**
     * {@code x[r] += bias[r] + sum_slot weight[slot] * partial[slot * dim + r]}: the attention
     * output's bias, deferred from the projection, and the experts' weighted sum.
     */
    public static void expertsCombineBias(
            KernelContext ctx,
            FloatArray partial,
            FloatArray weights,
            FloatArray bias,
            FloatArray x,
            int dim,
            int used) {
        int r = ctx.globalIdx;
        if (r < dim) {
            float acc = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                acc += weights.get(slot) * partial.get(slot * dim + r);
            }
            x.set(r, x.get(r) + bias.get(r) + acc);
        }
    }
}
