package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
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

    /** {@code 2^(e - 128)}, the E8M0 exponent halved; exponents below 2 are read as 2. */
    static float e8m0Half(int e) {
        int clamped = Math.max(e, 2);
        return Float.intBitsToFloat((clamped - 1) << 23);
    }

    /** gpt-oss's clamped SwiGLU. */
    static float swiglu(float gate, float up) {
        float g = Math.min(gate, LIMIT);
        float u = Math.min(Math.max(up, -LIMIT), LIMIT);
        return g / (1.0f + TornadoMath.exp(-ALPHA * g)) * (u + 1.0f);
    }

    /** One lane's share of an MXFP4 row dot product, {@code xOffset} into {@code x}. */
    /** {@code out[i] = weight[i] * scale * x[i]}, the RMS scale read from {@code scale[0]}. */
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
     * Grouped-query attention for one token with a learned sink per head and an optional sliding
     * window ({@code window} positions, the current one included; 0 = the whole context).
     *
     * <p>One work-group of {@link #HEAD} threads per query head. Each step scores up to {@link
     * #HEAD} positions (a thread a position), folds them into an online softmax, and each thread
     * accumulates its own output dimension. The sink joins the maximum and the denominator last.
     */
    /** {@code x[row] += W·in + bias[row]}, Q8_0, one warp per row. */
    /** Router logits: {@code out[e] = R[e]·in + bias[e]}, F32 weights, one warp per expert. */
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

    /**
     * Every chosen expert's gate and up rows (MXFP4) plus biases, through the clamped SwiGLU, into
     * {@code hidden[slot * hiddenDim + i]}. One warp per (slot, row).
     */
    /**
     * The chosen experts' down rows (MXFP4) plus bias, weighted by the routing weights, summed and
     * added to the residual: {@code x[row] += sum_k w_k (D_k[row]·hidden_k + b_k[row])}.
     */
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

    /** Position splits of the split-KV decode attention. */
    public static final int SPLITS = 8;

    /** Floats one split writes: its {@link #HEAD} accumulators, its maximum and its sum. */
    public static final int PARTIAL = HEAD + 2;

    /**
     * Grouped-query attention for one token, split over the attended positions (flash decoding):
     * work-group {@code g} is split {@code g % SPLITS} of head {@code g / SPLITS}, scores its share
     * of the positions in steps of {@link #HEAD} (a thread a position), folds them into an online
     * softmax with each thread accumulating its own output dimension, and writes the unnormalized
     * partial ({@link #PARTIAL} floats). {@code window} is the sliding window in positions, the
     * current one included; 0 attends to the whole context. {@link #combineSplits} merges the
     * splits with the sink. A split with no positions writes an empty partial.
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
            int per = (length + SPLITS - 1) / SPLITS;
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

    /**
     * Merges the {@link #SPLITS} partials of each head with its sink: {@code out = sum_s acc_s
     * e^(m_s - M) / (sum_s l_s e^(m_s - M) + e^(sink - M))}, {@code M} the largest of the splits'
     * maxima and the sink. One thread per output element.
     */
    public static void combineSplits(
            KernelContext ctx, FloatArray partials, FloatArray sinks, FloatArray out, int heads) {
        int id = ctx.globalIdx;
        if (id < heads * HEAD) {
            int h = id / HEAD;
            int t = id - h * HEAD;
            float sink = sinks.get(h);
            float max = sink;
            for (int s = 0; s < SPLITS; s++) {
                max = Math.max(max, partials.get((h * SPLITS + s) * PARTIAL + HEAD));
            }
            float sum = TornadoMath.exp(sink - max);
            float acc = 0.0f;
            for (int s = 0; s < SPLITS; s++) {
                int base = (h * SPLITS + s) * PARTIAL;
                float w = TornadoMath.exp(partials.get(base + HEAD) - max);
                sum += partials.get(base + HEAD + 1) * w;
                acc += partials.get(base + t) * w;
            }
            out.set(id, acc / sum);
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
            int per = (length + SPLITS - 1) / SPLITS;
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

    /** Expert rows one staged work-group computes: eight warps, four rows each. */
    public static final int EXPERT_ROWS = 32;

    /** The widest vector the staged kernels hold in shared memory, padded one float per 32. */
    public static final int MAX_STAGED = 3072;

    private static final int STAGED = MAX_STAGED + MAX_STAGED / 32;

    /**
     * {@code out[i] = weight[i] * x[i] / rms(x)}: every work-group reduces the whole vector (it is
     * a few thousand floats) and writes its own {@link #GROUP} elements, so no second launch waits
     * on a scale.
     */
    public static void normFused(
            KernelContext ctx,
            FloatArray out,
            FloatArray x,
            FloatArray weight,
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
            out.set(i, weight.get(i) * scale * x.get(i));
        }
    }

    /** Stages {@code n} floats of {@code x} from {@code offset} into {@code xs}, one pad per 32. */
    static void stage(KernelContext ctx, float[] xs, FloatArray x, int offset, int n) {
        for (int i = ctx.localIdx; i < n; i += GROUP) {
            xs[i + (i >> 5)] = x.get(offset + i);
        }
        ctx.localBarrier();
    }

    /**
     * One lane's share of an MXFP4 row dot product against a staged vector: whole 17-byte blocks,
     * the lane's blocks {@code lane, lane + 32, ...}. The staged vector is padded so the lanes'
     * reads of one element index fall in different banks.
     */
    /**
     * {@link #expertsGateUpMXFP4} with the normed input staged in shared memory once per work-group
     * of {@link #EXPERT_ROWS} rows, and a lane per whole block. Worker: {@code ceil(used *
     * hiddenDim / 32)} groups of {@link #GROUP}.
     */
    /**
     * Each chosen expert's down rows (MXFP4) plus bias, unweighted: {@code partial[slot * dim + r]
     * = D_slot[r]·hidden_slot + b_slot[r]}. A work-group's {@link #EXPERT_ROWS} rows belong to one
     * slot, whose hidden vector it stages. Requires {@code dim % 32 == 0}. Worker: {@code used *
     * dim / 32} groups of {@link #GROUP}.
     */
    /** {@code x[r] += sum_slot weight[slot] * partial[slot * dim + r]}, the slots in order. */
    public static void expertsCombine(
            KernelContext ctx,
            FloatArray partial,
            FloatArray weights,
            FloatArray x,
            int dim,
            int used) {
        int r = ctx.globalIdx;
        if (r < dim) {
            float acc = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                acc += weights.get(slot) * partial.get(slot * dim + r);
            }
            x.set(r, x.get(r) + acc);
        }
    }

    /** Ints of one staged MXFP4 row: up to {@link #MAX_STAGED} weights, two halfwords an int. */
    private static final int ROW_INTS = MAX_STAGED / 32 * MXFP4_BYTES / 4 + 1;

    private static final int WARPS = GROUP / WARP;

    /**
     * Copies MXFP4 row {@code row} into this warp's slice of {@code rows} with 16-bit loads (a row
     * of {@code cols / 32} 17-byte blocks is 2-byte aligned whenever the block count is even), two
     * halfwords an int, the lanes taking consecutive ints.
     */
    static void stageRow(ByteArray w, int row, int cols, int[] rows, int slice, int lane) {
        int bytes = cols / 32 * MXFP4_BYTES;
        int base = row * bytes;
        int last = bytes / 2 - 1;
        int ints = (last + 2) / 2;
        // Four ints a lane per step, their eight loads issued before any store.
        for (int k0 = lane; k0 < ints; k0 += 4 * WARP) {
            int k1 = k0 + WARP;
            int k2 = k0 + 2 * WARP;
            int k3 = k0 + 3 * WARP;
            int a0 = halfword(w, base, Math.min(2 * k0, last));
            int b0 = halfword(w, base, Math.min(2 * k0 + 1, last));
            int a1 = halfword(w, base, Math.min(2 * k1, last));
            int b1 = halfword(w, base, Math.min(2 * k1 + 1, last));
            int a2 = halfword(w, base, Math.min(2 * k2, last));
            int b2 = halfword(w, base, Math.min(2 * k2 + 1, last));
            int a3 = halfword(w, base, Math.min(2 * k3, last));
            int b3 = halfword(w, base, Math.min(2 * k3 + 1, last));
            rows[slice + k0] = a0 | (b0 << 16);
            if (k1 < ints) {
                rows[slice + k1] = a1 | (b1 << 16);
            }
            if (k2 < ints) {
                rows[slice + k2] = a2 | (b2 << 16);
            }
            if (k3 < ints) {
                rows[slice + k3] = a3 | (b3 << 16);
            }
        }
    }

    /** Halfword {@code i} of the row at byte {@code base}, as an unsigned 16-bit value. */
    static int halfword(ByteArray w, int base, int i) {
        return w.getHalfFloat(base + 2 * i).getHalfFloatValue() & 0xFFFF;
    }

    /** Byte {@code k} of a staged row. */
    static int stagedByte(int[] rows, int slice, int k) {
        return (rows[slice + (k >> 2)] >>> ((k & 3) << 3)) & 0xFF;
    }

    /** Fills the shared nibble table with twice the E2M1 values. */
    static void stageLut(KernelContext ctx, float[] lut) {
        if (ctx.localIdx < 16) {
            lut[ctx.localIdx] = fp4(ctx.localIdx);
        }
    }

    /**
     * A lane's whole blocks of a staged gate row and up row against the padded staged vector
     * ({@link #stage}), accumulated into {@code acc[0]} (gate) and {@code acc[1]} (up). Each x
     * value is read once for both rows; each block's exponents are decoded once.
     */
    static void lanePairDot(
            int[] gRows,
            int[] uRows,
            int slice,
            float[] lut,
            float[] xs,
            int cols,
            int lane,
            float[] acc) {
        int blocks = cols / 32;
        float g = 0.0f;
        float u = 0.0f;
        for (int b = lane; b < blocks; b += WARP) {
            int off = b * MXFP4_BYTES;
            int xb = b * 33;
            float gb = 0.0f;
            float ub = 0.0f;
            for (int j = 0; j < 16; j++) {
                int qg = stagedByte(gRows, slice, off + 1 + j);
                int qu = stagedByte(uRows, slice, off + 1 + j);
                float x0 = xs[xb + j];
                float x1 = xs[xb + j + 16];
                gb += lut[qg & 15] * x0 + lut[qg >> 4] * x1;
                ub += lut[qu & 15] * x0 + lut[qu >> 4] * x1;
            }
            g += e8m0Half(stagedByte(gRows, slice, off)) * gb;
            u += e8m0Half(stagedByte(uRows, slice, off)) * ub;
        }
        acc[0] = g;
        acc[1] = u;
    }

    /** A lane's whole blocks of one staged row against the padded staged vector. */
    static float laneDot(int[] rows, int slice, float[] lut, float[] xs, int cols, int lane) {
        int blocks = cols / 32;
        float sum = 0.0f;
        for (int b = lane; b < blocks; b += WARP) {
            int off = b * MXFP4_BYTES;
            int xb = b * 33;
            float block = 0.0f;
            for (int j = 0; j < 16; j++) {
                int q = stagedByte(rows, slice, off + 1 + j);
                block += lut[q & 15] * xs[xb + j] + lut[q >> 4] * xs[xb + j + 16];
            }
            sum += e8m0Half(stagedByte(rows, slice, off)) * block;
        }
        return sum;
    }

    /**
     * Every chosen expert's gate and up rows (MXFP4) plus biases, through the clamped SwiGLU, into
     * {@code hidden[slot * hiddenDim + i]}. The normed input is staged in shared memory once per
     * work-group of {@link #EXPERT_ROWS} rows; each warp copies its row pair into shared memory
     * ({@link #stageRow}) and decodes it a lane per block ({@link #lanePairDot}). Worker: {@code
     * ceil(used * hiddenDim / EXPERT_ROWS)} groups of {@link #GROUP}.
     */
    public static void expertsGateUpLanes(
            KernelContext ctx,
            FloatArray xb,
            IntArray selected,
            ByteArray gate,
            ByteArray up,
            FloatArray gateBias,
            FloatArray upBias,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int used) {
        float[] xs = ctx.allocateFloatLocalArray(STAGED);
        float[] lut = ctx.allocateFloatLocalArray(16);
        int[] gRows = ctx.allocateIntLocalArray(WARPS * ROW_INTS);
        int[] uRows = ctx.allocateIntLocalArray(WARPS * ROW_INTS);
        float[] acc = new float[2];
        stageLut(ctx, lut);
        stage(ctx, xs, xb, 0, dim);
        int lane = ctx.localIdx & (WARP - 1);
        int warp = ctx.localIdx / WARP;
        int slice = warp * ROW_INTS;
        for (int r = 0; r < EXPERT_ROWS / WARPS; r++) {
            int row = ctx.groupIdx * EXPERT_ROWS + warp * (EXPERT_ROWS / WARPS) + r;
            int valid = 1 - ((used * hiddenDim - 1 - row) >>> 31);
            int safe = Math.min(row, used * hiddenDim - 1);
            int slot = safe / hiddenDim;
            int i = safe - slot * hiddenDim;
            int wrow = selected.get(slot) * hiddenDim + i;
            stageRow(gate, wrow, dim, gRows, slice, lane);
            stageRow(up, wrow, dim, uRows, slice, lane);
            ctx.localBarrier();
            lanePairDot(gRows, uRows, slice, lut, xs, dim, lane, acc);
            float g = warpSum(ctx, acc[0]);
            float u = warpSum(ctx, acc[1]);
            if (lane == 0 && valid == 1) {
                hidden.set(row, swiglu(g + gateBias.get(wrow), u + upBias.get(wrow)));
            }
            ctx.localBarrier();
        }
    }

    /**
     * Each chosen expert's down rows (MXFP4) plus bias, unweighted: {@code partial[slot * dim + r]
     * = D_slot[r]·hidden_slot + b_slot[r]}; {@link #expertsCombine} weighs and adds them. A
     * work-group's {@link #EXPERT_ROWS} rows belong to one slot, whose hidden vector it stages.
     * Requires {@code dim % EXPERT_ROWS == 0}. Worker: {@code used * dim / EXPERT_ROWS} groups of
     * {@link #GROUP}.
     */
    public static void expertsDownLanes(
            KernelContext ctx,
            FloatArray hidden,
            IntArray selected,
            ByteArray down,
            FloatArray downBias,
            FloatArray partial,
            int dim,
            int hiddenDim,
            int used) {
        float[] xs = ctx.allocateFloatLocalArray(STAGED);
        float[] lut = ctx.allocateFloatLocalArray(16);
        int[] dRows = ctx.allocateIntLocalArray(WARPS * ROW_INTS);
        int first = ctx.groupIdx * EXPERT_ROWS;
        int slot = Math.min(first / dim, used - 1);
        stageLut(ctx, lut);
        stage(ctx, xs, hidden, slot * hiddenDim, hiddenDim);
        int lane = ctx.localIdx & (WARP - 1);
        int warp = ctx.localIdx / WARP;
        int slice = warp * ROW_INTS;
        int expert = selected.get(slot);
        for (int r = 0; r < EXPERT_ROWS / WARPS; r++) {
            int row = first + warp * (EXPERT_ROWS / WARPS) + r;
            int valid = 1 - ((used * dim - 1 - row) >>> 31);
            int out = Math.min(row, used * dim - 1) - slot * dim;
            int wrow = expert * dim + out;
            stageRow(down, wrow, hiddenDim, dRows, slice, lane);
            ctx.localBarrier();
            float d = warpSum(ctx, laneDot(dRows, slice, lut, xs, hiddenDim, lane));
            if (lane == 0 && valid == 1) {
                partial.set(row, d + downBias.get(wrow));
            }
            ctx.localBarrier();
        }
    }

    /** {@code x[i] += bias[i]}. */
    public static void addBias(KernelContext ctx, FloatArray x, FloatArray bias, int size) {
        int i = ctx.globalIdx;
        if (i < size) {
            x.set(i, x.get(i) + bias.get(i));
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

    /** Query heads one {@link #attentionGroupSplit} work-group serves: a key/value head's group. */
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
            int per = (length + SPLITS - 1) / SPLITS;
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
}
