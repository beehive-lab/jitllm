package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

// @formatter:off
/**
 * Decode kernels for the blocks {@code deepseek4} (DeepSeek-V4) adds to what glm5next and deepseek2
 * already have: attention over a sliding window plus compressed blocks with per-head sinks, the
 * gated-pooling compressor, the indexer's block scores and selection, the grouped output
 * projection, square-root-softplus routing with hash-routed layers, and the learned mHC head.
 *
 * <p>The hyper-connections, the routed and shared experts and the plain projections are {@link
 * Glm5NextKernels}' and the shared Q8_0 kernels. Rotations are computed from frequency tables: the
 * interleaved pairs {@code (2i, 2i + 1)} of a vector's last {@code ropeDim} values turn by {@code
 * position * freqs[i]}.
 *
 * <p>Every kernel takes the session's {@code positionHolder} where it needs the position (element
 * 0), so one captured graph serves every token. Workgroup reductions assume a power-of-two local
 * size no larger than {@link #GROUP}. Values that depend on a condition are blended arithmetically
 * or written from each branch: the CUDA backend miscompiles a value merged from an if/else.
 */
// @formatter:on
public final class DeepSeek4Kernels {

    /** The widest workgroup the reductions here are written for. */
    public static final int GROUP = 256;

    /** The widest compressor output ({@code headDim}) {@link #compressor} pools. */
    public static final int MAX_WIDTH = 512;

    /** The most experts {@link #routerSqrtSoftplus} routes over. */
    public static final int MAX_EXPERTS = 512;

    /** The entries of {@link #attention}'s shape array. */
    public static final int SHAPE_HEADS = 0;

    public static final int SHAPE_HEAD_DIM = 1;
    public static final int SHAPE_ROPE_DIM = 2;
    public static final int SHAPE_WINDOW = 3;
    public static final int SHAPE_WINDOW_OFFSET = 4;
    public static final int SHAPE_COMPRESSED_OFFSET = 5;
    public static final int SHAPE_STRIDE = 6;
    public static final int SHAPE_RATIO = 7;
    public static final int SHAPE_USE_SELECTION = 8;
    public static final int SHAPE_SELECTION_CAPACITY = 9;
    public static final int SHAPE_SIZE = 10;

    /** The entries of {@link #compressor}'s shape array. */
    public static final int C_RATIO = 0;

    public static final int C_OVERLAP = 1;
    public static final int C_WIDTH = 2;
    public static final int C_ROPE_DIM = 3;
    public static final int C_PENDING_OFFSET = 4;
    public static final int C_CACHE_OFFSET = 5;
    public static final int C_SIZE = 6;

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    private DeepSeek4Kernels() {}

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    private static float sumReduce(KernelContext context, float[] reduce, float value) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        reduce[tid] = value;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float total = reduce[0];
        context.localBarrier();
        return total;
    }

    private static float maxReduce(KernelContext context, float[] reduce, float value) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        reduce[tid] = value;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
            }
            context.localBarrier();
        }
        float total = reduce[0];
        context.localBarrier();
        return total;
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Prepares the query and the key/value of the current position.
     *
     * <ul>
     *   <li>workgroup {@code h < heads}: query head {@code h} RMS-normalized without a weight, then
     *       its tail rotated;
     *   <li>workgroup {@code heads}: the normalized key/value's tail rotated, the vector stored in
     *       the layer's window at slot {@code position % window}.
     * </ul>
     *
     * <p>Grid: {@code heads + 1} workgroups of {@code localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void attentionPrep(
            KernelContext context,
            FloatArray q,
            FloatArray kv,
            FloatArray window,
            FloatArray freqs,
            IntArray positionHolder,
            int heads,
            int headDim,
            int ropeDim,
            int windowSize,
            int windowOffset,
            float eps) {
        int g = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int position = positionHolder.get(0);
        int nope = headDim - ropeDim;
        int pairs = ropeDim / 2;
        if (g < heads) {
            int base = g * headDim;
            float partial = 0.0f;
            for (int i = tid; i < headDim; i += local) {
                float v = q.get(base + i);
                partial += v * v;
            }
            float total = sumReduce(context, reduce, partial);
            float scale = 1.0f / TornadoMath.sqrt(total / headDim + eps);
            for (int i = tid; i < headDim; i += local) {
                q.set(base + i, q.get(base + i) * scale);
            }
            context.localBarrier();
            for (int p = tid; p < pairs; p += local) {
                int a = base + nope + 2 * p;
                float angle = position * freqs.get(p);
                float cos = TornadoMath.cos(angle);
                float sin = TornadoMath.sin(angle);
                float x0 = q.get(a);
                float x1 = q.get(a + 1);
                q.set(a, x0 * cos - x1 * sin);
                q.set(a + 1, x0 * sin + x1 * cos);
            }
        }
        if (g == heads) {
            int slot = windowOffset + (position % windowSize) * headDim;
            for (int i = tid; i < nope; i += local) {
                window.set(slot + i, kv.get(i));
            }
            for (int p = tid; p < pairs; p += local) {
                int a = nope + 2 * p;
                float angle = position * freqs.get(p);
                float cos = TornadoMath.cos(angle);
                float sin = TornadoMath.sin(angle);
                float x0 = kv.get(a);
                float x1 = kv.get(a + 1);
                window.set(slot + a, x0 * cos - x1 * sin);
                window.set(slot + a + 1, x0 * sin + x1 * cos);
            }
        }
    }

    // @formatter:off
    /**
     * One query head's attention over the window and the visible compressed blocks, with the head's
     * sink in the softmax; the key/value is both key and value, and the output's tail is rotated
     * back.
     *
     * <ul>
     *   <li>the window: the last {@code min(position + 1, window)} positions, from the ring;
     *   <li>the blocks: none without compression; for a layer with an indexer ({@code
     *       useSelection}), {@code selected[0 .. count[0])}; otherwise every complete block, {@code
     *       (position + 1) / ratio} of them.
     * </ul>
     *
     * <p>{@code shape} holds the widths and offsets at the {@code SHAPE_*} indices. Head {@code
     * h}'s scores use {@code scratch[h * stride ...]}. Grid: {@code heads} workgroups of {@code
     * localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void attention(
            KernelContext context,
            FloatArray q,
            FloatArray window,
            FloatArray compressed,
            IntArray selected,
            IntArray selectedCount,
            FloatArray sinks,
            FloatArray scratch,
            FloatArray out,
            FloatArray freqs,
            IntArray positionHolder,
            IntArray shape,
            float scale) {
        int h = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int position = positionHolder.get(0);
        int heads = shape.get(SHAPE_HEADS);
        int headDim = shape.get(SHAPE_HEAD_DIM);
        int ropeDim = shape.get(SHAPE_ROPE_DIM);
        int windowSize = shape.get(SHAPE_WINDOW);
        int windowOffset = shape.get(SHAPE_WINDOW_OFFSET);
        int compressedOffset = shape.get(SHAPE_COMPRESSED_OFFSET);
        int stride = shape.get(SHAPE_STRIDE);
        int ratio = shape.get(SHAPE_RATIO);
        int useSelection = shape.get(SHAPE_USE_SELECTION);
        int capacity = shape.get(SHAPE_SELECTION_CAPACITY);

        int windowCount = TornadoMath.min(position + 1, windowSize);
        int first = position - windowCount + 1;
        int compressedLayer = TornadoMath.min(ratio, 1);
        int complete = (position + 1) / TornadoMath.max(ratio, 1);
        int blockCount =
                compressedLayer
                        * (useSelection * selectedCount.get(0) + (1 - useSelection) * complete);
        int scores = h * stride;
        int qBase = h * headDim;
        int head = TornadoMath.min(h, heads - 1);

        float max = Float.NEGATIVE_INFINITY;
        for (int j = tid; j < windowCount; j += local) {
            int key = windowOffset + ((first + j) % windowSize) * headDim;
            float dot = 0.0f;
            for (int i = 0; i < headDim; i++) {
                dot += q.get(qBase + i) * window.get(key + i);
            }
            scratch.set(scores + j, dot * scale);
            max = TornadoMath.max(max, dot * scale);
        }
        for (int j = tid; j < blockCount; j += local) {
            int pick = selected.get(TornadoMath.min(j, capacity - 1));
            int block = useSelection * pick + (1 - useSelection) * j;
            int key = compressedOffset + block * headDim;
            float dot = 0.0f;
            for (int i = 0; i < headDim; i++) {
                dot += q.get(qBase + i) * compressed.get(key + i);
            }
            scratch.set(scores + windowCount + j, dot * scale);
            max = TornadoMath.max(max, dot * scale);
        }
        float sink = sinks.get(head);
        float rowMax = TornadoMath.max(maxReduce(context, reduce, max), sink);
        int entries = windowCount + blockCount;
        float sum = 0.0f;
        for (int j = tid; j < entries; j += local) {
            float e = TornadoMath.exp(scratch.get(scores + j) - rowMax);
            scratch.set(scores + j, e);
            sum += e;
        }
        float total = sumReduce(context, reduce, sum) + TornadoMath.exp(sink - rowMax);

        int nope = headDim - ropeDim;
        for (int p = tid; p < headDim / 2; p += local) {
            int i0 = 2 * p;
            float a0 = 0.0f;
            float a1 = 0.0f;
            for (int j = 0; j < windowCount; j++) {
                int key = windowOffset + ((first + j) % windowSize) * headDim;
                float w = scratch.get(scores + j);
                a0 += w * window.get(key + i0);
                a1 += w * window.get(key + i0 + 1);
            }
            for (int j = 0; j < blockCount; j++) {
                int pick = selected.get(TornadoMath.min(j, capacity - 1));
                int block = useSelection * pick + (1 - useSelection) * j;
                int key = compressedOffset + block * headDim;
                float w = scratch.get(scores + windowCount + j);
                a0 += w * compressed.get(key + i0);
                a1 += w * compressed.get(key + i0 + 1);
            }
            a0 = a0 / total;
            a1 = a1 / total;
            float tail = i0 >= nope ? 1.0f : 0.0f;
            int f = TornadoMath.max(0, (i0 - nope) / 2);
            float angle = position * freqs.get(TornadoMath.min(f, ropeDim / 2 - 1));
            float cos = TornadoMath.cos(angle);
            float sin = TornadoMath.sin(angle);
            float r0 = a0 * cos + a1 * sin;
            float r1 = a1 * cos - a0 * sin;
            if (h < heads) {
                out.set(qBase + i0, tail * r0 + (1.0f - tail) * a0);
                out.set(qBase + i0 + 1, tail * r1 + (1.0f - tail) * a1);
            }
        }
    }

    // @formatter:off
    /**
     * {@code out[r] = W[r] . x[g * groupIn ...]} with {@code g = r / rank}: the grouped output
     * projection, each group of {@code rank} rows reading its own slice of the heads' output, over
     * the activation quantized per block. A warp per row; Q8_0 weight of {@code groupIn} columns.
     * Worker: {@code rows * 32} lanes.
     */
    // @formatter:on
    public static void groupedProjection(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            FloatArray out,
            int rows,
            int rank,
            int groupIn) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (row < rows) {
            int blocks = groupIn / QK;
            int base = (row / rank) * blocks;
            float partial = 0.0f;
            for (int b = lane; b < blocks; b += 32) {
                int offset = (row * blocks + b) * BLOCK_BYTES;
                partial +=
                        w.getHalfFloat(offset).getFloat32()
                                * xScales.get(base + b)
                                * blockDot(w, offset, xQuants, (base + b) * (QK / 4));
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                out.set(row, sum);
            }
        }
    }

    private static int blockDot(ByteArray w, int offset, IntArray xQuants, int quantBase) {
        int dot = 0;
        for (int g = 0; g < QK / 4; g++) {
            int o = offset + 2 + g * 4;
            int packed =
                    (w.getHalfFloat(o).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(o + 2).getHalfFloatValue() & 0xFFFF) << 16);
            dot = QuantizationUtils.dp4a_packed(packed, xQuants.get(quantBase + g), dot);
        }
        return dot;
    }

    // ── Compressor ────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Feeds the position to a compressor of ratio {@code r} and, when it completes a block, pools
     * the block into the cache at {@code position / r}.
     *
     * <ul>
     *   <li>the position's projection and gate score plus {@code ape[position % r]} are stored as a
     *       pending entry: for the overlapping ratio 4 ({@code overlap} 2) entry {@code r + slot}
     *       of rows {@code 2 * width} wide, otherwise entry {@code slot};
     *   <li>a block's key is, per channel, the softmax of the scores over its entries weighing the
     *       projections, RMS-normalized with {@code norm} and rotated at the block's first
     *       position. An overlapping block's softmax runs over the previous block's first halves
     *       (none for block 0) and its own second halves, and its entries then become the previous
     *       block's.
     * </ul>
     *
     * <p>{@code shape} holds the widths and offsets at the {@code C_*} indices. Grid: one workgroup
     * of {@code localSize ≤ GROUP} lanes; {@code width ≤ MAX_WIDTH}.
     */
    // @formatter:on
    public static void compressor(
            KernelContext context,
            FloatArray kv,
            FloatArray gate,
            FloatArray ape,
            FloatArray norm,
            FloatArray pendingKv,
            FloatArray pendingScore,
            FloatArray cache,
            FloatArray freqs,
            IntArray positionHolder,
            IntArray shape,
            float eps) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] pooled = context.allocateFloatLocalArray(MAX_WIDTH);
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int position = positionHolder.get(0);
        int r = shape.get(C_RATIO);
        int o = shape.get(C_OVERLAP);
        int width = shape.get(C_WIDTH);
        int ropeDim = shape.get(C_ROPE_DIM);
        int pendingOffset = shape.get(C_PENDING_OFFSET);
        int cacheOffset = shape.get(C_CACHE_OFFSET);
        int row = o * width;
        int slot = position % r;
        int entry = pendingOffset + ((o - 1) * r + slot) * row;
        for (int i = tid; i < row; i += local) {
            pendingKv.set(entry + i, kv.get(i));
            pendingScore.set(entry + i, gate.get(i) + ape.get(slot * row + i));
        }
        context.localBarrier();

        int complete = (position + 1) % r == 0 ? 1 : 0;
        int block = position / r;
        int overlapping = o - 1;
        int firstValid = overlapping * (block == 0 ? r : 0);
        int count = o * r;
        for (int c = tid; c < width * complete; c += local) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = firstValid; j < count; j++) {
                int half = overlapping * (j >= r ? 1 : 0);
                max =
                        TornadoMath.max(
                                max, pendingScore.get(pendingOffset + j * row + half * width + c));
            }
            float sum = 0.0f;
            float acc = 0.0f;
            for (int j = firstValid; j < count; j++) {
                int half = overlapping * (j >= r ? 1 : 0);
                int at = pendingOffset + j * row + half * width + c;
                float e = TornadoMath.exp(pendingScore.get(at) - max);
                sum += e;
                acc += e * pendingKv.get(at);
            }
            pooled[c] = acc / sum;
        }
        context.localBarrier();
        float partial = 0.0f;
        for (int c = tid; c < width * complete; c += local) {
            partial += pooled[c] * pooled[c];
        }
        float squares = sumReduce(context, reduce, partial);
        float inv = 1.0f / TornadoMath.sqrt(squares / width + eps);
        for (int c = tid; c < width * complete; c += local) {
            pooled[c] = norm.get(c) * (pooled[c] * inv);
        }
        context.localBarrier();
        int nope = width - ropeDim;
        int start = position + 1 - r;
        int key = cacheOffset + block * width;
        for (int p = tid; p < (width / 2) * complete; p += local) {
            int i0 = 2 * p;
            float a = pooled[i0];
            float b = pooled[i0 + 1];
            float tail = i0 >= nope ? 1.0f : 0.0f;
            int f = TornadoMath.max(0, (i0 - nope) / 2);
            float angle = start * freqs.get(TornadoMath.min(f, ropeDim / 2 - 1));
            float cos = TornadoMath.cos(angle);
            float sin = TornadoMath.sin(angle);
            cache.set(key + i0, tail * (a * cos - b * sin) + (1.0f - tail) * a);
            cache.set(key + i0 + 1, tail * (a * sin + b * cos) + (1.0f - tail) * b);
        }
        context.localBarrier();
        int shift = r * row * overlapping * complete;
        for (int i = tid; i < shift; i += local) {
            pendingKv.set(pendingOffset + i, pendingKv.get(pendingOffset + r * row + i));
            pendingScore.set(pendingOffset + i, pendingScore.get(pendingOffset + r * row + i));
        }
    }

    // ── Indexer ───────────────────────────────────────────────────────────────

    /**
     * Rotates the tail of each of {@code heads} vectors of {@code headDim} values at the position.
     * A lane per pair; worker: {@code heads * ropeDim / 2} lanes.
     */
    public static void rotateHeads(
            KernelContext context,
            FloatArray x,
            FloatArray freqs,
            IntArray positionHolder,
            int heads,
            int headDim,
            int ropeDim) {
        int id = context.globalIdx;
        int pairs = ropeDim / 2;
        if (id < heads * pairs) {
            int h = id / pairs;
            int p = id - h * pairs;
            int a = h * headDim + headDim - ropeDim + 2 * p;
            float angle = positionHolder.get(0) * freqs.get(p);
            float cos = TornadoMath.cos(angle);
            float sin = TornadoMath.sin(angle);
            float x0 = x.get(a);
            float x1 = x.get(a + 1);
            x.set(a, x0 * cos - x1 * sin);
            x.set(a + 1, x0 * sin + x1 * cos);
        }
    }

    // @formatter:off
    /**
     * The indexer's score of every complete block {@code b}: {@code sum_h w_h * scale * relu(q_h .
     * key_b)}. A warp per block, its lanes over the heads. Worker: {@code maxBlocks * 32} lanes.
     */
    // @formatter:on
    public static void indexerScores(
            KernelContext context,
            FloatArray query,
            FloatArray headWeights,
            FloatArray keys,
            FloatArray scores,
            IntArray positionHolder,
            int heads,
            int headDim,
            int ratio,
            int keysOffset,
            float scale) {
        int b = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int complete = (positionHolder.get(0) + 1) / ratio;
        if (b < complete) {
            float partial = 0.0f;
            for (int h = lane; h < heads; h += 32) {
                float dot = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    dot += query.get(h * headDim + i) * keys.get(keysOffset + b * headDim + i);
                }
                partial += headWeights.get(h) * scale * TornadoMath.max(dot, 0.0f);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                scores.set(b, sum);
            }
        }
    }

    // @formatter:off
    /**
     * The blocks a query attends to: every complete block while there are at most {@code topK},
     * otherwise the {@code topK} highest-scoring ones, a tie going to the lower block. Writes the
     * count to {@code count[0]}. Consumes {@code scores}.
     *
     * <p>Grid: one workgroup of {@code localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void indexerSelect(
            KernelContext context,
            FloatArray scores,
            IntArray selected,
            IntArray count,
            IntArray positionHolder,
            int ratio,
            int topK) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        int complete = (positionHolder.get(0) + 1) / ratio;
        int all = complete <= topK ? 1 : 0;
        for (int i = tid; i < complete * all; i += local) {
            selected.set(i, i);
        }
        int rounds = topK * (1 - all);
        for (int k = 0; k < rounds; k++) {
            float b = Float.NEGATIVE_INFINITY;
            int index = complete;
            for (int e = tid; e < complete; e += local) {
                float s = scores.get(e);
                int better = s > b ? 1 : 0;
                b = better == 1 ? s : b;
                index = better == 1 ? e : index;
            }
            best[tid] = b;
            bestIndex[tid] = index;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float other = best[tid + stride];
                    int otherIndex = bestIndex[tid + stride];
                    int take =
                            other > best[tid] || (other == best[tid] && otherIndex < bestIndex[tid])
                                    ? 1
                                    : 0;
                    best[tid] = take == 1 ? other : best[tid];
                    bestIndex[tid] = take == 1 ? otherIndex : bestIndex[tid];
                }
                context.localBarrier();
            }
            if (tid == 0) {
                int chosen = TornadoMath.min(bestIndex[0], complete - 1);
                selected.set(k, chosen);
                scores.set(chosen, Float.NEGATIVE_INFINITY);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            count.set(0, TornadoMath.min(complete, topK));
        }
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    private static float softplus(float x) {
        float clipped = TornadoMath.min(TornadoMath.max(x, -15.0f), 20.0f);
        float mid = TornadoMath.log(1.0f + TornadoMath.exp(clipped));
        float low = x < -15.0f ? 1.0f : 0.0f;
        float high = x > 20.0f ? 1.0f : 0.0f;
        return low * TornadoMath.exp(TornadoMath.max(x, -80.0f))
                + high * x
                + (1.0f - low - high) * mid;
    }

    // @formatter:off
    /**
     * Square-root-softplus routing in one workgroup: each expert's score is {@code
     * sqrt(softplus(logit))}; a hash-routed layer ({@code hashed} 1) takes its experts from {@code
     * hashExperts[hashOffset ...]}, any other the top {@code used} of score plus bias (a tie goes
     * to the lower expert). The chosen experts are weighted by their scores, normalized to sum to
     * one when {@code normalize} is 1, and scaled.
     *
     * <p>Grid: one workgroup of {@code localSize ≤ GROUP} lanes; at most {@link #MAX_EXPERTS}.
     */
    // @formatter:on
    public static void routerSqrtSoftplus(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray hashExperts,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used,
            int hashed,
            int hashOffset,
            int normalize,
            float scale) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] score = context.allocateFloatLocalArray(MAX_EXPERTS);
        float[] select = context.allocateFloatLocalArray(MAX_EXPERTS);
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        int biased = 1 - hashed;
        for (int e = tid; e < experts; e += local) {
            float s = TornadoMath.sqrt(softplus(logits.get(e)));
            score[e] = s;
            select[e] = s + biased * bias.get(TornadoMath.min(e, biased * (experts - 1)));
        }
        context.localBarrier();
        for (int k = tid; k < used * hashed; k += local) {
            ids.set(k, hashExperts.get(hashOffset + k));
        }
        for (int k = 0; k < used * biased; k++) {
            float b = Float.NEGATIVE_INFINITY;
            int index = experts;
            for (int e = tid; e < experts; e += local) {
                float s = select[e];
                int better = s > b ? 1 : 0;
                b = better == 1 ? s : b;
                index = better == 1 ? e : index;
            }
            best[tid] = b;
            bestIndex[tid] = index;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float other = best[tid + stride];
                    int otherIndex = bestIndex[tid + stride];
                    int take =
                            other > best[tid] || (other == best[tid] && otherIndex < bestIndex[tid])
                                    ? 1
                                    : 0;
                    best[tid] = take == 1 ? other : best[tid];
                    bestIndex[tid] = take == 1 ? otherIndex : bestIndex[tid];
                }
                context.localBarrier();
            }
            if (tid == 0) {
                int chosen = TornadoMath.min(bestIndex[0], experts - 1);
                ids.set(k, chosen);
                select[chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        context.localBarrier();
        if (tid == 0) {
            float total = 0.0f;
            for (int k = 0; k < used; k++) {
                float w = score[ids.get(k)];
                weights.set(k, w);
                total += w;
            }
            float n = normalize;
            float norm = n * total + (1.0f - n);
            for (int k = 0; k < used; k++) {
                weights.set(k, weights.get(k) / norm * scale);
            }
        }
    }

    // ── Head ──────────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * The learned mHC head's collapse from {@link Glm5NextKernels#hcMixes}' partial sums over its
     * {@code n} mixes: {@code x = sum_m (sigmoid(scale * mix_m + base_m) + eps) * streams[m]}. A
     * lane per hidden element.
     */
    // @formatter:on
    public static void hcHead(
            KernelContext context,
            FloatArray partials,
            FloatArray base,
            FloatArray scale,
            FloatArray streams,
            FloatArray x,
            int n,
            int dim,
            float eps,
            float rmsEps) {
        int i = context.globalIdx;
        if (i < dim) {
            int chunks = Glm5NextKernels.HC_CHUNKS;
            float square = 0.0f;
            for (int j = 0; j < chunks; j++) {
                square += partials.get(n * chunks + j);
            }
            float inv = 1.0f / TornadoMath.sqrt(square / (n * dim) + rmsEps);
            float sum = 0.0f;
            for (int m = 0; m < n; m++) {
                float dot = 0.0f;
                for (int j = 0; j < chunks; j++) {
                    dot += partials.get(m * chunks + j);
                }
                float pre =
                        1.0f / (1.0f + TornadoMath.exp(-(dot * inv * scale.get(0) + base.get(m))))
                                + eps;
                sum += pre * streams.get(m * dim + i);
            }
            x.set(i, sum);
        }
    }
}
