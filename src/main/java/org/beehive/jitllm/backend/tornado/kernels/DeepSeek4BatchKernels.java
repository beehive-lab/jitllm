package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

// @formatter:off
/**
 * Batched-prefill kernels for the blocks {@link DeepSeek4Kernels} decodes one token at a time: row
 * {@code b} of a chunk holds the token at position {@code start + b}.
 *
 * <p>Every kernel takes the chunk's {@code batchInfo}: the start position at {@code [0]} and the
 * number of real rows at {@code [1]}. The grids cover the plan's whole chunk width; a padding row
 * computes as the last real row does and writes only to its own row of a scratch array, never to a
 * cache.
 *
 * <p>The window keys of a chunk come from two places: the positions before the chunk from the
 * layer's ring, the chunk's own from its rotated key/value rows. The ring is written after the
 * attention ({@link #storeWindow}), so a chunk wider than the window does not overwrite a key a row
 * still reads. The compressor walks the chunk's rows in order in one workgroup.
 */
// @formatter:on
public final class DeepSeek4BatchKernels {

    public static final int GROUP = DeepSeek4Kernels.GROUP;

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    private DeepSeek4BatchKernels() {}

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

    /** Row {@code b}, or the last real row for a padding row. */
    private static int realRow(IntArray batchInfo, int b) {
        return TornadoMath.min(b, batchInfo.get(1) - 1);
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Every row's query heads RMS-normalized without a weight, then their tails rotated at the
     * row's position. Grid: {@code batch * heads} workgroups of {@code localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void queryPrep(
            KernelContext context,
            FloatArray q,
            FloatArray freqs,
            IntArray batchInfo,
            int heads,
            int headDim,
            int ropeDim,
            float eps) {
        int g = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int b = g / heads;
        int position = batchInfo.get(0) + realRow(batchInfo, b);
        int nope = headDim - ropeDim;
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
        for (int p = tid; p < ropeDim / 2; p += local) {
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

    /**
     * Rotates the tail of each of {@code heads} vectors of {@code headDim} values in every row at
     * the row's position: the key/value ({@code heads} 1) and the indexer's query. A lane per pair;
     * worker: {@code batch * heads * ropeDim / 2} lanes.
     */
    public static void rotateHeads(
            KernelContext context,
            FloatArray x,
            FloatArray freqs,
            IntArray batchInfo,
            int heads,
            int headDim,
            int ropeDim,
            int batch) {
        int id = context.globalIdx;
        int pairs = ropeDim / 2;
        if (id < batch * heads * pairs) {
            int v = id / pairs;
            int p = id - v * pairs;
            int b = v / heads;
            int position = batchInfo.get(0) + realRow(batchInfo, b);
            int a = v * headDim + headDim - ropeDim + 2 * p;
            float angle = position * freqs.get(p);
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
     * {@link DeepSeek4Kernels#attention} for every (row, head): the window's positions before the
     * chunk from the ring, its positions in the chunk from the rotated key/value rows {@code keys},
     * the compressed blocks complete at the row's position, or the indexer's choice for the row
     * ({@code selected[b * capacity ...]}, {@code count[b]}).
     *
     * <p>Grid: {@code batch * heads} workgroups of {@code localSize ≤ GROUP} lanes; workgroup
     * {@code g}'s scores use {@code scratch[g * stride ...]}.
     */
    // @formatter:on
    public static void attention(
            KernelContext context,
            FloatArray q,
            FloatArray keys,
            FloatArray window,
            FloatArray compressed,
            IntArray selected,
            IntArray selectedCount,
            FloatArray sinks,
            FloatArray scratch,
            FloatArray out,
            FloatArray freqs,
            IntArray batchInfo,
            IntArray shape,
            float scale) {
        int g = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int heads = shape.get(DeepSeek4Kernels.SHAPE_HEADS);
        int headDim = shape.get(DeepSeek4Kernels.SHAPE_HEAD_DIM);
        int ropeDim = shape.get(DeepSeek4Kernels.SHAPE_ROPE_DIM);
        int windowSize = shape.get(DeepSeek4Kernels.SHAPE_WINDOW);
        int windowOffset = shape.get(DeepSeek4Kernels.SHAPE_WINDOW_OFFSET);
        int compressedOffset = shape.get(DeepSeek4Kernels.SHAPE_COMPRESSED_OFFSET);
        int stride = shape.get(DeepSeek4Kernels.SHAPE_STRIDE);
        int ratio = shape.get(DeepSeek4Kernels.SHAPE_RATIO);
        int useSelection = shape.get(DeepSeek4Kernels.SHAPE_USE_SELECTION);
        int capacity = shape.get(DeepSeek4Kernels.SHAPE_SELECTION_CAPACITY);
        int b = g / heads;
        int h = g - b * heads;
        int row = realRow(batchInfo, b);
        int start = batchInfo.get(0);
        int position = start + row;

        int windowCount = TornadoMath.min(position + 1, windowSize);
        int first = position - windowCount + 1;
        int ringCount = TornadoMath.max(0, windowCount - 1 - row);
        int chunkCount = windowCount - ringCount;
        int chunkFirst = row - chunkCount + 1;
        int compressedLayer = TornadoMath.min(ratio, 1);
        int complete = (position + 1) / TornadoMath.max(ratio, 1);
        int blockCount =
                compressedLayer
                        * (useSelection * selectedCount.get(row) + (1 - useSelection) * complete);
        int selBase = row * capacity;
        int scores = g * stride;
        int qBase = (b * heads + h) * headDim;

        float max = Float.NEGATIVE_INFINITY;
        for (int j = tid; j < ringCount; j += local) {
            int key = windowOffset + ((first + j) % windowSize) * headDim;
            float dot = 0.0f;
            for (int i = 0; i < headDim; i++) {
                dot += q.get(qBase + i) * window.get(key + i);
            }
            scratch.set(scores + j, dot * scale);
            max = TornadoMath.max(max, dot * scale);
        }
        for (int j = tid; j < chunkCount; j += local) {
            int key = (chunkFirst + j) * headDim;
            float dot = 0.0f;
            for (int i = 0; i < headDim; i++) {
                dot += q.get(qBase + i) * keys.get(key + i);
            }
            scratch.set(scores + ringCount + j, dot * scale);
            max = TornadoMath.max(max, dot * scale);
        }
        for (int j = tid; j < blockCount; j += local) {
            int pick = selected.get(selBase + TornadoMath.min(j, capacity - 1));
            int block = useSelection * pick + (1 - useSelection) * j;
            int key = compressedOffset + block * headDim;
            float dot = 0.0f;
            for (int i = 0; i < headDim; i++) {
                dot += q.get(qBase + i) * compressed.get(key + i);
            }
            scratch.set(scores + windowCount + j, dot * scale);
            max = TornadoMath.max(max, dot * scale);
        }
        float sink = sinks.get(h);
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
            for (int j = 0; j < ringCount; j++) {
                int key = windowOffset + ((first + j) % windowSize) * headDim;
                float w = scratch.get(scores + j);
                a0 += w * window.get(key + i0);
                a1 += w * window.get(key + i0 + 1);
            }
            for (int j = 0; j < chunkCount; j++) {
                int key = (chunkFirst + j) * headDim;
                float w = scratch.get(scores + ringCount + j);
                a0 += w * keys.get(key + i0);
                a1 += w * keys.get(key + i0 + 1);
            }
            for (int j = 0; j < blockCount; j++) {
                int pick = selected.get(selBase + TornadoMath.min(j, capacity - 1));
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
            out.set(qBase + i0, tail * r0 + (1.0f - tail) * a0);
            out.set(qBase + i0 + 1, tail * r1 + (1.0f - tail) * a1);
        }
    }

    // @formatter:off
    /**
     * The chunk's last {@code window} rotated key/value rows into the layer's ring at slot {@code
     * position % window}, after the chunk's attention has read the ring. A lane per value; worker:
     * {@code batch * headDim} lanes.
     */
    // @formatter:on
    public static void storeWindow(
            KernelContext context,
            FloatArray keys,
            FloatArray window,
            IntArray batchInfo,
            int headDim,
            int windowSize,
            int windowOffset,
            int batch) {
        int id = context.globalIdx;
        int b = id / headDim;
        int count = batchInfo.get(1);
        if (id < batch * headDim && b < count && b >= count - windowSize) {
            int i = id - b * headDim;
            int slot = (batchInfo.get(0) + b) % windowSize;
            window.set(windowOffset + slot * headDim + i, keys.get(id));
        }
    }

    // @formatter:off
    /**
     * {@code out_b[r] = W[r] . x_b[(r / rank) * groupIn ...]}: {@link
     * DeepSeek4Kernels#groupedProjection} for every row, the activation quantized per block, rows
     * of {@code (rows / rank) * groupIn} values. A warp per (row, output); worker: {@code batch *
     * rows * 32} lanes.
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
            int groupIn,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < batch * rows) {
            int b = warp / rows;
            int row = warp - b * rows;
            int blocks = groupIn / QK;
            int base = b * (rows / rank) * blocks + (row / rank) * blocks;
            float partial = 0.0f;
            for (int k = lane; k < blocks; k += 32) {
                int offset = (row * blocks + k) * BLOCK_BYTES;
                partial +=
                        w.getHalfFloat(offset).getFloat32()
                                * xScales.get(base + k)
                                * blockDot(w, offset, xQuants, (base + k) * (QK / 4));
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                out.set(warp, sum);
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
     * {@link DeepSeek4Kernels#compressor} over the chunk's real rows in order: each row's
     * projection and gate score ({@code kv[b * row ...]}, {@code gate[b * row ...]}, {@code row =
     * overlap * width}) become a pending entry, and a row that completes a block pools it into the
     * cache. The rows run one after another in one workgroup, as the pending entries of a row
     * depend on the rows before it.
     *
     * <p>Grid: one workgroup of {@code localSize ≤ GROUP} lanes; {@code width ≤ MAX_WIDTH}.
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
            IntArray batchInfo,
            IntArray shape,
            float eps) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] pooled = context.allocateFloatLocalArray(DeepSeek4Kernels.MAX_WIDTH);
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int start = batchInfo.get(0);
        int count = batchInfo.get(1);
        int r = shape.get(DeepSeek4Kernels.C_RATIO);
        int o = shape.get(DeepSeek4Kernels.C_OVERLAP);
        int width = shape.get(DeepSeek4Kernels.C_WIDTH);
        int ropeDim = shape.get(DeepSeek4Kernels.C_ROPE_DIM);
        int pendingOffset = shape.get(DeepSeek4Kernels.C_PENDING_OFFSET);
        int cacheOffset = shape.get(DeepSeek4Kernels.C_CACHE_OFFSET);
        int row = o * width;
        int overlapping = o - 1;
        int entries = o * r;
        int nope = width - ropeDim;
        for (int b = 0; b < count; b++) {
            int position = start + b;
            int slot = position % r;
            int entry = pendingOffset + (overlapping * r + slot) * row;
            for (int i = tid; i < row; i += local) {
                pendingKv.set(entry + i, kv.get(b * row + i));
                pendingScore.set(entry + i, gate.get(b * row + i) + ape.get(slot * row + i));
            }
            context.localBarrier();

            int complete = (position + 1) % r == 0 ? 1 : 0;
            int block = position / r;
            int firstValid = overlapping * (block == 0 ? r : 0);
            for (int c = tid; c < width * complete; c += local) {
                float max = Float.NEGATIVE_INFINITY;
                for (int j = firstValid; j < entries; j++) {
                    int half = overlapping * (j >= r ? 1 : 0);
                    max =
                            TornadoMath.max(
                                    max,
                                    pendingScore.get(pendingOffset + j * row + half * width + c));
                }
                float sum = 0.0f;
                float acc = 0.0f;
                for (int j = firstValid; j < entries; j++) {
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
            int first = position + 1 - r;
            int key = cacheOffset + block * width;
            for (int p = tid; p < (width / 2) * complete; p += local) {
                int i0 = 2 * p;
                float a = pooled[i0];
                float c1 = pooled[i0 + 1];
                float tail = i0 >= nope ? 1.0f : 0.0f;
                int f = TornadoMath.max(0, (i0 - nope) / 2);
                float angle = first * freqs.get(TornadoMath.min(f, ropeDim / 2 - 1));
                float cos = TornadoMath.cos(angle);
                float sin = TornadoMath.sin(angle);
                cache.set(key + i0, tail * (a * cos - c1 * sin) + (1.0f - tail) * a);
                cache.set(key + i0 + 1, tail * (a * sin + c1 * cos) + (1.0f - tail) * c1);
            }
            context.localBarrier();
            int shift = r * row * overlapping * complete;
            for (int i = tid; i < shift; i += local) {
                pendingKv.set(pendingOffset + i, pendingKv.get(pendingOffset + r * row + i));
                pendingScore.set(pendingOffset + i, pendingScore.get(pendingOffset + r * row + i));
            }
            context.localBarrier();
        }
    }

    // ── Indexer ───────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * {@link DeepSeek4Kernels#indexerScores} for every row, over the blocks complete at the row's
     * position, into {@code scores[b * maxBlocks ...]}. A warp per (row, block); worker: {@code
     * batch * maxBlocks * 32} lanes.
     */
    // @formatter:on
    public static void indexerScores(
            KernelContext context,
            FloatArray query,
            FloatArray headWeights,
            FloatArray keys,
            FloatArray scores,
            IntArray batchInfo,
            int heads,
            int headDim,
            int ratio,
            int keysOffset,
            int maxBlocks,
            float scale,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int b = warp / maxBlocks;
        int k = warp - b * maxBlocks;
        int row = realRow(batchInfo, b);
        int complete = (batchInfo.get(0) + row + 1) / ratio;
        if (b < batch && k < complete) {
            int q = b * heads * headDim;
            float partial = 0.0f;
            for (int h = lane; h < heads; h += 32) {
                float dot = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    dot += query.get(q + h * headDim + i) * keys.get(keysOffset + k * headDim + i);
                }
                partial += headWeights.get(b * heads + h) * scale * TornadoMath.max(dot, 0.0f);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                scores.set(warp, sum);
            }
        }
    }

    // @formatter:off
    /**
     * {@link DeepSeek4Kernels#indexerSelect} for every row: the blocks row {@code b} attends to
     * into {@code selected[b * capacity ...]}, their count into {@code count[b]}. Consumes {@code
     * scores}. Grid: {@code batch} workgroups of {@code localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void indexerSelect(
            KernelContext context,
            FloatArray scores,
            IntArray selected,
            IntArray count,
            IntArray batchInfo,
            int ratio,
            int topK,
            int maxBlocks,
            int capacity) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        int row = realRow(batchInfo, b);
        int complete = (batchInfo.get(0) + row + 1) / ratio;
        int base = b * maxBlocks;
        int out = b * capacity;
        int all = complete <= topK ? 1 : 0;
        for (int i = tid; i < complete * all; i += local) {
            selected.set(out + i, i);
        }
        int rounds = topK * (1 - all);
        for (int k = 0; k < rounds; k++) {
            float best0 = Float.NEGATIVE_INFINITY;
            int index = complete;
            for (int e = tid; e < complete; e += local) {
                float s = scores.get(base + e);
                int better = s > best0 ? 1 : 0;
                best0 = better == 1 ? s : best0;
                index = better == 1 ? e : index;
            }
            best[tid] = best0;
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
                selected.set(out + k, chosen);
                scores.set(base + chosen, Float.NEGATIVE_INFINITY);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            count.set(b, TornadoMath.min(complete, topK));
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
     * {@link DeepSeek4Kernels#routerSqrtSoftplus} for every row: logits at {@code logits[b *
     * experts ...]}, the chosen experts and weights at {@code [b * used ...]}, a hash-routed
     * layer's experts from {@code hashExperts[b * hashStride + hashOffset ...]}.
     *
     * <p>Grid: {@code batch} workgroups of {@code localSize ≤ GROUP} lanes; at most {@link
     * DeepSeek4Kernels#MAX_EXPERTS} experts.
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
            int hashStride,
            int normalize,
            float scale) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] score = context.allocateFloatLocalArray(DeepSeek4Kernels.MAX_EXPERTS);
        float[] select = context.allocateFloatLocalArray(DeepSeek4Kernels.MAX_EXPERTS);
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        int biased = 1 - hashed;
        int logitBase = b * experts;
        int idBase = b * used;
        for (int e = tid; e < experts; e += local) {
            float s = TornadoMath.sqrt(softplus(logits.get(logitBase + e)));
            score[e] = s;
            select[e] = s + biased * bias.get(TornadoMath.min(e, biased * (experts - 1)));
        }
        context.localBarrier();
        for (int k = tid; k < used * hashed; k += local) {
            ids.set(idBase + k, hashExperts.get(b * hashStride + hashOffset + k));
        }
        for (int k = 0; k < used * biased; k++) {
            float best0 = Float.NEGATIVE_INFINITY;
            int index = experts;
            for (int e = tid; e < experts; e += local) {
                float s = select[e];
                int better = s > best0 ? 1 : 0;
                best0 = better == 1 ? s : best0;
                index = better == 1 ? e : index;
            }
            best[tid] = best0;
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
                ids.set(idBase + k, chosen);
                select[chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        context.localBarrier();
        if (tid == 0) {
            float total = 0.0f;
            for (int k = 0; k < used; k++) {
                float w = score[ids.get(idBase + k)];
                weights.set(idBase + k, w);
                total += w;
            }
            float n = normalize;
            float norm = n * total + (1.0f - n);
            for (int k = 0; k < used; k++) {
                weights.set(idBase + k, weights.get(idBase + k) / norm * scale);
            }
        }
    }
}
