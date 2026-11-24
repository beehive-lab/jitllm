package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Decode kernels for the blocks {@code glm5next} (GLM-5.3-Flash) adds to what deepseek2 and
 * qwen35moe already have: the hyper-connections around every sublayer, KDA's per-channel decay and
 * its delta rule, the k-pool indexer, and latent attention restricted to the indexer's selection.
 *
 * <p>The projections themselves are the existing matrix-vector kernels; the absorbed query and the
 * latent decompression are {@link DeepSeek2Kernels#absorbQuery} and {@link
 * DeepSeek2Kernels#decompressLatent} with a rotated width of zero.
 *
 * <p>Every kernel takes the session's {@code positionHolder} where it needs the position (element
 * 0), so one captured graph serves every token. Workgroup reductions assume a power-of-two local
 * size no larger than {@link #GROUP}.
 */
public final class Glm5NextKernels {

    /** The widest workgroup the reductions here are written for. */
    public static final int GROUP = 256;

    /** The workgroups {@link #hcMixes} splits each mixing coefficient's sum over. */
    public static final int HC_CHUNKS = 16;

    /** The most mixing coefficients {@link #hcCoefficients} takes: {@code (2 + n) * n}. */
    public static final int HC_MAX_MIXES = 128;

    /** The lanes {@link #indexerScores} gives each pool, one per head and then round again. */
    public static final int INDEX_LANES = 32;

    private Glm5NextKernels() {}

    // ── Hyper-connections ─────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Partial sums of the mHC mixing coefficients {@code mixes[k] = fn[k] . s / rms(s)} over the
     * flattened streams {@code s} of {@code width} values: workgroup {@code k * HC_CHUNKS + j}
     * takes chunk {@code j} of the streams and writes {@code fn[k] . s} over it to {@code
     * partials[k * HC_CHUNKS + j]} and {@code s . s} over it to {@code partials[(mixCount + k) *
     * HC_CHUNKS + j]}. {@link #hcCoefficients} adds them up.
     *
     * <p>Grid: {@code mixCount * HC_CHUNKS} workgroups of {@code localSize ≤ GROUP} lanes.
     */
    // @formatter:on
    public static void hcMixes(
            KernelContext context,
            FloatArray streams,
            FloatArray fn,
            FloatArray partials,
            int width,
            int mixCount) {
        int group = context.groupIdx;
        int k = group / HC_CHUNKS;
        int j = group - k * HC_CHUNKS;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] dots = context.allocateFloatLocalArray(GROUP);
        float[] squares = context.allocateFloatLocalArray(GROUP);
        int chunk = (width + HC_CHUNKS - 1) / HC_CHUNKS;
        int start = j * chunk;
        int end = TornadoMath.min(width, start + chunk);
        float dot = 0.0f;
        float square = 0.0f;
        if (k < mixCount) {
            for (int i = start + tid; i < end; i += local) {
                float v = streams.get(i);
                dot += fn.get(k * width + i) * v;
                square += v * v;
            }
        }
        dots[tid] = dot;
        squares[tid] = square;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                dots[tid] += dots[tid + stride];
                squares[tid] += squares[tid + stride];
            }
            context.localBarrier();
        }
        if (k < mixCount && tid == 0) {
            partials.set(group, dots[0]);
            partials.set(mixCount * HC_CHUNKS + group, squares[0]);
        }
    }

    // @formatter:off
    /**
     * The mixes from {@link #hcMixes}' partial sums, and the coefficients from them, in one
     * workgroup: {@code coeffs} is {@code pre[n]}, then {@code post[n]}, then the mix {@code
     * comb[n][n]} with {@code comb[m][k]} feeding input stream {@code m} into output stream {@code
     * k}.
     *
     * <ul>
     *   <li>{@code pre = sigmoid(mix * s0 + b) + eps}, {@code post = postScale * sigmoid(mix * s1 +
     *       b)};
     *   <li>the mix: a row softmax plus eps, a column normalization, then {@code iterations - 1}
     *       row-then-column normalizations (Sinkhorn), a lane per entry.
     * </ul>
     *
     * <p>Grid: one workgroup of at least {@code (2 + n) * n} lanes, at most {@link #HC_MAX_MIXES}.
     */
    // @formatter:on
    public static void hcCoefficients(
            KernelContext context,
            FloatArray partials,
            FloatArray base,
            FloatArray scale,
            FloatArray mixes,
            FloatArray coeffs,
            int n,
            int width,
            int iterations,
            float eps,
            float postScale,
            float rmsEps) {
        int id = context.localIdx;
        int mixCount = (2 + n) * n;
        float[] mix = context.allocateFloatLocalArray(HC_MAX_MIXES);
        float[] comb = context.allocateFloatLocalArray(HC_MAX_MIXES);
        if (id < mixCount) {
            float dot = 0.0f;
            float square = 0.0f;
            for (int j = 0; j < HC_CHUNKS; j++) {
                dot += partials.get(id * HC_CHUNKS + j);
                square += partials.get((mixCount + id) * HC_CHUNKS + j);
            }
            float v = dot / TornadoMath.sqrt(square / width + rmsEps);
            mix[id] = v;
            mixes.set(id, v);
        }
        context.localBarrier();
        if (id < n) {
            float pre = mix[id] * scale.get(0) + base.get(id);
            coeffs.set(id, 1.0f / (1.0f + TornadoMath.exp(-pre)) + eps);
            float post = mix[n + id] * scale.get(1) + base.get(n + id);
            coeffs.set(n + id, postScale / (1.0f + TornadoMath.exp(-post)));
        }
        int entries = n * n;
        int row = id / n;
        int c0 = 2 * n;
        if (id < entries) {
            comb[id] = mix[c0 + id] * scale.get(2) + base.get(c0 + id);
        }
        context.localBarrier();
        // The row softmax plus eps.
        float max = Float.NEGATIVE_INFINITY;
        if (id < entries) {
            for (int c = 0; c < n; c++) {
                max = TornadoMath.max(max, comb[row * n + c]);
            }
        }
        context.localBarrier();
        if (id < entries) {
            comb[id] = TornadoMath.exp(comb[id] - max);
        }
        context.localBarrier();
        float sum = 0.0f;
        if (id < entries) {
            for (int c = 0; c < n; c++) {
                sum += comb[row * n + c];
            }
        }
        context.localBarrier();
        if (id < entries) {
            comb[id] = comb[id] / sum + eps;
        }
        context.localBarrier();
        int column = id - row * n;
        for (int it = 0; it < iterations; it++) {
            // Rows, from the second iteration on.
            int rows = TornadoMath.min(it, 1);
            float rowSum = 0.0f;
            if (id < entries) {
                for (int c = 0; c < n; c++) {
                    rowSum += comb[row * n + c];
                }
            }
            context.localBarrier();
            if (id < entries) {
                float scaled = comb[id] / (rowSum + eps);
                comb[id] = rows * scaled + (1 - rows) * comb[id];
            }
            context.localBarrier();
            float columnSum = 0.0f;
            if (id < entries) {
                for (int r = 0; r < n; r++) {
                    columnSum += comb[r * n + column];
                }
            }
            context.localBarrier();
            if (id < entries) {
                comb[id] = comb[id] / (columnSum + eps);
            }
            context.localBarrier();
        }
        if (id < entries) {
            coeffs.set(c0 + id, comb[id]);
        }
    }

    /**
     * {@code x = sum_m pre[m] * streams[m]}, saving the streams to {@code residual} for {@link
     * #hcExpand}. A lane per hidden element.
     */
    public static void hcCollapse(
            KernelContext context,
            FloatArray streams,
            FloatArray coeffs,
            FloatArray residual,
            FloatArray x,
            int dim,
            int n) {
        int i = context.globalIdx;
        if (i < dim) {
            float sum = 0.0f;
            for (int m = 0; m < n; m++) {
                float v = streams.get(m * dim + i);
                residual.set(m * dim + i, v);
                sum += coeffs.get(m) * v;
            }
            x.set(i, sum);
        }
    }

    /**
     * {@code streams[k] = post[k] * branch + sum_m comb[m][k] * residual[m]}. A lane per stream
     * element.
     */
    public static void hcExpand(
            KernelContext context,
            FloatArray residual,
            FloatArray branch,
            FloatArray coeffs,
            FloatArray streams,
            int dim,
            int n) {
        int id = context.globalIdx;
        if (id < n * dim) {
            int k = id / dim;
            int i = id - k * dim;
            float v = coeffs.get(n + k) * branch.get(i);
            for (int m = 0; m < n; m++) {
                v += coeffs.get(2 * n + m * n + k) * residual.get(m * dim + i);
            }
            streams.set(id, v);
        }
    }

    /** The embedding into every stream. A lane per stream element. */
    public static void hcBroadcast(
            KernelContext context, FloatArray x, FloatArray streams, int dim, int n) {
        int id = context.globalIdx;
        if (id < n * dim) {
            streams.set(id, x.get(id % dim));
        }
    }

    /** The head's collapse: the plain mean of the streams. A lane per hidden element. */
    public static void hcMean(
            KernelContext context, FloatArray streams, FloatArray x, int dim, int n) {
        int i = context.globalIdx;
        if (i < dim) {
            float sum = 0.0f;
            for (int m = 0; m < n; m++) {
                sum += streams.get(m * dim + i);
            }
            x.set(i, sum / n);
        }
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    // ── Indexer ───────────────────────────────────────────────────────────────

    /**
     * The indexer key {@code LayerNorm(raw)} and the pool gate, stored at this position of the
     * layer's caches; one workgroup.
     */
    public static void indexerStore(
            KernelContext context,
            FloatArray rawKey,
            FloatArray gate,
            FloatArray weight,
            FloatArray bias,
            FloatArray keyCache,
            FloatArray gateCache,
            IntArray positionHolder,
            int headDim,
            int cacheOffset,
            float eps) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] sums = context.allocateFloatLocalArray(GROUP);
        float[] squares = context.allocateFloatLocalArray(GROUP);
        float sum = 0.0f;
        float square = 0.0f;
        for (int i = tid; i < headDim; i += local) {
            float v = rawKey.get(i);
            sum += v;
            square += v * v;
        }
        sums[tid] = sum;
        squares[tid] = square;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sums[tid] += sums[tid + stride];
                squares[tid] += squares[tid + stride];
            }
            context.localBarrier();
        }
        float mean = sums[0] / headDim;
        float variance = squares[0] / headDim - mean * mean;
        float inv = 1.0f / TornadoMath.sqrt(variance + eps);
        int row = cacheOffset + positionHolder.get(0) * headDim;
        for (int i = tid; i < headDim; i += local) {
            keyCache.set(row + i, (rawKey.get(i) - mean) * inv * weight.get(i) + bias.get(i));
            gateCache.set(row + i, gate.get(i));
        }
    }

    // @formatter:off
    /**
     * When this position completes a pool, its compressed key: per channel, a softmax over the pool
     * of {@code gate + ape[slot]} weighting the pool's keys. A lane per channel.
     */
    // @formatter:on
    public static void indexerPoolKey(
            KernelContext context,
            FloatArray keyCache,
            FloatArray gateCache,
            FloatArray ape,
            FloatArray poolKeys,
            IntArray positionHolder,
            int headDim,
            int pool,
            int cacheOffset,
            int poolOffset) {
        int c = context.globalIdx;
        int position = positionHolder.get(0);
        if (c < headDim && (position + 1) % pool == 0) {
            int p = position / pool;
            int first = cacheOffset + p * pool * headDim + c;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < pool; j++) {
                max =
                        TornadoMath.max(
                                max, gateCache.get(first + j * headDim) + ape.get(j * headDim + c));
            }
            float sum = 0.0f;
            float acc = 0.0f;
            for (int j = 0; j < pool; j++) {
                float e =
                        TornadoMath.exp(
                                gateCache.get(first + j * headDim)
                                        + ape.get(j * headDim + c)
                                        - max);
                sum += e;
                acc += e * keyCache.get(first + j * headDim);
            }
            poolKeys.set(poolOffset + p * headDim + c, acc / sum);
        }
    }

    // @formatter:off
    /**
     * Each complete pool's score {@code sum_h w_h / sqrt(H) * relu(q_h . key / sqrt(d))}; {@link
     * #INDEX_LANES} lanes per pool, a head per lane for {@code INDEX_LANES} heads at a time, and
     * the pool's first lane adding the heads' terms in head order. Pools not yet complete score
     * negative infinity.
     *
     * <p>Grid: {@code maxPools * INDEX_LANES} lanes in workgroups of a multiple of {@code
     * INDEX_LANES}, at most {@link #GROUP}.
     */
    // @formatter:on
    public static void indexerScores(
            KernelContext context,
            FloatArray query,
            FloatArray headWeights,
            FloatArray poolKeys,
            FloatArray scores,
            IntArray positionHolder,
            int heads,
            int headDim,
            int pool,
            int maxPools,
            int poolOffset) {
        int id = context.globalIdx;
        int tid = context.localIdx;
        int p = id / INDEX_LANES;
        int lane = id - p * INDEX_LANES;
        int first = tid - lane;
        float[] terms = context.allocateFloatLocalArray(GROUP);
        int complete = (positionHolder.get(0) + 1) / pool;
        float headScale = 1.0f / TornadoMath.sqrt(heads);
        float keyScale = 1.0f / TornadoMath.sqrt(headDim);
        int key = poolOffset + p * headDim;
        float score = 0.0f;
        for (int h0 = 0; h0 < heads; h0 += INDEX_LANES) {
            int h = h0 + lane;
            float term = 0.0f;
            if (p < complete && h < heads) {
                float dot = 0.0f;
                for (int c = 0; c < headDim; c++) {
                    dot += query.get(h * headDim + c) * poolKeys.get(key + c);
                }
                term = headWeights.get(h) * headScale * TornadoMath.max(dot * keyScale, 0.0f);
            }
            terms[tid] = term;
            context.localBarrier();
            if (lane == 0) {
                int count = TornadoMath.min(INDEX_LANES, heads - h0);
                for (int i = 0; i < count; i++) {
                    score += terms[first + i];
                }
            }
            context.localBarrier();
        }
        if (p < maxPools && lane == 0) {
            int done = TornadoMath.min(1, TornadoMath.max(0, complete - p));
            if (done == 1) {
                scores.set(p, score);
            }
            if (done == 0) {
                scores.set(p, Float.NEGATIVE_INFINITY);
            }
        }
    }

    // @formatter:off
    /**
     * Marks the {@code budget} best complete pools in {@code selected} (1 chosen, 0 not), in one
     * workgroup.
     *
     * <p>With no more complete pools than the budget all are chosen. Otherwise a bisection on the
     * score finds {@code hi}, above which fewer than {@code budget} pools lie, and {@code lo}, at
     * or above which at least {@code budget} do; every pool at or above {@code hi} is chosen and
     * the rest are taken from {@code [lo, hi)} in pool order — the tie rule of the host path, which
     * the bisection's last interval holds only ties within float resolution of.
     */
    // @formatter:on
    public static void indexerSelect(
            KernelContext context,
            FloatArray scores,
            IntArray selected,
            IntArray positionHolder,
            int pool,
            int maxPools,
            int budget) {
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduceMin = context.allocateFloatLocalArray(GROUP);
        float[] reduceMax = context.allocateFloatLocalArray(GROUP);
        int[] counts = context.allocateIntLocalArray(GROUP);
        int complete = TornadoMath.min((positionHolder.get(0) + 1) / pool, maxPools);
        for (int p = tid; p < maxPools; p += local) {
            selected.set(p, 0);
        }
        context.localBarrier();
        if (complete <= budget) {
            for (int p = tid; p < complete; p += local) {
                selected.set(p, 1);
            }
        } else {
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (int p = tid; p < complete; p += local) {
                lo = TornadoMath.min(lo, scores.get(p));
                hi = TornadoMath.max(hi, scores.get(p));
            }
            reduceMin[tid] = lo;
            reduceMax[tid] = hi;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduceMin[tid] = TornadoMath.min(reduceMin[tid], reduceMin[tid + stride]);
                    reduceMax[tid] = TornadoMath.max(reduceMax[tid], reduceMax[tid + stride]);
                }
                context.localBarrier();
            }
            // Invariant: count(score >= lo) >= budget and count(score >= hi) < budget.
            float low = reduceMin[0];
            float high = reduceMax[0] + 1.0f + TornadoMath.abs(reduceMax[0]);
            for (int it = 0; it < 64; it++) {
                float mid = 0.5f * (low + high);
                int count = 0;
                for (int p = tid; p < complete; p += local) {
                    count += scores.get(p) >= mid ? 1 : 0;
                }
                counts[tid] = count;
                context.localBarrier();
                for (int stride = local / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        counts[tid] += counts[tid + stride];
                    }
                    context.localBarrier();
                }
                int total = counts[0];
                context.localBarrier();
                float nextLow = total >= budget ? mid : low;
                float nextHigh = total >= budget ? high : mid;
                low = nextLow;
                high = nextHigh;
            }
            int above = 0;
            for (int p = tid; p < complete; p += local) {
                int chosen = scores.get(p) >= high ? 1 : 0;
                selected.set(p, chosen);
                above += chosen;
            }
            counts[tid] = above;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    counts[tid] += counts[tid + stride];
                }
                context.localBarrier();
            }
            if (tid == 0) {
                int need = budget - counts[0];
                for (int p = 0; p < complete && need > 0; p++) {
                    float s = scores.get(p);
                    if (s >= low && s < high) {
                        selected.set(p, 1);
                        need--;
                    }
                }
            }
        }
    }

    // ── Sparse latent attention ───────────────────────────────────────────────

    // @formatter:off
    /**
     * Absorbed attention over the selected positions, a workgroup per head: the chosen pools, and
     * when {@code selectTail} is 1 the incomplete pool holding this position.
     *
     * <p>{@code absorbed} is every head's query in latent space ({@code heads × rank}); the cache
     * holds one normalized latent of {@code rank} per position from {@code cacheOffset}. The scores
     * of head {@code h} use {@code scratch[h * context ...]}. The attended latent goes to {@code
     * latentOut}.
     */
    // @formatter:on
    public static void sparseLatentAttention(
            KernelContext context,
            FloatArray absorbed,
            FloatArray latents,
            IntArray selected,
            FloatArray scratch,
            FloatArray latentOut,
            IntArray positionHolder,
            int heads,
            int rank,
            int pool,
            int contextLength,
            int cacheOffset,
            int selectTail,
            float scale) {
        int h = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        int position = positionHolder.get(0);
        int tailStart = (position + 1) / pool * pool;
        int scores = h * contextLength;
        float max = Float.NEGATIVE_INFINITY;
        if (h < heads) {
            for (int s = tid; s <= position; s += local) {
                // No merge of branch assignments: the CUDA backend miscompiles them.
                int tail = selectTail == 1 && s >= tailStart ? 1 : 0;
                int visible = TornadoMath.max(selected.get(s / pool), tail);
                if (visible == 1) {
                    float dot = 0.0f;
                    for (int r = 0; r < rank; r++) {
                        dot += absorbed.get(h * rank + r) * latents.get(cacheOffset + s * rank + r);
                    }
                    scratch.set(scores + s, dot * scale);
                } else {
                    scratch.set(scores + s, Float.NEGATIVE_INFINITY);
                }
                max = TornadoMath.max(max, scratch.get(scores + s));
            }
        }
        reduce[tid] = max;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
            }
            context.localBarrier();
        }
        float rowMax = reduce[0];
        context.localBarrier();
        float sum = 0.0f;
        if (h < heads) {
            for (int s = tid; s <= position; s += local) {
                float e = TornadoMath.exp(scratch.get(scores + s) - rowMax);
                scratch.set(scores + s, e);
                sum += e;
            }
        }
        reduce[tid] = sum;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float total = reduce[0];
        if (h < heads) {
            for (int r = tid; r < rank; r += local) {
                float acc = 0.0f;
                for (int s = 0; s <= position; s++) {
                    acc += scratch.get(scores + s) * latents.get(cacheOffset + s * rank + r);
                }
                latentOut.set(h * rank + r, acc / total);
            }
        }
    }

    // ── Feed-forward ──────────────────────────────────────────────────────────

    /** {@code gate = silu(min(gate, limit)) * clamp(up, -limit, limit)}, a lane per element. */
    public static void clampedSwiGLU(
            KernelContext context, FloatArray gate, FloatArray up, int size, float limit) {
        int i = context.globalIdx;
        if (i < size) {
            float g = TornadoMath.min(gate.get(i), limit);
            float u = TornadoMath.max(-limit, TornadoMath.min(up.get(i), limit));
            gate.set(i, g / (1.0f + TornadoMath.exp(-g)) * u);
        }
    }

    // ── Experts ───────────────────────────────────────────────────────────────

    /**
     * {@code out = w . x} over an F32 weight, a warp per output row: the projections llama.cpp
     * keeps in F32, such as the indexer's k-pool gate. Worker: {@code d * 32} lanes.
     */
    public static void projectF32(
            KernelContext context, FloatArray x, FloatArray w, FloatArray out, int n, int d) {
        int row = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (row < d) {
            float partial = 0.0f;
            for (int i = lane; i < n; i += 32) {
                partial += w.get(row * n + i) * x.get(i);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                out.set(row, sum);
            }
        }
    }

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }
}
