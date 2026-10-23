package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Batched-prefill kernels for {@code glm5next} (GLM-5.3-Flash): the {@link Glm5NextKernels} decode
 * steps over a chunk of tokens, row {@code b} of each batch buffer holding token {@code b}.
 *
 * <p>{@code batchInfo} is the session's batch holder: element 0 the chunk's first position, element
 * 1 how many of its rows are real. Rows past that are padding, which the token-parallel kernels may
 * compute and the stateful ones — the convolution, the delta rule, every cache write — skip.
 *
 * <p>The KDA convolution and delta rule walk the chunk's tokens in order inside one launch: a lane
 * owns a channel (or a state column) and carries it from token to token. Sparse attention is an
 * online softmax over tiles of positions, so its scores need no buffer the size of the context.
 */
public final class Glm5NextBatchKernels {

    /** The widest workgroup the reductions here are written for. */
    public static final int GROUP = 256;

    /** The widest latent {@link #sparseLatentAttention} holds per workgroup. */
    public static final int MAX_RANK = 1024;

    /** The most experts {@link #routerTopKSigmoid} routes over. */
    public static final int MAX_EXPERTS = 512;

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    private Glm5NextBatchKernels() {}

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + TornadoMath.exp(-x));
    }

    // ── Hyper-connections ─────────────────────────────────────────────────────

    /** The embedding into every stream of every row. A lane per stream element. */
    public static void hcBroadcast(
            KernelContext context, FloatArray x, FloatArray streams, int dim, int n, int batch) {
        int id = context.globalIdx;
        if (id < batch * n * dim) {
            int b = id / (n * dim);
            int i = id % dim;
            streams.set(id, x.get(b * dim + i));
        }
    }

    /** {@code mixes[b][k] = fn[k] . s_b / rms(s_b)}, a workgroup per (row, mix). */
    public static void hcMixes(
            KernelContext context,
            FloatArray streams,
            FloatArray fn,
            FloatArray mixes,
            int width,
            int mixCount,
            float rmsEps,
            int batch) {
        int g = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] dots = context.allocateFloatLocalArray(GROUP);
        float[] squares = context.allocateFloatLocalArray(GROUP);
        int b = g / mixCount;
        int k = g - b * mixCount;
        float dot = 0.0f;
        float square = 0.0f;
        if (b < batch) {
            for (int i = tid; i < width; i += local) {
                float v = streams.get(b * width + i);
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
        if (b < batch && tid == 0) {
            mixes.set(g, dots[0] / TornadoMath.sqrt(squares[0] / width + rmsEps));
        }
    }

    /** {@link Glm5NextKernels#hcCoefficients} for every row, a lane per row. */
    public static void hcCoefficients(
            KernelContext context,
            FloatArray mixes,
            FloatArray base,
            FloatArray scale,
            FloatArray coeffs,
            int n,
            int iterations,
            float eps,
            float postScale,
            int batch) {
        int b = context.globalIdx;
        if (b < batch) {
            int o = b * (2 + n) * n;
            float s0 = scale.get(0);
            float s1 = scale.get(1);
            float s2 = scale.get(2);
            for (int i = 0; i < n; i++) {
                coeffs.set(o + i, sigmoid(mixes.get(o + i) * s0 + base.get(i)) + eps);
                coeffs.set(
                        o + n + i,
                        postScale * sigmoid(mixes.get(o + n + i) * s1 + base.get(n + i)));
            }
            int c0 = o + 2 * n;
            int b0 = 2 * n;
            for (int r = 0; r < n; r++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < n; c++) {
                    float v = mixes.get(c0 + r * n + c) * s2 + base.get(b0 + r * n + c);
                    coeffs.set(c0 + r * n + c, v);
                    max = TornadoMath.max(max, v);
                }
                float sum = 0.0f;
                for (int c = 0; c < n; c++) {
                    float e = TornadoMath.exp(coeffs.get(c0 + r * n + c) - max);
                    coeffs.set(c0 + r * n + c, e);
                    sum += e;
                }
                for (int c = 0; c < n; c++) {
                    coeffs.set(c0 + r * n + c, coeffs.get(c0 + r * n + c) / sum + eps);
                }
            }
            for (int it = 0; it < iterations; it++) {
                if (it > 0) {
                    for (int r = 0; r < n; r++) {
                        float sum = 0.0f;
                        for (int c = 0; c < n; c++) {
                            sum += coeffs.get(c0 + r * n + c);
                        }
                        for (int c = 0; c < n; c++) {
                            coeffs.set(c0 + r * n + c, coeffs.get(c0 + r * n + c) / (sum + eps));
                        }
                    }
                }
                for (int c = 0; c < n; c++) {
                    float sum = 0.0f;
                    for (int r = 0; r < n; r++) {
                        sum += coeffs.get(c0 + r * n + c);
                    }
                    for (int r = 0; r < n; r++) {
                        coeffs.set(c0 + r * n + c, coeffs.get(c0 + r * n + c) / (sum + eps));
                    }
                }
            }
        }
    }

    /** {@code x_b = sum_m pre_b[m] * streams_b[m]}, saving the streams. A lane per element. */
    public static void hcCollapse(
            KernelContext context,
            FloatArray streams,
            FloatArray coeffs,
            FloatArray residual,
            FloatArray x,
            int dim,
            int n,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * dim) {
            int b = id / dim;
            int i = id - b * dim;
            int o = b * (2 + n) * n;
            float sum = 0.0f;
            for (int m = 0; m < n; m++) {
                float v = streams.get((b * n + m) * dim + i);
                residual.set((b * n + m) * dim + i, v);
                sum += coeffs.get(o + m) * v;
            }
            x.set(id, sum);
        }
    }

    /** {@code streams_b[k] = post_b[k] * branch_b + sum_m comb_b[m][k] * residual_b[m]}. */
    public static void hcExpand(
            KernelContext context,
            FloatArray residual,
            FloatArray branch,
            FloatArray coeffs,
            FloatArray streams,
            int dim,
            int n,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * n * dim) {
            int b = id / (n * dim);
            int k = (id / dim) % n;
            int i = id % dim;
            int o = b * (2 + n) * n;
            float v = coeffs.get(o + n + k) * branch.get(b * dim + i);
            for (int m = 0; m < n; m++) {
                v += coeffs.get(o + 2 * n + m * n + k) * residual.get((b * n + m) * dim + i);
            }
            streams.set(id, v);
        }
    }

    // ── Norms and projections ─────────────────────────────────────────────────

    /** {@code out_b = weight * in_b / rms(in_b)} over {@code size} values, a workgroup per row. */
    public static void rmsNorm(
            KernelContext context,
            FloatArray in,
            FloatArray out,
            FloatArray weight,
            int size,
            float eps,
            int batch) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        float ss = 0.0f;
        if (b < batch) {
            for (int i = tid; i < size; i += local) {
                float v = in.get(b * size + i);
                ss += v * v;
            }
        }
        reduce[tid] = ss;
        context.localBarrier();
        for (int stride = local / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        if (b < batch) {
            float inv = 1.0f / TornadoMath.sqrt(reduce[0] / size + eps);
            for (int i = tid; i < size; i += local) {
                out.set(b * size + i, weight.get(i) * (inv * in.get(b * size + i)));
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

    /**
     * This lane's share of {@code w[row] . x}, the activation's blocks starting at {@code xBlock}.
     */
    private static float rowPartial(
            ByteArray w,
            int row,
            int blocks,
            IntArray xQuants,
            FloatArray xScales,
            int xBlock,
            int lane) {
        float partial = 0.0f;
        for (int b = lane; b < blocks; b += 32) {
            int offset = (row * blocks + b) * BLOCK_BYTES;
            partial +=
                    w.getHalfFloat(offset).getFloat32()
                            * xScales.get(xBlock + b)
                            * blockDot(w, offset, xQuants, (xBlock + b) * (QK / 4));
        }
        return partial;
    }

    // @formatter:off
    /**
     * {@code out_b = w . x_b} for every row of the chunk, Q8_0 weights against the activation
     * quantized per block, a warp per (row, output). Worker: {@code batch * d * 32} lanes.
     */
    // @formatter:on
    public static void projectQ8_0(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray out,
            ByteArray w,
            int n,
            int d,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < batch * d) {
            int b = warp / d;
            int r = warp - b * d;
            int blocks = n / QK;
            float sum =
                    warpSum(context, rowPartial(w, r, blocks, xQuants, xScales, b * blocks, lane));
            if (lane == 0) {
                out.set(warp, sum);
            }
        }
    }

    // @formatter:off
    /**
     * {@code out_b = w . x_b} with FP32 activations: a warp per (row, output) of a Q8_0 matrix
     * whose rows are {@code n} wide, the activation of output {@code r} of row {@code b} at {@code
     * x[b * xStride + (r / rowsPerSlice) * n]}. It is the absorbed query (rows per head, the head's
     * query) and the latent decompression (rows per head, the head's latent).
     */
    // @formatter:on
    public static void projectSlicedF32(
            KernelContext context,
            ByteArray w,
            FloatArray x,
            FloatArray out,
            int n,
            int d,
            int rowsPerSlice,
            int xStride,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < batch * d) {
            int b = warp / d;
            int r = warp - b * d;
            int xBase = b * xStride + (r / rowsPerSlice) * n;
            int blocks = n / QK;
            float partial = 0.0f;
            for (int block = 0; block < blocks; block++) {
                int offset = (r * blocks + block) * BLOCK_BYTES;
                float scale = w.getHalfFloat(offset).getFloat32();
                partial += scale * w.get(offset + 2 + lane) * x.get(xBase + block * QK + lane);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                out.set(warp, sum);
            }
        }
    }

    /** {@code dst[offset + (start + b) * width + i] = src_b[i]} for the real rows. */
    public static void storeRows(
            KernelContext context,
            FloatArray src,
            FloatArray dst,
            IntArray batchInfo,
            int width,
            int offset,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * width) {
            int b = id / width;
            if (b < batchInfo.get(1)) {
                int i = id - b * width;
                dst.set(offset + (batchInfo.get(0) + b) * width + i, src.get(id));
            }
        }
    }

    // ── KDA ───────────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * The causal depthwise convolution over the chunk's real rows in order, a lane per channel,
     * carrying the channel's window from token to token; the window is the decode kernel's ({@link
     * Qwen35DeltaNetKernels#causalConv1d}), so the decode resumes where this leaves it.
     */
    // @formatter:on
    public static void causalConv1d(
            KernelContext context,
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            IntArray batchInfo,
            int channels,
            int kernel,
            int windowOffset) {
        int c = context.globalIdx;
        if (c < channels) {
            int history = kernel - 1;
            int wBase = c * kernel;
            int hBase = windowOffset + c * history;
            int rows = batchInfo.get(1);
            for (int b = 0; b < rows; b++) {
                float x = input.get(b * channels + c);
                float sum = 0.0f;
                for (int t = 0; t < history; t++) {
                    sum += weight.get(wBase + t) * window.get(hBase + t);
                }
                sum += weight.get(wBase + history) * x;
                out.set(b * channels + c, sum);
                for (int t = 0; t + 1 < history; t++) {
                    window.set(hBase + t, window.get(hBase + t + 1));
                }
                if (history > 0) {
                    window.set(hBase + history - 1, x);
                }
            }
        }
    }

    /**
     * {@link Glm5NextKernels#kdaDecayAndBeta} over every row: {@code decay} and {@code beta} flat.
     */
    public static void kdaDecayAndBeta(
            KernelContext context,
            FloatArray decay,
            FloatArray dtBias,
            FloatArray aLog,
            FloatArray beta,
            int width,
            int headDim,
            int heads,
            float lowerBound,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * width) {
            int c = id % width;
            float g = TornadoMath.exp(aLog.get(c / headDim)) * (decay.get(id) + dtBias.get(c));
            decay.set(id, TornadoMath.exp(lowerBound * sigmoid(g)));
        }
        if (id < batch * heads) {
            beta.set(id, sigmoid(beta.get(id)));
        }
    }

    // @formatter:off
    /**
     * The KDA delta rule over the chunk's real rows in order, a lane per (head, value channel)
     * owning one state column, as {@link Glm5NextKernels#kdaDeltaRule} does for one token.
     */
    // @formatter:on
    public static void kdaDeltaRule(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            IntArray batchInfo,
            int heads,
            int headDim,
            int stateOffset) {
        int lane = context.globalIdx;
        if (lane < heads * headDim) {
            int h = lane / headDim;
            int j = lane - h * headDim;
            int width = heads * headDim;
            int stateBase = stateOffset + h * headDim * headDim + j;
            int rows = batchInfo.get(1);
            for (int b = 0; b < rows; b++) {
                int base = b * width + h * headDim;
                float correction = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float s = state.get(stateBase + i * headDim) * decay.get(base + i);
                    state.set(stateBase + i * headDim, s);
                    correction += s * k.get(base + i);
                }
                float delta = (v.get(base + j) - correction) * beta.get(b * heads + h);
                float o = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float s = state.get(stateBase + i * headDim) + k.get(base + i) * delta;
                    state.set(stateBase + i * headDim, s);
                    o += s * q.get(base + i);
                }
                out.set(base + j, o);
            }
        }
    }

    // ── Indexer ───────────────────────────────────────────────────────────────

    /** Each real row's indexer key {@code LayerNorm(raw)} and pool gate, into the caches. */
    public static void indexerStore(
            KernelContext context,
            FloatArray rawKey,
            FloatArray gate,
            FloatArray weight,
            FloatArray bias,
            FloatArray keyCache,
            FloatArray gateCache,
            IntArray batchInfo,
            int headDim,
            int cacheOffset,
            float eps) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] sums = context.allocateFloatLocalArray(GROUP);
        float[] squares = context.allocateFloatLocalArray(GROUP);
        int rows = batchInfo.get(1);
        float sum = 0.0f;
        float square = 0.0f;
        if (b < rows) {
            for (int i = tid; i < headDim; i += local) {
                float v = rawKey.get(b * headDim + i);
                sum += v;
                square += v * v;
            }
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
        if (b < rows) {
            float mean = sums[0] / headDim;
            float inv = 1.0f / TornadoMath.sqrt(squares[0] / headDim - mean * mean + eps);
            int row = cacheOffset + (batchInfo.get(0) + b) * headDim;
            for (int i = tid; i < headDim; i += local) {
                keyCache.set(
                        row + i,
                        (rawKey.get(b * headDim + i) - mean) * inv * weight.get(i) + bias.get(i));
                gateCache.set(row + i, gate.get(b * headDim + i));
            }
        }
    }

    // @formatter:off
    /**
     * The compressed key of every pool the chunk completes, a lane per (pool slot, channel): pool
     * {@code start / pool + j} for {@code j} below {@code poolSlots}, when its last position is one
     * of the chunk's real rows.
     */
    // @formatter:on
    public static void indexerPoolKeys(
            KernelContext context,
            FloatArray keyCache,
            FloatArray gateCache,
            FloatArray ape,
            FloatArray poolKeys,
            IntArray batchInfo,
            int headDim,
            int pool,
            int cacheOffset,
            int poolOffset,
            int poolSlots) {
        int id = context.globalIdx;
        if (id < poolSlots * headDim) {
            int j = id / headDim;
            int c = id - j * headDim;
            int start = batchInfo.get(0);
            int p = start / pool + j;
            int end = p * pool + pool - 1;
            if (end >= start && end < start + batchInfo.get(1)) {
                int first = cacheOffset + p * pool * headDim + c;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = 0; s < pool; s++) {
                    max =
                            TornadoMath.max(
                                    max,
                                    gateCache.get(first + s * headDim) + ape.get(s * headDim + c));
                }
                float sum = 0.0f;
                float acc = 0.0f;
                for (int s = 0; s < pool; s++) {
                    float e =
                            TornadoMath.exp(
                                    gateCache.get(first + s * headDim)
                                            + ape.get(s * headDim + c)
                                            - max);
                    sum += e;
                    acc += e * keyCache.get(first + s * headDim);
                }
                poolKeys.set(poolOffset + p * headDim + c, acc / sum);
            }
        }
    }

    /**
     * Every row's pool scores, as {@link Glm5NextKernels#indexerScores}; a lane per (row, pool).
     */
    public static void indexerScores(
            KernelContext context,
            FloatArray query,
            FloatArray headWeights,
            FloatArray poolKeys,
            FloatArray scores,
            IntArray batchInfo,
            int heads,
            int headDim,
            int pool,
            int maxPools,
            int poolOffset,
            int batch) {
        int id = context.globalIdx;
        if (id < batch * maxPools) {
            int b = id / maxPools;
            int p = id - b * maxPools;
            int complete = (batchInfo.get(0) + b + 1) / pool;
            if (b < batchInfo.get(1) && p < complete) {
                float headScale = 1.0f / TornadoMath.sqrt(heads);
                float keyScale = 1.0f / TornadoMath.sqrt(headDim);
                float score = 0.0f;
                for (int h = 0; h < heads; h++) {
                    float dot = 0.0f;
                    for (int c = 0; c < headDim; c++) {
                        dot +=
                                query.get((b * heads + h) * headDim + c)
                                        * poolKeys.get(poolOffset + p * headDim + c);
                    }
                    score +=
                            headWeights.get(b * heads + h)
                                    * headScale
                                    * TornadoMath.max(dot * keyScale, 0.0f);
                }
                scores.set(id, score);
            } else {
                scores.set(id, Float.NEGATIVE_INFINITY);
            }
        }
    }

    /** {@link Glm5NextKernels#indexerSelect} for every real row, a workgroup per row. */
    public static void indexerSelect(
            KernelContext context,
            FloatArray scores,
            IntArray selected,
            IntArray batchInfo,
            int pool,
            int maxPools,
            int budget) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] reduceMin = context.allocateFloatLocalArray(GROUP);
        float[] reduceMax = context.allocateFloatLocalArray(GROUP);
        int[] counts = context.allocateIntLocalArray(GROUP);
        int o = b * maxPools;
        int real = b < batchInfo.get(1) ? 1 : 0;
        int complete = real * TornadoMath.min((batchInfo.get(0) + b + 1) / pool, maxPools);
        for (int p = tid; p < maxPools; p += local) {
            selected.set(o + p, 0);
        }
        context.localBarrier();
        if (complete <= budget) {
            for (int p = tid; p < complete; p += local) {
                selected.set(o + p, 1);
            }
        } else {
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (int p = tid; p < complete; p += local) {
                lo = TornadoMath.min(lo, scores.get(o + p));
                hi = TornadoMath.max(hi, scores.get(o + p));
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
            float low = reduceMin[0];
            float high = reduceMax[0] + 1.0f + TornadoMath.abs(reduceMax[0]);
            for (int it = 0; it < 64; it++) {
                float mid = 0.5f * (low + high);
                int count = 0;
                for (int p = tid; p < complete; p += local) {
                    count += scores.get(o + p) >= mid ? 1 : 0;
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
                int chosen = scores.get(o + p) >= high ? 1 : 0;
                selected.set(o + p, chosen);
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
                    float s = scores.get(o + p);
                    if (s >= low && s < high) {
                        selected.set(o + p, 1);
                        need--;
                    }
                }
            }
        }
    }

    // ── Sparse latent attention ───────────────────────────────────────────────

    // @formatter:off
    /**
     * Absorbed attention over each real row's selection, a workgroup per (row, head), as an online
     * softmax over tiles of {@code local} positions: each lane scores one position of the tile, the
     * tile's maximum rescales the running sum and the accumulated latent, and each lane then adds
     * the tile's weighted latents into its own channels. Rows see positions up to their own.
     */
    // @formatter:on
    public static void sparseLatentAttention(
            KernelContext context,
            FloatArray absorbed,
            FloatArray latents,
            IntArray selected,
            FloatArray latentOut,
            IntArray batchInfo,
            int heads,
            int rank,
            int pool,
            int maxPools,
            int cacheOffset,
            int selectTail,
            float scale) {
        int g = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] tile = context.allocateFloatLocalArray(GROUP);
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        float[] acc = context.allocateFloatLocalArray(MAX_RANK);
        float[] query = context.allocateFloatLocalArray(MAX_RANK);
        int b = g / heads;
        int real = b < batchInfo.get(1) ? 1 : 0;
        int position = batchInfo.get(0) + b;
        int last = real * (position + 1);
        int tailStart = (position + 1) / pool * pool;
        for (int r = tid; r < rank; r += local) {
            acc[r] = 0.0f;
            query[r] = absorbed.get(g * rank + r);
        }
        context.localBarrier();
        float runningMax = Float.NEGATIVE_INFINITY;
        float runningSum = 0.0f;
        for (int t0 = 0; t0 < last; t0 += local) {
            int s = t0 + tid;
            int inRange = s < last ? 1 : 0;
            int safe = inRange * s;
            int tail = selectTail == 1 && s >= tailStart ? 1 : 0;
            int visible = inRange * TornadoMath.max(selected.get(b * maxPools + safe / pool), tail);
            if (visible == 1) {
                float dot = 0.0f;
                for (int r = 0; r < rank; r++) {
                    dot += query[r] * latents.get(cacheOffset + s * rank + r);
                }
                tile[tid] = dot * scale;
            } else {
                tile[tid] = Float.NEGATIVE_INFINITY;
            }
            reduce[tid] = tile[tid];
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
                }
                context.localBarrier();
            }
            float newMax = TornadoMath.max(runningMax, reduce[0]);
            context.localBarrier();
            // A tile with nothing visible leaves the running values as they were.
            float safeMax = newMax == Float.NEGATIVE_INFINITY ? 0.0f : newMax;
            float rescale = TornadoMath.exp(runningMax - safeMax);
            float p = TornadoMath.exp(tile[tid] - safeMax);
            tile[tid] = p;
            reduce[tid] = p;
            context.localBarrier();
            for (int stride = local / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                context.localBarrier();
            }
            runningSum = runningSum * rescale + reduce[0];
            runningMax = newMax;
            int count = TornadoMath.min(local, last - t0);
            for (int r = tid; r < rank; r += local) {
                float a = acc[r] * rescale;
                for (int j = 0; j < count; j++) {
                    a += tile[j] * latents.get(cacheOffset + (t0 + j) * rank + r);
                }
                acc[r] = a;
            }
            context.localBarrier();
        }
        if (real == 1) {
            for (int r = tid; r < rank; r += local) {
                latentOut.set(g * rank + r, acc[r] / runningSum);
            }
        }
    }

    // ── Feed-forward ──────────────────────────────────────────────────────────

    /** {@code logits_b[e] = router[e] . x_b}, F32, a warp per (row, expert). */
    public static void routerLogits(
            KernelContext context,
            FloatArray x,
            FloatArray router,
            FloatArray logits,
            int dim,
            int experts,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < batch * experts) {
            int b = warp / experts;
            int e = warp - b * experts;
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += router.get(e * dim + i) * x.get(b * dim + i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                logits.set(warp, sum);
            }
        }
    }

    /** {@link Glm5NextKernels#routerTopKSigmoid} for every row, a workgroup per row. */
    public static void routerTopKSigmoid(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used,
            int normalize,
            float scale,
            int batch) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int local = context.localGroupSizeX;
        float[] select = context.allocateFloatLocalArray(MAX_EXPERTS);
        float[] best = context.allocateFloatLocalArray(GROUP);
        int[] bestIndex = context.allocateIntLocalArray(GROUP);
        int lo = b * experts;
        for (int e = tid; e < experts; e += local) {
            select[e] = sigmoid(logits.get(lo + e)) + bias.get(e);
        }
        context.localBarrier();
        for (int k = 0; k < used; k++) {
            float bv = Float.NEGATIVE_INFINITY;
            int index = experts;
            for (int e = tid; e < experts; e += local) {
                float s = select[e];
                int better = s > bv ? 1 : 0;
                bv = better == 1 ? s : bv;
                index = better == 1 ? e : index;
            }
            best[tid] = bv;
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
            if (tid == 0 && b < batch) {
                int chosen = bestIndex[0];
                ids.set(b * used + k, chosen);
                weights.set(b * used + k, sigmoid(logits.get(lo + chosen)));
                select[chosen] = Float.NEGATIVE_INFINITY;
            }
            context.localBarrier();
        }
        if (tid == 0 && b < batch) {
            float total = 0.0f;
            for (int k = 0; k < used; k++) {
                total += weights.get(b * used + k);
            }
            // A blend rather than a branch: a value merged from an if/else is miscompiled by the
            // CUDA backend.
            float n = normalize;
            float norm = n * (total + 1e-20f) + (1.0f - n);
            for (int k = 0; k < used; k++) {
                weights.set(b * used + k, weights.get(b * used + k) / norm * scale);
            }
        }
    }

    private static float clampedSilu(float gate, float up, float limit) {
        float g = TornadoMath.min(gate, limit);
        float u = TornadoMath.max(-limit, TornadoMath.min(up, limit));
        return g / (1.0f + TornadoMath.exp(-g)) * u;
    }

    // @formatter:off
    /**
     * {@link Glm5NextKernels#expertsGateUpClamped} for every row: row {@code b}'s hidden
     * activations at {@code hidden[b * (used * hiddenDim + sharedHiddenDim)]}. Worker: {@code batch
     * * (used * hiddenDim + sharedHiddenDim) * 32} lanes.
     */
    // @formatter:on
    public static void expertsGateUpClamped(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray ids,
            ByteArray gate0,
            ByteArray up0,
            ByteArray gate1,
            ByteArray up1,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split,
            float limit,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int perRow = used * hiddenDim + sharedHiddenDim;
        int blocks = dim / QK;
        if (warp < batch * perRow) {
            int b = warp / perRow;
            int w = warp - b * perRow;
            int xBlock = b * blocks;
            if (w < used * hiddenDim) {
                int slot = w / hiddenDim;
                int r = w - slot * hiddenDim;
                int e = ids.get(b * used + slot);
                if (e < split) {
                    int row = e * hiddenDim + r;
                    float g =
                            warpSum(
                                    context,
                                    rowPartial(gate0, row, blocks, xQuants, xScales, xBlock, lane));
                    float u =
                            warpSum(
                                    context,
                                    rowPartial(up0, row, blocks, xQuants, xScales, xBlock, lane));
                    if (lane == 0) {
                        hidden.set(warp, clampedSilu(g, u, limit));
                    }
                } else {
                    int row = (e - split) * hiddenDim + r;
                    float g =
                            warpSum(
                                    context,
                                    rowPartial(gate1, row, blocks, xQuants, xScales, xBlock, lane));
                    float u =
                            warpSum(
                                    context,
                                    rowPartial(up1, row, blocks, xQuants, xScales, xBlock, lane));
                    if (lane == 0) {
                        hidden.set(warp, clampedSilu(g, u, limit));
                    }
                }
            } else {
                int row = w - used * hiddenDim;
                float g =
                        warpSum(
                                context,
                                rowPartial(
                                        sharedGate, row, blocks, xQuants, xScales, xBlock, lane));
                float u =
                        warpSum(
                                context,
                                rowPartial(sharedUp, row, blocks, xQuants, xScales, xBlock, lane));
                if (lane == 0) {
                    hidden.set(warp, clampedSilu(g, u, limit));
                }
            }
        }
    }

    /** {@link Glm5NextKernels#expertsDown} for every row, a warp per (row, output). */
    public static void expertsDown(
            KernelContext context,
            IntArray hQuants,
            FloatArray hScales,
            IntArray ids,
            FloatArray weights,
            ByteArray down0,
            ByteArray down1,
            ByteArray sharedDown,
            FloatArray out,
            int dim,
            int hiddenDim,
            int sharedHiddenDim,
            int used,
            int split,
            int batch) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < batch * dim) {
            int b = warp / dim;
            int row = warp - b * dim;
            int perSlot = hiddenDim / QK;
            int routedPairs = used * perSlot;
            int hBlock = b * (used * hiddenDim + sharedHiddenDim) / QK;
            float first = 0.0f;
            float second = 0.0f;
            for (int p = lane; p < routedPairs; p += 32) {
                int slot = p / perSlot;
                int block = p - slot * perSlot;
                int e = ids.get(b * used + slot);
                int inFirst = e < split ? 1 : 0;
                int e0 = inFirst * e;
                int e1 = (1 - inFirst) * (e - split);
                int offset0 = ((e0 * dim + row) * perSlot + block) * BLOCK_BYTES;
                int offset1 = ((e1 * dim + row) * perSlot + block) * BLOCK_BYTES;
                float w = weights.get(b * used + slot) * hScales.get(hBlock + p);
                first +=
                        inFirst
                                * w
                                * down0.getHalfFloat(offset0).getFloat32()
                                * blockDot(down0, offset0, hQuants, (hBlock + p) * (QK / 4));
                second +=
                        (1 - inFirst)
                                * w
                                * down1.getHalfFloat(offset1).getFloat32()
                                * blockDot(down1, offset1, hQuants, (hBlock + p) * (QK / 4));
            }
            int sharedBlocks = sharedHiddenDim / QK;
            float shared = 0.0f;
            for (int s = lane; s < sharedBlocks; s += 32) {
                int offset = (row * sharedBlocks + s) * BLOCK_BYTES;
                int p = hBlock + routedPairs + s;
                shared +=
                        sharedDown.getHalfFloat(offset).getFloat32()
                                * hScales.get(p)
                                * blockDot(sharedDown, offset, hQuants, p * (QK / 4));
            }
            float sum = warpSum(context, first + second + shared);
            if (lane == 0) {
                out.set(warp, sum);
            }
        }
    }
}
