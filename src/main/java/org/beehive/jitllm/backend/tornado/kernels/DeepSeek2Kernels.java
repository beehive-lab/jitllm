package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@code deepseek2} decode kernels that no other family has: absorbed multi-head latent
 * attention, and routing by sigmoid scores with a selection bias.
 *
 * <p>Attention follows llama.cpp's absorbed form. A position's cache row is {@code keyWidth = rank
 * + rope} wide: the normalized latent, then the rotated key. It is every head's key, and its latent
 * half is every head's value. A head's query is its no-rope part pushed through {@code attn_k_b}
 * into the latent space, followed by its rotated part; its output is the attended latent pushed
 * back out through {@code attn_v_b}.
 *
 * <p>The projections the layer shares with other families run through the Q8_0 packed-integer
 * kernels; {@code attn_k_b} and {@code attn_v_b} are read here because each head multiplies its own
 * slice of the input.
 */
// @formatter:on
public final class DeepSeek2Kernels {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    /** Lanes of a norm or attention workgroup. */
    public static final int GROUP = 256;

    private DeepSeek2Kernels() {}

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    /**
     * {@code out[i] = weight[i] * in[i] / rms(in[0..n))}, by one workgroup of {@link #GROUP} lanes.
     * {@code in} and {@code out} may be the same array: each lane writes only what it read.
     */
    public static void rmsNorm(
            KernelContext context,
            FloatArray in,
            FloatArray out,
            FloatArray weight,
            int n,
            float eps) {
        int tid = context.localIdx;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        float partial = 0.0f;
        for (int i = tid; i < n; i += GROUP) {
            float v = in.get(i);
            partial += v * v;
        }
        reduce[tid] = partial;
        context.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float scale = 1.0f / TornadoMath.sqrt(reduce[0] / n + eps);
        for (int i = tid; i < n; i += GROUP) {
            out.set(i, weight.get(i) * (in.get(i) * scale));
        }
    }

    // @formatter:off
    /**
     * Rotates every head's query and the position's key in interleaved pairs, and writes the
     * position's cache row: the latent as normalized, then the rotated key. Lanes: {@code heads *
     * rope / 2} query pairs, then {@code rope / 2} key pairs, then {@code rank} latent values.
     */
    // @formatter:on
    public static void ropeAndCache(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,
            FloatArray compressedKv,
            FloatArray freqReal,
            FloatArray freqImag,
            FloatArray keyCache,
            int heads,
            int queryHead,
            int nope,
            int rope,
            int rank,
            int cacheOffset) {
        int idx = context.globalIdx;
        int pos = positionHolder.get(0);
        int pairs = rope / 2;
        int queryPairs = heads * pairs;
        int keyWidth = rank + rope;
        int row = cacheOffset + pos * keyWidth;
        if (idx < queryPairs) {
            int h = idx / pairs;
            int i = idx - h * pairs;
            int base = h * queryHead + nope + 2 * i;
            float fcr = freqReal.get(pos * pairs + i);
            float fci = freqImag.get(pos * pairs + i);
            float v0 = q.get(base);
            float v1 = q.get(base + 1);
            q.set(base, v0 * fcr - v1 * fci);
            q.set(base + 1, v0 * fci + v1 * fcr);
        }
        if (idx >= queryPairs && idx < queryPairs + pairs) {
            int i = idx - queryPairs;
            float fcr = freqReal.get(pos * pairs + i);
            float fci = freqImag.get(pos * pairs + i);
            float v0 = compressedKv.get(rank + 2 * i);
            float v1 = compressedKv.get(rank + 2 * i + 1);
            keyCache.set(row + rank + 2 * i, v0 * fcr - v1 * fci);
            keyCache.set(row + rank + 2 * i + 1, v0 * fci + v1 * fcr);
        }
        if (idx >= queryPairs + pairs && idx < queryPairs + pairs + rank) {
            int r = idx - queryPairs - pairs;
            keyCache.set(row + r, compressedKv.get(r));
        }
    }

    /** {@link #ropeAndCache} over the half-precision cache. */
    public static void ropeAndCacheFP16(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,
            FloatArray compressedKv,
            FloatArray freqReal,
            FloatArray freqImag,
            HalfFloatArray keyCache,
            int heads,
            int queryHead,
            int nope,
            int rope,
            int rank,
            int cacheOffset) {
        int idx = context.globalIdx;
        int pos = positionHolder.get(0);
        int pairs = rope / 2;
        int queryPairs = heads * pairs;
        int keyWidth = rank + rope;
        int row = cacheOffset + pos * keyWidth;
        if (idx < queryPairs) {
            int h = idx / pairs;
            int i = idx - h * pairs;
            int base = h * queryHead + nope + 2 * i;
            float fcr = freqReal.get(pos * pairs + i);
            float fci = freqImag.get(pos * pairs + i);
            float v0 = q.get(base);
            float v1 = q.get(base + 1);
            q.set(base, v0 * fcr - v1 * fci);
            q.set(base + 1, v0 * fci + v1 * fcr);
        }
        if (idx >= queryPairs && idx < queryPairs + pairs) {
            int i = idx - queryPairs;
            float fcr = freqReal.get(pos * pairs + i);
            float fci = freqImag.get(pos * pairs + i);
            float v0 = compressedKv.get(rank + 2 * i);
            float v1 = compressedKv.get(rank + 2 * i + 1);
            keyCache.set(row + rank + 2 * i, new HalfFloat(v0 * fcr - v1 * fci));
            keyCache.set(row + rank + 2 * i + 1, new HalfFloat(v0 * fci + v1 * fcr));
        }
        if (idx >= queryPairs + pairs && idx < queryPairs + pairs + rank) {
            int r = idx - queryPairs - pairs;
            keyCache.set(row + r, new HalfFloat(compressedKv.get(r)));
        }
    }

    // @formatter:off
    /**
     * {@code absorbed[h * keyWidth + r] = kB[h * rank + r] . q[h * queryHead .. + nope]}, a warp
     * per row of {@code attn_k_b} (Q8_0, {@code nope} wide); then warps past the last row copy each
     * head's rotated query after its latent part. Worker: {@code heads * rank + heads * rope / 32}
     * warps.
     */
    // @formatter:on
    public static void absorbQuery(
            KernelContext context,
            ByteArray kB,
            FloatArray q,
            FloatArray absorbed,
            int heads,
            int rank,
            int nope,
            int rope,
            int queryHead) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int rows = heads * rank;
        int keyWidth = rank + rope;
        if (warp < rows) {
            int h = warp / rank;
            int r = warp - h * rank;
            int blocks = nope / QK;
            float partial = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int offset = (warp * blocks + b) * BLOCK_BYTES;
                partial +=
                        kB.getHalfFloat(offset).getFloat32()
                                * kB.get(offset + 2 + lane)
                                * q.get(h * queryHead + b * QK + lane);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                absorbed.set(h * keyWidth + r, sum);
            }
        }
        if (warp >= rows) {
            int idx = (warp - rows) * 32 + lane;
            if (idx < heads * rope) {
                int h = idx / rope;
                int i = idx - h * rope;
                absorbed.set(h * keyWidth + rank + i, q.get(h * queryHead + nope + i));
            }
        }
    }

    // @formatter:off
    /**
     * Decode attention of one head over one slice of the positions so far, a workgroup of {@link
     * #GROUP} lanes per (head, slice). Scores a warp per position, coalesced over the key row; the
     * slice's softmax state; and the unnormalized attended latent, a lane per latent value. Each
     * slice writes {@code rank} numerators, its maximum and its sum at {@code (head * splits +
     * slice) * (rank + 2)}; an empty slice writes zeros and a maximum of minus infinity.
     */
    // @formatter:on
    public static void attentionSplit(
            KernelContext context,
            IntArray positionHolder,
            FloatArray absorbed,
            FloatArray keyCache,
            FloatArray att,
            FloatArray partial,
            int heads,
            int rank,
            int rope,
            int cacheOffset,
            int contextLength,
            int splits,
            float scale) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int h = group / splits;
        int split = group - h * splits;
        if (h < heads) {
            int keyWidth = rank + rope;
            int pos = positionHolder.get(0);
            int total = pos + 1;
            int chunk = (total + splits - 1) / splits;
            int from = split * chunk;
            int to = Math.min(pos, from + chunk - 1);
            int base = (h * splits + split) * (rank + 2);
            int scores = h * contextLength;

            float[] qs = context.allocateFloatLocalArray(1024);
            float[] reduce = context.allocateFloatLocalArray(GROUP);

            if (from > to) {
                for (int r = tid; r < rank; r += GROUP) {
                    partial.set(base + r, 0.0f);
                }
                if (tid == 0) {
                    partial.set(base + rank, Float.NEGATIVE_INFINITY);
                    partial.set(base + rank + 1, 0.0f);
                }
                return;
            }

            for (int i = tid; i < keyWidth; i += GROUP) {
                qs[i] = absorbed.get(h * keyWidth + i);
            }
            context.localBarrier();

            for (int t = from + warp; t <= to; t += GROUP / 32) {
                int row = cacheOffset + t * keyWidth;
                float dot = 0.0f;
                for (int i = lane; i < keyWidth; i += 32) {
                    dot += qs[i] * keyCache.get(row + i);
                }
                dot = warpSum(context, dot);
                if (lane == 0) {
                    att.set(scores + t, dot * scale);
                }
            }
            context.localBarrier();

            float localMax = Float.NEGATIVE_INFINITY;
            for (int t = from + tid; t <= to; t += GROUP) {
                localMax = TornadoMath.max(localMax, att.get(scores + t));
            }
            reduce[tid] = localMax;
            context.localBarrier();
            for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
                }
                context.localBarrier();
            }
            float sliceMax = reduce[0];
            context.localBarrier();

            float localSum = 0.0f;
            for (int t = from + tid; t <= to; t += GROUP) {
                float e = TornadoMath.exp(att.get(scores + t) - sliceMax);
                att.set(scores + t, e);
                localSum += e;
            }
            reduce[tid] = localSum;
            context.localBarrier();
            for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                context.localBarrier();
            }
            float sliceSum = reduce[0];

            for (int r = tid; r < rank; r += GROUP) {
                float acc = 0.0f;
                for (int t = from; t <= to; t++) {
                    acc += att.get(scores + t) * keyCache.get(cacheOffset + t * keyWidth + r);
                }
                partial.set(base + r, acc);
            }
            if (tid == 0) {
                partial.set(base + rank, sliceMax);
                partial.set(base + rank + 1, sliceSum);
            }
        }
    }

    /** {@link #attentionSplit} over the half-precision cache. */
    public static void attentionSplitFP16(
            KernelContext context,
            IntArray positionHolder,
            FloatArray absorbed,
            HalfFloatArray keyCache,
            FloatArray att,
            FloatArray partial,
            int heads,
            int rank,
            int rope,
            int cacheOffset,
            int contextLength,
            int splits,
            float scale) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int group = context.groupIdx;
        int h = group / splits;
        int split = group - h * splits;
        if (h < heads) {
            int keyWidth = rank + rope;
            int pos = positionHolder.get(0);
            int total = pos + 1;
            int chunk = (total + splits - 1) / splits;
            int from = split * chunk;
            int to = Math.min(pos, from + chunk - 1);
            int base = (h * splits + split) * (rank + 2);
            int scores = h * contextLength;

            float[] qs = context.allocateFloatLocalArray(1024);
            float[] reduce = context.allocateFloatLocalArray(GROUP);

            if (from > to) {
                for (int r = tid; r < rank; r += GROUP) {
                    partial.set(base + r, 0.0f);
                }
                if (tid == 0) {
                    partial.set(base + rank, Float.NEGATIVE_INFINITY);
                    partial.set(base + rank + 1, 0.0f);
                }
                return;
            }

            for (int i = tid; i < keyWidth; i += GROUP) {
                qs[i] = absorbed.get(h * keyWidth + i);
            }
            context.localBarrier();

            for (int t = from + warp; t <= to; t += GROUP / 32) {
                int row = cacheOffset + t * keyWidth;
                float dot = 0.0f;
                for (int i = lane; i < keyWidth; i += 32) {
                    dot += qs[i] * keyCache.get(row + i).getFloat32();
                }
                dot = warpSum(context, dot);
                if (lane == 0) {
                    att.set(scores + t, dot * scale);
                }
            }
            context.localBarrier();

            float localMax = Float.NEGATIVE_INFINITY;
            for (int t = from + tid; t <= to; t += GROUP) {
                localMax = TornadoMath.max(localMax, att.get(scores + t));
            }
            reduce[tid] = localMax;
            context.localBarrier();
            for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
                }
                context.localBarrier();
            }
            float sliceMax = reduce[0];
            context.localBarrier();

            float localSum = 0.0f;
            for (int t = from + tid; t <= to; t += GROUP) {
                float e = TornadoMath.exp(att.get(scores + t) - sliceMax);
                att.set(scores + t, e);
                localSum += e;
            }
            reduce[tid] = localSum;
            context.localBarrier();
            for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                context.localBarrier();
            }
            float sliceSum = reduce[0];

            for (int r = tid; r < rank; r += GROUP) {
                float acc = 0.0f;
                for (int t = from; t <= to; t++) {
                    acc +=
                            att.get(scores + t)
                                    * keyCache.get(cacheOffset + t * keyWidth + r).getFloat32();
                }
                partial.set(base + r, acc);
            }
            if (tid == 0) {
                partial.set(base + rank, sliceMax);
                partial.set(base + rank + 1, sliceSum);
            }
        }
    }

    /**
     * Merges the slices of {@link #attentionSplit}: a lane per (head, latent value). Worker: {@code
     * heads * rank} lanes.
     */
    public static void combineSplits(
            KernelContext context,
            FloatArray partial,
            FloatArray latentOut,
            int heads,
            int rank,
            int splits) {
        int idx = context.globalIdx;
        if (idx < heads * rank) {
            int h = idx / rank;
            int r = idx - h * rank;
            int stride = rank + 2;
            float max = Float.NEGATIVE_INFINITY;
            for (int s = 0; s < splits; s++) {
                max = TornadoMath.max(max, partial.get((h * splits + s) * stride + rank));
            }
            float num = 0.0f;
            float den = 0.0f;
            for (int s = 0; s < splits; s++) {
                int base = (h * splits + s) * stride;
                float f = TornadoMath.exp(partial.get(base + rank) - max);
                num += f * partial.get(base + r);
                den += f * partial.get(base + rank + 1);
            }
            latentOut.set(idx, num / den);
        }
    }

    /**
     * {@code out[h * valueHead + j] = vB[h * valueHead + j] . latent[h * rank .. + rank]}, a warp
     * per row of {@code attn_v_b} (Q8_0, {@code rank} wide). Worker: {@code heads * valueHead}
     * warps.
     */
    public static void decompressLatent(
            KernelContext context,
            ByteArray vB,
            FloatArray latent,
            FloatArray out,
            int heads,
            int valueHead,
            int rank) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < heads * valueHead) {
            int h = warp / valueHead;
            int blocks = rank / QK;
            float partial = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int offset = (warp * blocks + b) * BLOCK_BYTES;
                partial +=
                        vB.getHalfFloat(offset).getFloat32()
                                * vB.get(offset + 2 + lane)
                                * latent.get(h * rank + b * QK + lane);
            }
            float sum = warpSum(context, partial);
            if (lane == 0) {
                out.set(warp, sum);
            }
        }
    }

    /** {@code logits[e] = router[e] . x}, a warp per expert (F32 router). */
    public static void routerLogits(
            KernelContext context,
            FloatArray x,
            FloatArray router,
            FloatArray logits,
            int dim,
            int experts) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (warp < experts) {
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += router.get(warp * dim + i) * x.get(i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                logits.set(warp, sum);
            }
        }
    }

    // @formatter:off
    /**
     * llama.cpp's sigmoid routing, by one warp: probabilities {@code p = sigmoid(logit)}; the
     * {@code used} experts with the highest {@code p + bias} (a tie goes to the lower index); their
     * weights the unbiased {@code p}, divided by their sum when {@code normalize} is one (floored
     * at 6.1e-5), then multiplied by {@code scale}. Also sets the shared expert's weight to one.
     * Worker: one block of 32 lanes.
     */
    // @formatter:on
    public static void routerTopKSigmoid(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            FloatArray sharedGate,
            int experts,
            int used,
            int normalize,
            float scale) {
        int lane = context.localIdx;
        float[] probs = context.allocateFloatLocalArray(256);
        float[] select = context.allocateFloatLocalArray(256);
        if (lane < 32) {
            for (int e = lane; e < experts; e += 32) {
                float p = 1.0f / (1.0f + TornadoMath.exp(-logits.get(e)));
                probs[e] = p;
                select[e] = p + bias.get(e);
            }
        }
        context.localBarrier();
        float total = 0.0f;
        float mine = 0.0f;
        for (int k = 0; k < used; k++) {
            float best = Float.NEGATIVE_INFINITY;
            float bestIndex = 1.0e9f;
            for (int e = lane; e < experts; e += 32) {
                float s = select[e];
                if (s > best) {
                    best = s;
                    bestIndex = e;
                }
            }
            float max = best;
            max = TornadoMath.max(max, context.simdShuffleDown(max, 16));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 8));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 4));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 2));
            max = TornadoMath.max(max, context.simdShuffleDown(max, 1));
            max = context.simdBroadcastFirst(max);
            float index = best == max ? bestIndex : 1.0e9f;
            index = TornadoMath.min(index, context.simdShuffleDown(index, 16));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 8));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 4));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 2));
            index = TornadoMath.min(index, context.simdShuffleDown(index, 1));
            int chosen = (int) context.simdBroadcastFirst(index);
            context.localBarrier();
            float w = probs[chosen];
            total += w;
            if (lane == k) {
                mine = w;
            }
            if ((chosen & 31) == lane) {
                select[chosen] = Float.NEGATIVE_INFINITY;
            }
            if (lane == 0) {
                ids.set(k, chosen);
            }
            context.localBarrier();
        }
        // No branch on normalize: a value merged from an if/else is miscompiled by the CUDA
        // backend, so the denominator is a blend of the two cases.
        float n = normalize;
        float denominator = n * TornadoMath.max(total, 6.103515625e-5f) + (1.0f - n);
        if (lane < used) {
            weights.set(lane, mine / denominator * scale);
        }
        if (lane == 0) {
            sharedGate.set(0, 1.0f);
        }
    }

    // ── fused decode kernels ──────────────────────────────────────────────────

    /**
     * {@code out = weight * in / rms(in)} over {@code n} by the whole workgroup; {@code reduce} is
     * its scratch.
     */
    private static void normInto(
            KernelContext context,
            float[] reduce,
            FloatArray in,
            FloatArray out,
            FloatArray weight,
            int n,
            float eps) {
        int tid = context.localIdx;
        float partial = 0.0f;
        for (int i = tid; i < n; i += GROUP) {
            float v = in.get(i);
            partial += v * v;
        }
        reduce[tid] = partial;
        context.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float scale = 1.0f / TornadoMath.sqrt(reduce[0] / n + eps);
        for (int i = tid; i < n; i += GROUP) {
            out.set(i, weight.get(i) * (in.get(i) * scale));
        }
        context.localBarrier();
    }

    // @formatter:off
    /**
     * {@code x}'s first {@code n} values in Q8 blocks — the arithmetic of {@link
     * TransformerComputeKernelsQ4_0#quantizeActivationQ8Blocks} — a warp per block: the block's
     * maximum and its sum of quants by shuffles, and the quants packed four to an int by the lanes
     * that hold the first of each four.
     */
    // @formatter:on
    private static void quantizeInto(
            KernelContext context,
            FloatArray x,
            IntArray quants,
            FloatArray scales,
            IntArray sums,
            int n) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int blocks = n / QK;
        for (int b = warp; b < blocks; b += GROUP / 32) {
            float v = x.get(b * QK + lane);
            float m = TornadoMath.abs(v);
            m = TornadoMath.max(m, context.simdShuffleDown(m, 16));
            m = TornadoMath.max(m, context.simdShuffleDown(m, 8));
            m = TornadoMath.max(m, context.simdShuffleDown(m, 4));
            m = TornadoMath.max(m, context.simdShuffleDown(m, 2));
            m = TornadoMath.max(m, context.simdShuffleDown(m, 1));
            m = context.simdBroadcastFirst(m);
            float inverse = m > 0.0f ? 127.0f / m : 0.0f;
            float scaled = v * inverse;
            int q = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
            q = TornadoMath.min(127, TornadoMath.max(-127, q));
            float qf = q;
            float q1 = context.simdShuffleDown(qf, 1);
            float q2 = context.simdShuffleDown(qf, 2);
            float q3 = context.simdShuffleDown(qf, 3);
            float total = warpSum(context, qf);
            if ((lane & 3) == 0) {
                int packed =
                        (q & 0xFF)
                                | (((int) q1 & 0xFF) << 8)
                                | (((int) q2 & 0xFF) << 16)
                                | (((int) q3 & 0xFF) << 24);
                quants.set(b * (QK / 4) + (lane >> 2), packed);
            }
            if (lane == 0) {
                scales.set(b, m / 127.0f);
                sums.set(b, (int) total);
            }
        }
    }

    /**
     * {@link #rmsNorm} and the Q8 block quantization of its output, by one workgroup of {@link
     * #GROUP} lanes.
     */
    public static void rmsNormQuantize(
            KernelContext context,
            FloatArray in,
            FloatArray out,
            FloatArray weight,
            IntArray quants,
            FloatArray scales,
            IntArray sums,
            int n,
            float eps) {
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        normInto(context, reduce, in, out, weight, n, eps);
        quantizeInto(context, out, quants, scales, sums, n);
    }

    // @formatter:off
    /**
     * The two latent norms in one launch: workgroup 0 normalizes the compressed query in place and
     * quantizes it for {@code attn_q_b}; workgroup 1 normalizes the latent half of the compressed
     * key/value in place.
     */
    // @formatter:on
    public static void latentNorms(
            KernelContext context,
            FloatArray queryLatent,
            FloatArray queryNorm,
            FloatArray compressedKv,
            FloatArray kvNorm,
            IntArray quants,
            FloatArray scales,
            IntArray sums,
            int queryRank,
            int rank,
            float eps) {
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        if (context.groupIdx == 0) {
            normInto(context, reduce, queryLatent, queryLatent, queryNorm, queryRank, eps);
            quantizeInto(context, queryLatent, quants, scales, sums, queryRank);
        }
        if (context.groupIdx == 1) {
            normInto(context, reduce, compressedKv, compressedKv, kvNorm, rank, eps);
        }
    }

    /** One Q8_0 block's packed integer dot product with the matching activation block. */
    private static int blockDot(ByteArray w, int offset, IntArray xQuants, int quantBase) {
        int dot = 0;
        for (int g = 0; g < QK / 4; g++) {
            int o = offset + 2 + g * 4;
            int packed =
                    (w.getHalfFloat(o).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(o + 2).getHalfFloatValue() & 0xFFFF) << 16);
            dot =
                    uk.ac.manchester.tornado.api.utils.QuantizationUtils.dp4a_packed(
                            packed, xQuants.get(quantBase + g), dot);
        }
        return dot;
    }

    private static float rowDot(
            ByteArray w, int row, int blocks, IntArray xQuants, FloatArray xScales, int lane) {
        float partial = 0.0f;
        for (int b = lane; b < blocks; b += 32) {
            int offset = (row * blocks + b) * BLOCK_BYTES;
            partial +=
                    w.getHalfFloat(offset).getFloat32()
                            * xScales.get(b)
                            * blockDot(w, offset, xQuants, b * (QK / 4));
        }
        return partial;
    }

    // @formatter:off
    /**
     * Two Q8_0 projections of one quantized activation in one launch, a warp per output row: {@code
     * out1 = w1 . x} for the first {@code d1} warps and {@code out2 = w2 . x} for the next {@code
     * d2}. Worker: {@code (d1 + d2) * 32} lanes in blocks of 128.
     */
    // @formatter:on
    public static void twoProjectionsQ8_0(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w1,
            FloatArray out1,
            ByteArray w2,
            FloatArray out2,
            int n,
            int d1,
            int d2) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int blocks = n / QK;
        if (warp < d1) {
            float sum = warpSum(context, rowDot(w1, warp, blocks, xQuants, xScales, lane));
            if (lane == 0) {
                out1.set(warp, sum);
            }
        }
        if (warp >= d1 && warp < d1 + d2) {
            int row = warp - d1;
            float sum = warpSum(context, rowDot(w2, row, blocks, xQuants, xScales, lane));
            if (lane == 0) {
                out2.set(row, sum);
            }
        }
    }

    // @formatter:off
    /**
     * {@link #routerLogits} and {@link #routerTopKSigmoid} in one workgroup of {@link #GROUP}
     * lanes: every warp scores its share of the experts into shared memory, then the first warp
     * selects. Requires at most 256 experts.
     */
    // @formatter:on
    public static void routeSigmoid(
            KernelContext context,
            FloatArray x,
            FloatArray router,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            FloatArray sharedGate,
            int dim,
            int experts,
            int used,
            int normalize,
            float scale) {
        int tid = context.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        float[] probs = context.allocateFloatLocalArray(256);
        float[] select = context.allocateFloatLocalArray(256);
        for (int e = warp; e < experts; e += GROUP / 32) {
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += router.get(e * dim + i) * x.get(i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                float p = 1.0f / (1.0f + TornadoMath.exp(-sum));
                probs[e] = p;
                select[e] = p + bias.get(e);
            }
        }
        context.localBarrier();
        if (warp == 0) {
            float total = 0.0f;
            float mine = 0.0f;
            for (int k = 0; k < used; k++) {
                float best = Float.NEGATIVE_INFINITY;
                float bestIndex = 1.0e9f;
                for (int e = lane; e < experts; e += 32) {
                    float sc = select[e];
                    if (sc > best) {
                        best = sc;
                        bestIndex = e;
                    }
                }
                float max = best;
                max = TornadoMath.max(max, context.simdShuffleDown(max, 16));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 8));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 4));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 2));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 1));
                max = context.simdBroadcastFirst(max);
                float index = best == max ? bestIndex : 1.0e9f;
                index = TornadoMath.min(index, context.simdShuffleDown(index, 16));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 8));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 4));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 2));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 1));
                int chosen = (int) context.simdBroadcastFirst(index);
                float w = probs[chosen];
                total += w;
                if (lane == k) {
                    mine = w;
                }
                if ((chosen & 31) == lane) {
                    select[chosen] = Float.NEGATIVE_INFINITY;
                }
                if (lane == 0) {
                    ids.set(k, chosen);
                }
            }
            float nf = normalize;
            float denominator = nf * TornadoMath.max(total, 6.103515625e-5f) + (1.0f - nf);
            if (lane < used) {
                weights.set(lane, mine / denominator * scale);
            }
            if (lane == 0) {
                sharedGate.set(0, 1.0f);
            }
        }
    }
}
