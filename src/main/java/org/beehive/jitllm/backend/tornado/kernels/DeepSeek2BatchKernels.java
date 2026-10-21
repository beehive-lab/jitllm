package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@code deepseek2} batched-prefill kernels: a chunk of prompt tokens through absorbed latent
 * attention as tensor-core GEMMs.
 *
 * <p>Per layer, after the projections: the rotation and the chunk's cache rows; every head's query
 * absorbed through {@code attn_k_b} (one strided GEMM over the heads); the scores of every (token,
 * head) row against every cached key (one GEMM, the keys in FP16); a causal softmax; the attended
 * latents (one GEMM against the latents transposed); and the latents pushed out through {@code
 * attn_v_b} (one strided GEMM over the heads).
 *
 * <p>{@code batchInfo} is the chunk's {@code [start position, active rows, slot]}; rows past the
 * active ones are padding the GEMMs compute and nothing else reads. The GEMMs and the softmax are
 * in {@link BatchMmaKernels}.
 */
// @formatter:on
public final class DeepSeek2BatchKernels {

    /** Lanes of a row-wise workgroup. */
    public static final int GROUP = 256;

    private DeepSeek2BatchKernels() {}

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
     * {@code out[t][i] = weight[i] * in[t][i] / rms(in[t][0..n))} for the active rows, a workgroup
     * of {@link #GROUP} lanes per row; rows are {@code inStride} and {@code outStride} apart.
     */
    public static void rmsNormRows(
            KernelContext context,
            FloatArray in,
            FloatArray out,
            FloatArray weight,
            IntArray batchInfo,
            int n,
            int inStride,
            int outStride,
            float eps) {
        int row = context.groupIdx;
        int tid = context.localIdx;
        float[] reduce = context.allocateFloatLocalArray(GROUP);
        if (row < batchInfo.get(1)) {
            float partial = 0.0f;
            for (int i = tid; i < n; i += GROUP) {
                float v = in.get(row * inStride + i);
                partial += v * v;
            }
            reduce[tid] = partial;
        }
        context.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        if (row < batchInfo.get(1)) {
            float scale = 1.0f / TornadoMath.sqrt(reduce[0] / n + eps);
            for (int i = tid; i < n; i += GROUP) {
                out.set(row * outStride + i, weight.get(i) * (in.get(row * inStride + i) * scale));
            }
        }
    }

    // @formatter:off
    /**
     * {@link DeepSeek2Kernels#ropeAndCache} for every active row of the chunk, at position {@code
     * start + row}; the queries are {@code queryStride} apart and the compressed key/value rows
     * {@code kvStride} apart. Worker: {@code rows * (heads * rope / 2 + rope / 2 + rank)} lanes.
     */
    // @formatter:on
    public static void ropeAndCacheRows(
            KernelContext context,
            IntArray batchInfo,
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
            int queryStride,
            int kvStride,
            int cacheOffset) {
        int pairs = rope / 2;
        int perRow = heads * pairs + pairs + rank;
        int idx = context.globalIdx;
        int t = idx / perRow;
        if (t < batchInfo.get(1)) {
            int j = idx - t * perRow;
            int pos = batchInfo.get(0) + t;
            int queryPairs = heads * pairs;
            int keyWidth = rank + rope;
            int row = cacheOffset + pos * keyWidth;
            int kv = t * kvStride;
            if (j < queryPairs) {
                int h = j / pairs;
                int i = j - h * pairs;
                int base = t * queryStride + h * queryHead + nope + 2 * i;
                float fcr = freqReal.get(pos * pairs + i);
                float fci = freqImag.get(pos * pairs + i);
                float v0 = q.get(base);
                float v1 = q.get(base + 1);
                q.set(base, v0 * fcr - v1 * fci);
                q.set(base + 1, v0 * fci + v1 * fcr);
            }
            if (j >= queryPairs && j < queryPairs + pairs) {
                int i = j - queryPairs;
                float fcr = freqReal.get(pos * pairs + i);
                float fci = freqImag.get(pos * pairs + i);
                float v0 = compressedKv.get(kv + rank + 2 * i);
                float v1 = compressedKv.get(kv + rank + 2 * i + 1);
                keyCache.set(row + rank + 2 * i, v0 * fcr - v1 * fci);
                keyCache.set(row + rank + 2 * i + 1, v0 * fci + v1 * fcr);
            }
            if (j >= queryPairs + pairs) {
                int r = j - queryPairs - pairs;
                keyCache.set(row + r, compressedKv.get(kv + r));
            }
        }
    }

    /** {@link #ropeAndCacheRows} over the half-precision cache. */
    public static void ropeAndCacheRowsFP16(
            KernelContext context,
            IntArray batchInfo,
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
            int queryStride,
            int kvStride,
            int cacheOffset) {
        int pairs = rope / 2;
        int perRow = heads * pairs + pairs + rank;
        int idx = context.globalIdx;
        int t = idx / perRow;
        if (t < batchInfo.get(1)) {
            int j = idx - t * perRow;
            int pos = batchInfo.get(0) + t;
            int queryPairs = heads * pairs;
            int keyWidth = rank + rope;
            int row = cacheOffset + pos * keyWidth;
            int kv = t * kvStride;
            if (j < queryPairs) {
                int h = j / pairs;
                int i = j - h * pairs;
                int base = t * queryStride + h * queryHead + nope + 2 * i;
                float fcr = freqReal.get(pos * pairs + i);
                float fci = freqImag.get(pos * pairs + i);
                float v0 = q.get(base);
                float v1 = q.get(base + 1);
                q.set(base, v0 * fcr - v1 * fci);
                q.set(base + 1, v0 * fci + v1 * fcr);
            }
            if (j >= queryPairs && j < queryPairs + pairs) {
                int i = j - queryPairs;
                float fcr = freqReal.get(pos * pairs + i);
                float fci = freqImag.get(pos * pairs + i);
                float v0 = compressedKv.get(kv + rank + 2 * i);
                float v1 = compressedKv.get(kv + rank + 2 * i + 1);
                keyCache.set(row + rank + 2 * i, new HalfFloat(v0 * fcr - v1 * fci));
                keyCache.set(row + rank + 2 * i + 1, new HalfFloat(v0 * fci + v1 * fcr));
            }
            if (j >= queryPairs + pairs) {
                int r = j - queryPairs - pairs;
                keyCache.set(row + r, new HalfFloat(compressedKv.get(kv + r)));
            }
        }
    }

    // @formatter:off
    /**
     * Every (token, head) row's attention query in FP16: the absorbed latent part, then the rotated
     * part from the projected queries. Rows are {@code token * heads + head}, {@code rank + rope}
     * wide; the absorbed part is read from {@code absorbed} at the same layout. Worker: {@code
     * paddedRows * heads * (rank + rope)} lanes.
     */
    // @formatter:on
    public static void assembleQueries(
            KernelContext context,
            FloatArray absorbed,
            FloatArray q,
            HalfFloatArray out,
            int heads,
            int rank,
            int rope,
            int nope,
            int queryHead,
            int queryStride,
            int total) {
        int idx = context.globalIdx;
        if (idx < total) {
            int keyWidth = rank + rope;
            int row = idx / keyWidth;
            int c = idx - row * keyWidth;
            if (c < rank) {
                out.set(idx, new HalfFloat(absorbed.get(idx)));
            }
            if (c >= rank) {
                int t = row / heads;
                int h = row - t * heads;
                out.set(
                        idx,
                        new HalfFloat(q.get(t * queryStride + h * queryHead + nope + c - rank)));
            }
        }
    }

    // @formatter:off
    /**
     * The layer's cache rows up to the chunk's last position, in FP16 for the score GEMM ({@code
     * keys[t][c]}) and transposed for the value GEMM ({@code latentT[c][t]}, the latent half only);
     * every position past them is zero, so the GEMMs over the padded width read nothing stale.
     * Worker: {@code paddedPositions * (rank + rope)} lanes.
     */
    // @formatter:on
    public static void cacheToHalf(
            KernelContext context,
            IntArray batchInfo,
            FloatArray keyCache,
            HalfFloatArray keys,
            HalfFloatArray latentT,
            int cacheOffset,
            int rank,
            int rope,
            int paddedPositions) {
        int keyWidth = rank + rope;
        int idx = context.globalIdx;
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * keyWidth) {
            int t = idx / keyWidth;
            int c = idx - t * keyWidth;
            int valid = batchInfo.get(0) + batchInfo.get(1);
            if (t < valid) {
                float v = keyCache.get(cacheOffset + idx);
                keys.set(idx, new HalfFloat(v));
                if (c < rank) {
                    latentT.set(c * paddedPositions + t, new HalfFloat(v));
                }
            }
            if (t >= valid) {
                keys.set(idx, new HalfFloat(0.0f));
                if (c < rank) {
                    latentT.set(c * paddedPositions + t, new HalfFloat(0.0f));
                }
            }
        }
    }

    /** {@link #cacheToHalf} over the half-precision cache. */
    public static void cacheToHalfFP16(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray keys,
            HalfFloatArray latentT,
            int cacheOffset,
            int rank,
            int rope,
            int paddedPositions) {
        int keyWidth = rank + rope;
        int idx = context.globalIdx;
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * keyWidth) {
            int t = idx / keyWidth;
            int c = idx - t * keyWidth;
            int valid = batchInfo.get(0) + batchInfo.get(1);
            if (t < valid) {
                float v = keyCache.get(cacheOffset + idx).getFloat32();
                keys.set(idx, new HalfFloat(v));
                if (c < rank) {
                    latentT.set(c * paddedPositions + t, new HalfFloat(v));
                }
            }
            if (t >= valid) {
                keys.set(idx, new HalfFloat(0.0f));
                if (c < rank) {
                    latentT.set(c * paddedPositions + t, new HalfFloat(0.0f));
                }
            }
        }
    }

    // @formatter:off
    /**
     * llama.cpp's sigmoid routing for every active token, a 32-lane block each: as {@link
     * DeepSeek2Kernels#routerTopKSigmoid}, the ids and weights at {@code token * used}, and the
     * token's shared-expert weight set to one. Worker: {@code batch * 32} lanes, local 32.
     */
    // @formatter:on
    public static void routerTopKSigmoidBatch(
            KernelContext context,
            FloatArray logits,
            FloatArray bias,
            IntArray ids,
            FloatArray weights,
            FloatArray sharedGate,
            IntArray batchInfo,
            int experts,
            int used,
            int normalize,
            float scale) {
        int token = context.groupIdx;
        int lane = context.localIdx;
        float[] probs = context.allocateFloatLocalArray(256);
        float[] select = context.allocateFloatLocalArray(256);
        boolean active = token < batchInfo.get(1);
        if (active) {
            for (int e = lane; e < experts; e += 32) {
                float p = 1.0f / (1.0f + TornadoMath.exp(-logits.get(token * experts + e)));
                probs[e] = p;
                select[e] = p + bias.get(e);
            }
        }
        context.localBarrier();
        if (active) {
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
                float w = probs[chosen];
                total += w;
                if (lane == k) {
                    mine = w;
                }
                if ((chosen & 31) == lane) {
                    select[chosen] = Float.NEGATIVE_INFINITY;
                }
                if (lane == 0) {
                    ids.set(token * used + k, chosen);
                }
            }
            float n = normalize;
            float denominator = n * TornadoMath.max(total, 6.103515625e-5f) + (1.0f - n);
            if (lane < used) {
                weights.set(token * used + lane, mine / denominator * scale);
            }
            if (lane == 0) {
                sharedGate.set(token, 1.0f);
            }
        }
    }
}
