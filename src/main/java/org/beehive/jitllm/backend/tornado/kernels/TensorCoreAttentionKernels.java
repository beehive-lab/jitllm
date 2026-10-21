package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The layout work around a prefill attention computed as tensor-core GEMMs: queries in FP16, and
 * the paged FP16 key/value cache gathered into contiguous per-key/value-head tiles, the values
 * transposed. The scores, the causal softmax and the weighted values are {@link
 * BatchMmaKernels#gemmMMAStrided} and {@link BatchMmaKernels#causalSoftmax}.
 *
 * <p>{@code batchInfo} is the chunk's {@code [start position, active rows, slot]}.
 */
// @formatter:on
public final class TensorCoreAttentionKernels {

    private TensorCoreAttentionKernels() {}

    /**
     * {@code out[t][c] = qkv[t * qkvStride + c]} in FP16 for the {@code dim} query columns of every
     * padded row. Worker: {@code rows * dim} lanes.
     */
    public static void queriesToHalf(
            KernelContext context,
            FloatArray qkv,
            HalfFloatArray out,
            int dim,
            int qkvStride,
            int total) {
        int i = context.globalIdx;
        if (i < total) {
            int t = i / dim;
            int c = i - t * dim;
            out.set(i, new HalfFloat(qkv.get(t * qkvStride + c)));
        }
    }

    // @formatter:off
    /**
     * The layer's paged FP16 cache up to the chunk's last position, per key/value head {@code g}:
     * {@code keys[g][t][d]} and {@code valuesT[g][d][t]}, {@code paddedPositions} positions each.
     * Positions past the chunk's last one, up to the next multiple of 128, are written as zero;
     * nothing past that is read. Worker: {@code paddedPositions * kvDim} lanes.
     */
    // @formatter:on
    public static void gatherPagedKeyValues(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keys,
            HalfFloatArray valuesT,
            int layer,
            int kvDim,
            int headSize,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int kIndex = (g * paddedPositions + t) * headSize + d;
            int vIndex = (g * headSize + d) * paddedPositions + t;
            if (t < valid) {
                int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
                int off =
                        KvBlockAddress.offset(
                                        blockTable,
                                        batchInfo.get(2),
                                        t,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + c;
                keys.set(kIndex, keyCache.get(off));
                valuesT.set(vIndex, valueCache.get(off));
            }
            if (t >= valid) {
                keys.set(kIndex, new HalfFloat(0.0f));
                valuesT.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }

    // @formatter:off
    /**
     * {@link #gatherPagedKeyValues} for a contiguous FP16 cache: the layer's rows start at {@code
     * cacheOffset}, one {@code kvDim}-wide row per position.
     */
    // @formatter:on
    public static void gatherKeyValues(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keys,
            HalfFloatArray valuesT,
            int cacheOffset,
            int kvDim,
            int headSize,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int kIndex = (g * paddedPositions + t) * headSize + d;
            int vIndex = (g * headSize + d) * paddedPositions + t;
            if (t < valid) {
                keys.set(kIndex, keyCache.get(cacheOffset + idx));
                valuesT.set(vIndex, valueCache.get(cacheOffset + idx));
            }
            if (t >= valid) {
                keys.set(kIndex, new HalfFloat(0.0f));
                valuesT.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }

    /** Lanes of a softmax workgroup: one workgroup per (token, head) row. */
    public static final int SOFTMAX_LANES = 256;

    // @formatter:off
    /**
     * The causal, sliding-window softmax of every (token, head) row of scores, in FP16: row {@code
     * token * heads + head} attends to positions {@code max(0, p - window + 1) .. p}, {@code p =
     * start + token}, scaled; every other entry up to the chunk's last position, and every row past
     * the active ones, is zero. A workgroup of {@link #SOFTMAX_LANES} lanes per row.
     */
    // @formatter:on
    public static void windowedSoftmax(
            KernelContext context,
            IntArray batchInfo,
            FloatArray scores,
            HalfFloatArray probs,
            int heads,
            int paddedPositions,
            float scale,
            int window) {
        int row = context.groupIdx;
        int tid = context.localIdx;
        int t = row / heads;
        int base = row * paddedPositions;
        float[] reduce = context.allocateFloatLocalArray(SOFTMAX_LANES);
        // A padding row attends to nothing, without a branch whose merged value is miscompiled.
        int active = Math.min(1, Math.max(0, batchInfo.get(1) - t));
        int pos = batchInfo.get(0) + t;
        int end = active * (pos + 1);
        int lo = Math.max(0, pos - window + 1);
        float localMax = Float.NEGATIVE_INFINITY;
        for (int j = lo + tid; j < end; j += SOFTMAX_LANES) {
            localMax = TornadoMath.max(localMax, scores.get(base + j) * scale);
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = SOFTMAX_LANES / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
            }
            context.localBarrier();
        }
        float max = reduce[0];
        context.localBarrier();
        float localSum = 0.0f;
        for (int j = lo + tid; j < end; j += SOFTMAX_LANES) {
            localSum += TornadoMath.exp(scores.get(base + j) * scale - max);
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = SOFTMAX_LANES / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float inverse = 1.0f / TornadoMath.max(reduce[0], 1.0e-30f);
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        int width = Math.min(paddedPositions, filled);
        for (int j = tid; j < width; j += SOFTMAX_LANES) {
            if (j >= lo && j < end) {
                probs.set(
                        base + j,
                        new HalfFloat(
                                TornadoMath.exp(scores.get(base + j) * scale - max) * inverse));
            }
            if (j < lo || j >= end) {
                probs.set(base + j, new HalfFloat(0.0f));
            }
        }
    }
}
