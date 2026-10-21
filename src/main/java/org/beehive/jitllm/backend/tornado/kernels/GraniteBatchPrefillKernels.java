package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Batched-prefill kernels specific to the Granite family. */
public final class GraniteBatchPrefillKernels {

    private GraniteBatchPrefillKernels() {}

    /** {@code x[i] *= scale} for the first {@code n} elements: Granite's embedding multiplier. */
    public static void batchedScale(KernelContext context, FloatArray x, float scale, int n) {
        int i = context.globalIdx;
        if (i < n) {
            x.set(i, x.get(i) * scale);
        }
    }

    /**
     * {@code TransformerBatchPrefillKernels.batchedFFNSwiGLUFP16Packed} with the product scaled by
     * {@code scale} before it is rounded to FP16. Granite's feed-forward activations peak above the
     * FP16 range (Granite 4.0 1B reaches 1.3e5 in its fifth layer), and an infinity here turns the
     * down projection, and every layer after it, into NaN. One thread per (token, hidden row).
     */
    public static void batchedFFNSwiGLUFP16PackedScaled(
            KernelContext context,
            HalfFloatArray wrapHbFP16Batch,
            FloatArray gateUpResult,
            int hiddenDim,
            float scale,
            int n) {
        int gid = context.globalIdx;
        if (gid < n) {
            int b = gid / hiddenDim;
            int i = gid % hiddenDim;
            int rowBase = b * 2 * hiddenDim;
            float g = gateUpResult.get(rowBase + i);
            float u = gateUpResult.get(rowBase + hiddenDim + i);
            float silu = g / (1.0f + TornadoMath.exp(-g));
            wrapHbFP16Batch.set(gid, new HalfFloat(silu * u * scale));
        }
    }

    /**
     * Scales the Q rows of the packed {@code [q|k|v]} rows the tensor-core QKV GEMM writes, by
     * {@code scale}: row {@code r < dim} of token {@code t} sits at {@code t * (dim + 2 * kvDim) +
     * r}. One thread per (token, Q row); threads past {@code batch} tokens do nothing.
     */
    public static void batchedPackedScaleQ(
            KernelContext context,
            FloatArray qkvBatch,
            float scale,
            int dim,
            int kvDim,
            int batch) {
        int index = context.globalIdx;
        if (index < batch * dim) {
            int token = index / dim;
            int row = index % dim;
            int o = token * (dim + 2 * kvDim) + row;
            qkvBatch.set(o, qkvBatch.get(o) * scale);
        }
    }

    /**
     * {@code GranitePagedKvKernels.ropeRotationWithCacheCopyPaged} for a batch: interleaved pairs
     * {@code (i, i + 1)}, the frequency computed from {@code ropeTheta} in the kernel, the rotated
     * K and the V of every active token written to the paged cache. One thread per (token, pair);
     * padding tokens past {@code batchStartPosHolder[1]} do nothing.
     */
    public static void batchedRopeWithKVCacheGranitePaged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int i = (globalIdx % halfDim) * 2;
        if (batchIdx >= batchStartPosHolder.get(1)) {
            return;
        }
        int pos = batchStartPosHolder.get(0) + batchIdx;
        int qOffset = batchIdx * dim;
        int kOffset = batchIdx * kvDim;
        int headDim = i % headSize;
        float freq = 1.0f / TornadoMath.pow(ropeTheta, headDim / (float) headSize);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        float v0q = wrapQBatch.get(qOffset + i);
        float v1q = wrapQBatch.get(qOffset + i + 1);
        wrapQBatch.set(qOffset + i, v0q * fcr - v1q * fci);
        wrapQBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

        if (i + 1 < kvDim) {
            float v0k = wrapKBatch.get(kOffset + i);
            float v1k = wrapKBatch.get(kOffset + i + 1);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            wrapKBatch.set(kOffset + i, rotK0);
            wrapKBatch.set(kOffset + i + 1, rotK1);
            int cacheOff =
                    KvBlockAddress.offset(
                            blockTable,
                            batchStartPosHolder.get(2),
                            pos,
                            KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                            kvDim,
                            blockCfg,
                            blockStride);
            wrapKeyCache.set(cacheOff + i, rotK0);
            wrapKeyCache.set(cacheOff + i + 1, rotK1);
            wrapValueCache.set(cacheOff + i, wrapVBatch.get(kOffset + i));
            wrapValueCache.set(cacheOff + i + 1, wrapVBatch.get(kOffset + i + 1));
        }
    }

    /** {@link #batchedRopeWithKVCacheGranitePaged} writing the FP16 cache. */
    public static void batchedRopeWithKVCacheGraniteFP16Paged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim) {
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int i = (globalIdx % halfDim) * 2;
        if (batchIdx >= batchStartPosHolder.get(1)) {
            return;
        }
        int pos = batchStartPosHolder.get(0) + batchIdx;
        int qOffset = batchIdx * dim;
        int kOffset = batchIdx * kvDim;
        int headDim = i % headSize;
        float freq = 1.0f / TornadoMath.pow(ropeTheta, headDim / (float) headSize);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        float v0q = wrapQBatch.get(qOffset + i);
        float v1q = wrapQBatch.get(qOffset + i + 1);
        wrapQBatch.set(qOffset + i, v0q * fcr - v1q * fci);
        wrapQBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

        if (i + 1 < kvDim) {
            float v0k = wrapKBatch.get(kOffset + i);
            float v1k = wrapKBatch.get(kOffset + i + 1);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            wrapKBatch.set(kOffset + i, rotK0);
            wrapKBatch.set(kOffset + i + 1, rotK1);
            int cacheOff =
                    KvBlockAddress.offset(
                            blockTable,
                            batchStartPosHolder.get(2),
                            pos,
                            KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                            kvDim,
                            blockCfg,
                            blockStride);
            wrapKeyCache.set(cacheOff + i, new HalfFloat(rotK0));
            wrapKeyCache.set(cacheOff + i + 1, new HalfFloat(rotK1));
            wrapValueCache.set(cacheOff + i, new HalfFloat(wrapVBatch.get(kOffset + i)));
            wrapValueCache.set(cacheOff + i + 1, new HalfFloat(wrapVBatch.get(kOffset + i + 1)));
        }
    }
}
