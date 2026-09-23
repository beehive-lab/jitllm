package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Qwen2 and Qwen3 KV kernels that address the cache through a block table.
 *
 * <p>The {@code Paged} twins of the KV kernels in {@link Qwen3Kernels}, differing from them in the
 * KV index and nothing else. Review by diffing each against its original.
 *
 * <p>Addressing is defined once, in {@link KvBlockAddress}.
 */
public class Qwen3PagedKvKernels {

    public Qwen3PagedKvKernels() {}

    public static void processHeadsParallelPaged(
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int nEmbdHead, /* = nEmbdHead, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int
                    nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            IntArray positionHolder,
            FloatArray wrapAtt,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, nEmbdGqa, blockCfg);

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            //noinspection ExternalInspection
            processHeadTornadoPaged(
                    q,
                    key_cache,
                    value_cache,
                    xb,
                    h,
                    nEmbdHead, /* headSize */
                    nEmbdHeadK, /* headSize in line 255 */
                    nEmbdHeadV, /* headSize in lines: 266, 268, 273 */
                    nEmbdGqa, /* kvDim */
                    gqa, /* kvMul */
                    blockTable,
                    slot,
                    layerOff,
                    blockCfg,
                    blockStride,
                    pos,
                    wrapAtt);
        }
    }

    private static void processHeadTornadoPaged(
            FloatArray allQ,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray allXb,
            int h,
            int nEmbdHead, /* = nEmbdHeadV, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int
                    nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int slot,
            int layerOff,
            int pos,
            FloatArray wrapAtt) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / gqa;
            int keyOffset =
                    KvBlockAddress.offset(
                                    blockTable, slot, t, layerOff, nEmbdGqa, blockCfg, blockStride)
                            + (kvHeadIdx * nEmbdHeadK); // line 255

            float score = 0.0f;
            for (int i = 0; i < nEmbdHeadK; i++) {
                score += allQ.get(h * nEmbdHeadK + i) * key_cache.get(keyOffset + i); // line 255
            }
            score = score / TornadoMath.sqrt(nEmbdHead); // line 257

            // Store in attention buffer
            wrapAtt.set(headOffset + t, score);
        }

        // STEP 2: Find max score for softmax stability
        float maxScore = wrapAtt.get(headOffset);
        for (int t = 1; t <= pos; t++) {
            float val = wrapAtt.get(headOffset + t);
            if (val > maxScore) {
                maxScore = val;
            }
        }

        // STEP 3: Compute exponentials and sum
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            float expScore = TornadoMath.exp(wrapAtt.get(idx) - maxScore);
            wrapAtt.set(idx, expScore);
            sum += expScore;
        }

        // STEP 4: Normalize
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos + 1));
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            wrapAtt.set(idx, wrapAtt.get(idx) * normFactor);
        }

        // STEP 5: Compute weighted sum of values for each dimension
        for (int i = 0; i < nEmbdHeadV; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int kvHeadIdx = h / gqa;
                int valueOffset =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        t,
                                        layerOff,
                                        nEmbdGqa,
                                        blockCfg,
                                        blockStride)
                                + (kvHeadIdx * nEmbdHeadV); // line 273
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            allXb.set(h * nEmbdHeadV + i, weightedSum); // offset from line 266
        }
    }

    public static void ropeRotationWithCacheCopyPaged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q, // Q vector (in/out)
            FloatArray k, // K vector (in/out)
            FloatArray v, // V vector (in only)
            FloatArray keyCache, // Key cache (out)
            FloatArray valueCache, // Value cache (out)
            float ropeTheta, // from the model metadata, NOT a constant
            int numberOfKeyValueHeads,
            int nEmbdHead,
            int nEmbdGqa,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int h = context.globalIdx;
        int ic = context.globalIdy;

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int rotn = h < numberOfKeyValueHeads ? 2 : 1;
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        // The base belongs to the model: Qwen2/Qwen3 checkpoints usually use 1000000, but a
        // Qwen2-architecture distill such as DeepSeek-R1-Distill-Qwen uses 10000, and a constant
        // here silently rotated it wrong.
        float theta = ropeTheta;
        int i = ic * 2;
        float freq = 1.0f / TornadoMath.pow(theta, (float) i / (float) nEmbdHead);

        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q (all heads)
        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + nComplEmbdHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + nComplEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K and copy K/V to cache (only for KV heads)
        if (rotn > 1 && (poffset + ic + nComplEmbdHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + nComplEmbdHead);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            // Write rotated K back
            k.set(poffset + ic, rotatedK0);
            k.set(poffset + ic + nComplEmbdHead, rotatedK1);

            // Direct cache write (fused - no separate copy kernel!)
            int cacheOffset =
                    KvBlockAddress.offset(
                            blockTable,
                            slot,
                            pos,
                            KvBlockAddress.layerOffset(layer, nEmbdGqa, blockCfg),
                            nEmbdGqa,
                            blockCfg,
                            blockStride);
            int kvIdx = h * nEmbdHead;

            keyCache.set(cacheOffset + kvIdx + ic, rotatedK0);
            keyCache.set(cacheOffset + kvIdx + ic + nComplEmbdHead, rotatedK1);

            // Copy V to cache (V doesn't need rotation)
            valueCache.set(cacheOffset + kvIdx + ic, v.get(poffset + ic));
            valueCache.set(
                    cacheOffset + kvIdx + ic + nComplEmbdHead,
                    v.get(poffset + ic + nComplEmbdHead));
        }
    }

    public static void ropeRotationWithCacheCopyFP16Paged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q, // Q vector (in/out)
            FloatArray k, // K vector (in/out)
            FloatArray v, // V vector (in only)
            HalfFloatArray keyCache, // Key cache (out)
            HalfFloatArray valueCache, // Value cache (out)
            float ropeTheta, // from the model metadata, NOT a constant
            int numberOfKeyValueHeads,
            int nEmbdHead,
            int nEmbdGqa,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int h = context.globalIdx;
        int ic = context.globalIdy;

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int rotn = h < numberOfKeyValueHeads ? 2 : 1;
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        float theta = ropeTheta;
        int i = ic * 2;
        float freq = 1.0f / TornadoMath.pow(theta, (float) i / (float) nEmbdHead);

        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q (all heads)
        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + nComplEmbdHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + nComplEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K and copy K/V to cache (only for KV heads)
        if (rotn > 1 && (poffset + ic + nComplEmbdHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + nComplEmbdHead);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            k.set(poffset + ic, rotatedK0);
            k.set(poffset + ic + nComplEmbdHead, rotatedK1);

            int cacheOffset =
                    KvBlockAddress.offset(
                            blockTable,
                            slot,
                            pos,
                            KvBlockAddress.layerOffset(layer, nEmbdGqa, blockCfg),
                            nEmbdGqa,
                            blockCfg,
                            blockStride);
            int kvIdx = h * nEmbdHead;

            keyCache.set(cacheOffset + kvIdx + ic, new HalfFloat(rotatedK0));
            keyCache.set(cacheOffset + kvIdx + ic + nComplEmbdHead, new HalfFloat(rotatedK1));

            valueCache.set(cacheOffset + kvIdx + ic, new HalfFloat(v.get(poffset + ic)));
            valueCache.set(
                    cacheOffset + kvIdx + ic + nComplEmbdHead,
                    new HalfFloat(v.get(poffset + ic + nComplEmbdHead)));
        }
    }

    public static void batchedRopeWithKVCacheQwen3Paged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int qDim) {

        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;

        int pos = batchStartPosHolder.get(0) + batchIdx;
        int slot = batchStartPosHolder.get(2);

        // Qwen3 uses split-half RoPE: pair element ic with ic + nEmbdHead/2 within each head.
        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        // Base from model metadata, not a constant — see ropeRotationWithCacheCopy.
        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q (split-half pairs within each head)
        int qHeadBase = batchIdx * qDim + headIdx * nEmbdHead;
        float v0q = wrapQBatch.get(qHeadBase + ic);
        float v1q = wrapQBatch.get(qHeadBase + ic + halfEmbdHead);
        wrapQBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        wrapQBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K and write K,V to cache (only for KV pairs)
        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * kvDim + kHeadIdx * nEmbdHead;
            float v0k = wrapKBatch.get(kHeadBase + ic);
            float v1k = wrapKBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            wrapKBatch.set(kHeadBase + ic, rotK0);
            wrapKBatch.set(kHeadBase + ic + halfEmbdHead, rotK1);

            int cacheOff =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    pos,
                                    KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + kHeadIdx * nEmbdHead;
            wrapKeyCache.set(cacheOff + ic, rotK0);
            wrapKeyCache.set(cacheOff + ic + halfEmbdHead, rotK1);
            wrapValueCache.set(cacheOff + ic, wrapVBatch.get(kHeadBase + ic));
            wrapValueCache.set(
                    cacheOff + ic + halfEmbdHead, wrapVBatch.get(kHeadBase + ic + halfEmbdHead));
        }
    }

    // FP16-cache twin of batchedRopeWithKVCacheQwen3Paged: identical arithmetic, the cache stored
    // in binary16.
    public static void batchedRopeWithKVCacheQwen3FP16Paged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int qDim) {

        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;

        int pos = batchStartPosHolder.get(0) + batchIdx;
        int slot = batchStartPosHolder.get(2);

        // Qwen3 uses split-half RoPE: pair element ic with ic + nEmbdHead/2 within each head.
        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        // Base from model metadata, not a constant — see ropeRotationWithCacheCopy.
        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q (split-half pairs within each head)
        int qHeadBase = batchIdx * qDim + headIdx * nEmbdHead;
        float v0q = wrapQBatch.get(qHeadBase + ic);
        float v1q = wrapQBatch.get(qHeadBase + ic + halfEmbdHead);
        wrapQBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        wrapQBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K and write K,V to cache (only for KV pairs)
        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * kvDim + kHeadIdx * nEmbdHead;
            float v0k = wrapKBatch.get(kHeadBase + ic);
            float v1k = wrapKBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            wrapKBatch.set(kHeadBase + ic, rotK0);
            wrapKBatch.set(kHeadBase + ic + halfEmbdHead, rotK1);

            int cacheOff =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    pos,
                                    KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + kHeadIdx * nEmbdHead;
            wrapKeyCache.set(cacheOff + ic, new HalfFloat(rotK0));
            wrapKeyCache.set(cacheOff + ic + halfEmbdHead, new HalfFloat(rotK1));
            wrapValueCache.set(cacheOff + ic, new HalfFloat(wrapVBatch.get(kHeadBase + ic)));
            wrapValueCache.set(
                    cacheOff + ic + halfEmbdHead,
                    new HalfFloat(wrapVBatch.get(kHeadBase + ic + halfEmbdHead)));
        }
    }

    public static void batchedRopeWithKVCacheQwen3PackedPaged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray qkvBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int qDim) {

        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;
        int qkvStride = qDim + 2 * kvDim;

        // Padded rows are not tokens: see the FP16 variant below. No barriers here, so an
        // early return is safe per lane.
        if (batchIdx >= batchStartPosHolder.get(1)) {
            return;
        }

        int pos = batchStartPosHolder.get(0) + batchIdx;
        int slot = batchStartPosHolder.get(2);

        // Qwen3 uses split-half RoPE: pair element ic with ic + nEmbdHead/2 within each head.
        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        // Base from model metadata, not a constant — see ropeRotationWithCacheCopy.
        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q in place (packed offset 0)
        int qHeadBase = batchIdx * qkvStride + headIdx * nEmbdHead;
        float v0q = qkvBatch.get(qHeadBase + ic);
        float v1q = qkvBatch.get(qHeadBase + ic + halfEmbdHead);
        qkvBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        qkvBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K (packed offset qDim) and write K,V to cache
        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            int vHeadBase = batchIdx * qkvStride + qDim + kvDim + kHeadIdx * nEmbdHead;
            float v0k = qkvBatch.get(kHeadBase + ic);
            float v1k = qkvBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;

            int cacheOff =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    pos,
                                    KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + kHeadIdx * nEmbdHead;
            wrapKeyCache.set(cacheOff + ic, rotK0);
            wrapKeyCache.set(cacheOff + ic + halfEmbdHead, rotK1);
            wrapValueCache.set(cacheOff + ic, qkvBatch.get(vHeadBase + ic));
            wrapValueCache.set(
                    cacheOff + ic + halfEmbdHead, qkvBatch.get(vHeadBase + ic + halfEmbdHead));
        }
    }

    public static void batchedRopeWithKVCacheQwen3PackedFP16Paged(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray qkvBatch,
            HalfFloatArray wrapKeyCache,
            HalfFloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int qDim) {

        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;
        int qkvStride = qDim + 2 * kvDim;

        // A padded row is not a token and has no key/value to store. Writing one anyway
        // addresses the block table at pos/blockSize for a position the session never
        // allocated: when the context is smaller than the prefill width -- a short prompt in
        // a wide chunk -- that index is past the end of this slot's table, and the physical
        // block it reads back scatters the write over blocks that DO hold live keys. The
        // damage is deterministic, so a single engine stays self-consistent and the defect
        // hides; it only surfaces when two attention paths leave different values in the
        // padded rows. Measured on Qwen3-0.6B-FP16, prompt 300 in a 512-wide chunk:
        // capacity 320 < 512, and the two paths' logits diverged by relL2 ~1.0. Raising the
        // context so capacity >= 512 dropped that to 6.4e-04 with nothing else changed.
        if (batchIdx >= batchStartPosHolder.get(1)) {
            return;
        }

        int pos = batchStartPosHolder.get(0) + batchIdx;
        int slot = batchStartPosHolder.get(2);

        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        // Base from model metadata, not a constant — see ropeRotationWithCacheCopy.
        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q in place (packed offset 0)
        int qHeadBase = batchIdx * qkvStride + headIdx * nEmbdHead;
        float v0q = qkvBatch.get(qHeadBase + ic);
        float v1q = qkvBatch.get(qHeadBase + ic + halfEmbdHead);
        qkvBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        qkvBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K (packed offset qDim) and write K,V to the half-precision cache
        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            int vHeadBase = batchIdx * qkvStride + qDim + kvDim + kHeadIdx * nEmbdHead;
            float v0k = qkvBatch.get(kHeadBase + ic);
            float v1k = qkvBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;

            int cacheOff =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    pos,
                                    KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + kHeadIdx * nEmbdHead;
            wrapKeyCache.set(cacheOff + ic, new HalfFloat(rotK0));
            wrapKeyCache.set(cacheOff + ic + halfEmbdHead, new HalfFloat(rotK1));
            wrapValueCache.set(cacheOff + ic, new HalfFloat(qkvBatch.get(vHeadBase + ic)));
            wrapValueCache.set(
                    cacheOff + ic + halfEmbdHead,
                    new HalfFloat(qkvBatch.get(vHeadBase + ic + halfEmbdHead)));
        }
    }
}
