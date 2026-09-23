package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Granite KV kernels that address the cache through a block table.
 *
 * <p>The {@code Paged} twins of the KV kernels in {@link GraniteKernels}, differing from them in
 * the KV index and nothing else. Review by diffing each against its original.
 *
 * <p>Granite carries an attention scale its siblings do not, which is why it has its own copies of
 * attention and RoPE in the first place; that scale is untouched here. Addressing is defined once,
 * in {@link KvBlockAddress}.
 */
public class GranitePagedKvKernels {

    public GranitePagedKvKernels() {}

    public static void processHeadsFlashAttentionWithGraniteScalePaged(
            KernelContext context,
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            float attentionScale) {

        // Thread and workgroup information
        int tid = context.localIdx;
        int h = context.groupIdx; // Each workgroup processes one head
        int localSize = context.localGroupSizeX;

        // Early exit if this workgroup is beyond our head count
        // This relies on the kernel being launched with nHeads workgroups.
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        int BLOCK_SIZE_C = 16;

        // Allocate shared memory for tiled computation
        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_tile_max_holder =
                context.allocateFloatLocalArray(1); // FIX: For broadcasting tile max

        // Thread-local accumulators for online softmax
        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        // Thread-local output accumulation
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        // Load query vector into shared memory
        for (int i = tid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }

        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = 0; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Load key and value vectors for this tile
            // Each thread loads a portion of the K and V vectors for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile
                int tileMemOffset = k_v_idx_in_tile * headSize;
                int kvBase =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        tIdxInSeq,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHeadIdx * headSize;
                for (int d = 0; d < headSize; d++) {
                    k_tile[tileMemOffset + d] = key_cache.get(kvBase + d);
                    v_tile[tileMemOffset + d] = value_cache.get(kvBase + d);
                }
            }

            context.localBarrier();

            // Compute attention scores for this tile
            // Each thread computes one score for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                score *= attentionScale;
                s_tile[score_idx_in_tile] = score;
            }

            context.localBarrier();

            // Find max score in this tile (all threads compute it redundantly over the small
            // s_tile)
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) { // Iterate over valid scores in s_tile
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            // Broadcast max to all threads via shared memory
            if (tid == 0) {
                shared_tile_max_holder[0] = tileLocalMax; // FIX: Use dedicated holder
            }
            context.localBarrier();
            float currentTileMax = shared_tile_max_holder[0]; // FIX: Read from dedicated holder

            // Determine if we need to rescale previous results
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
            }
            maxScore = newMax;

            // Process each key-value pair using original scores from s_tile
            // All threads iterate over all scores in the current tile
            for (int t_idx_in_s_tile = 0; t_idx_in_s_tile <= tileEnd - tileC; t_idx_in_s_tile++) {
                // s_tile[t_idx_in_s_tile] now correctly refers to the original score
                float expScore = TornadoMath.exp(s_tile[t_idx_in_s_tile] - maxScore);
                sumExp += expScore;

                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * v_tile[t_idx_in_s_tile * headSize + d];
                }
            }
            context.localBarrier(); // Ensure all threads finish with s_tile, k_tile, v_tile before
            // next tile load
        }

        // Normalize and write final results
        float normFactor =
                (sumExp > 0.0f)
                        ? (1.0f / sumExp)
                        : 0.0f; // Avoid division by zero, return 0 if sumExp is 0
        for (int d = tid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output[d] * normFactor);
        }
    }

    // FP16-cache twin of processHeadsFlashAttentionWithGraniteScalePaged: identical arithmetic, the
    // cache stored in binary16.
    public static void processHeadsFlashAttentionWithGraniteScaleFP16Paged(
            KernelContext context,
            FloatArray q,
            HalfFloatArray key_cache,
            HalfFloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            float attentionScale) {

        // Thread and workgroup information
        int tid = context.localIdx;
        int h = context.groupIdx; // Each workgroup processes one head
        int localSize = context.localGroupSizeX;

        // Early exit if this workgroup is beyond our head count
        // This relies on the kernel being launched with nHeads workgroups.
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        int BLOCK_SIZE_C = 16;

        // Allocate shared memory for tiled computation
        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_tile_max_holder =
                context.allocateFloatLocalArray(1); // FIX: For broadcasting tile max

        // Thread-local accumulators for online softmax
        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        // Thread-local output accumulation
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        // Load query vector into shared memory
        for (int i = tid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }

        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = 0; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Load key and value vectors for this tile
            // Each thread loads a portion of the K and V vectors for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile
                int tileMemOffset = k_v_idx_in_tile * headSize;
                int kvBase =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        tIdxInSeq,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHeadIdx * headSize;
                for (int d = 0; d < headSize; d++) {
                    k_tile[tileMemOffset + d] = key_cache.get(kvBase + d).getFloat32();
                    v_tile[tileMemOffset + d] = value_cache.get(kvBase + d).getFloat32();
                }
            }

            context.localBarrier();

            // Compute attention scores for this tile
            // Each thread computes one score for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                score *= attentionScale;
                s_tile[score_idx_in_tile] = score;
            }

            context.localBarrier();

            // Find max score in this tile (all threads compute it redundantly over the small
            // s_tile)
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) { // Iterate over valid scores in s_tile
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            // Broadcast max to all threads via shared memory
            if (tid == 0) {
                shared_tile_max_holder[0] = tileLocalMax; // FIX: Use dedicated holder
            }
            context.localBarrier();
            float currentTileMax = shared_tile_max_holder[0]; // FIX: Read from dedicated holder

            // Determine if we need to rescale previous results
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
            }
            maxScore = newMax;

            // Process each key-value pair using original scores from s_tile
            // All threads iterate over all scores in the current tile
            for (int t_idx_in_s_tile = 0; t_idx_in_s_tile <= tileEnd - tileC; t_idx_in_s_tile++) {
                // s_tile[t_idx_in_s_tile] now correctly refers to the original score
                float expScore = TornadoMath.exp(s_tile[t_idx_in_s_tile] - maxScore);
                sumExp += expScore;

                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * v_tile[t_idx_in_s_tile * headSize + d];
                }
            }
            context.localBarrier(); // Ensure all threads finish with s_tile, k_tile, v_tile before
            // next tile load
        }

        // Normalize and write final results
        float normFactor =
                (sumExp > 0.0f)
                        ? (1.0f / sumExp)
                        : 0.0f; // Avoid division by zero, return 0 if sumExp is 0
        for (int d = tid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output[d] * normFactor);
        }
    }

    public static void processHeadsParallelGranitePaged(
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int seqLen,
            IntArray positionHolder,
            FloatArray wrapAtt,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            float attentionScale) {

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            processHeadTornadoPaged(
                    q,
                    key_cache,
                    value_cache,
                    xb,
                    h,
                    headSize,
                    kvDim,
                    kvMul,
                    blockTable,
                    slot,
                    layerOff,
                    blockCfg,
                    blockStride,
                    pos,
                    wrapAtt,
                    attentionScale);
        }
    }

    public static void ropeRotationWithCacheCopyPaged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray sq, // Q vector (in/out)
            FloatArray sk, // K vector (in/out)
            FloatArray sv, // V vector (in only)
            FloatArray keyCache, // Key cache (out)
            FloatArray valueCache, // Value cache (out)
            int kvDim,
            int headSize,
            float ropeTheta,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int i = context.globalIdx * 2;
        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);

        // Bounds check for Q rotation (Q has dim elements, processed in pairs)
        if (i + 1 < sq.getSize()) {
            // RoPE frequency calculation
            int head_dim = i % headSize;
            //            TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            // Rotate Q
            float v0q = sq.get(i);
            float v1q = sq.get(i + 1);
            sq.set(i, v0q * fcr - v1q * fci);
            sq.set(i + 1, v0q * fci + v1q * fcr);

            // Rotate K AND write to cache (only for kvDim elements)
            if (i + 1 < kvDim) {
                float v0k = sk.get(i);
                float v1k = sk.get(i + 1);
                float rotated0 = v0k * fcr - v1k * fci;
                float rotated1 = v0k * fci + v1k * fcr;

                // Write rotated K back to sk
                sk.set(i, rotated0);
                sk.set(i + 1, rotated1);

                // Direct cache write (fused - no separate copy kernel!)
                int cacheOffset =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(cacheOffset + i, rotated0);
                keyCache.set(cacheOffset + i + 1, rotated1);

                // Copy V to cache (V doesn't need rotation)
                valueCache.set(cacheOffset + i, sv.get(i));
                valueCache.set(cacheOffset + i + 1, sv.get(i + 1));
            }
        }
    }

    // FP16-cache twin of ropeRotationWithCacheCopyPaged: identical arithmetic, the cache stored in
    // binary16.
    public static void ropeRotationWithCacheCopyFP16Paged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray sq, // Q vector (in/out)
            FloatArray sk, // K vector (in/out)
            FloatArray sv, // V vector (in only)
            HalfFloatArray keyCache, // Key cache (out)
            HalfFloatArray valueCache, // Value cache (out)
            int kvDim,
            int headSize,
            float ropeTheta,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int i = context.globalIdx * 2;
        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);

        // Bounds check for Q rotation (Q has dim elements, processed in pairs)
        if (i + 1 < sq.getSize()) {
            // RoPE frequency calculation
            int head_dim = i % headSize;
            //            TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            // Rotate Q
            float v0q = sq.get(i);
            float v1q = sq.get(i + 1);
            sq.set(i, v0q * fcr - v1q * fci);
            sq.set(i + 1, v0q * fci + v1q * fcr);

            // Rotate K AND write to cache (only for kvDim elements)
            if (i + 1 < kvDim) {
                float v0k = sk.get(i);
                float v1k = sk.get(i + 1);
                float rotated0 = v0k * fcr - v1k * fci;
                float rotated1 = v0k * fci + v1k * fcr;

                // Write rotated K back to sk
                sk.set(i, rotated0);
                sk.set(i + 1, rotated1);

                // Direct cache write (fused - no separate copy kernel!)
                int cacheOffset =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(cacheOffset + i, new HalfFloat(rotated0));
                keyCache.set(cacheOffset + i + 1, new HalfFloat(rotated1));

                // Copy V to cache (V doesn't need rotation)
                valueCache.set(cacheOffset + i, new HalfFloat(sv.get(i)));
                valueCache.set(cacheOffset + i + 1, new HalfFloat(sv.get(i + 1)));
            }
        }
    }

    private static void processHeadTornadoPaged(
            FloatArray allQ,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray allXb,
            int h,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int slot,
            int layerOff,
            int pos,
            FloatArray wrapAtt,
            float attentionScale) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / kvMul;
            int keyOffset =
                    KvBlockAddress.offset(
                                    blockTable, slot, t, layerOff, kvDim, blockCfg, blockStride)
                            + (kvHeadIdx * headSize);

            float score = 0.0f;
            for (int i = 0; i < headSize; i++) {
                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
            }
            score *= attentionScale; // TODO: might need score = score * attentionScale;
            //            score = score / TornadoMath.sqrt(headSize);

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
        //
        // The loop nest is t-outer here where the legacy kernel had i-outer, and the accumulator
        // array is what makes that safe: element i still sums over t in ascending order, so the
        // floating-point result is identical. The reason to swap is the block walk — it depends on
        // t and not on i, and i-outer would repeat it headSize times per position. That cost is
        // what made Mistral -3.8% while every other family sat under 1%.
        int kvHeadIdx = h / kvMul;
        float[] weightedSums = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            weightedSums[i] = 0.0f;
        }
        for (int t = 0; t <= pos; t++) {
            int valueOffset =
                    KvBlockAddress.offset(
                                    blockTable, slot, t, layerOff, kvDim, blockCfg, blockStride)
                            + kvHeadIdx * headSize;
            float attWeight = wrapAtt.get(headOffset + t);
            for (int i = 0; i < headSize; i++) {
                weightedSums[i] += attWeight * value_cache.get(valueOffset + i);
            }
        }
        for (int i = 0; i < headSize; i++) {
            allXb.set(h * headSize + i, weightedSums[i]);
        }
    }
}
