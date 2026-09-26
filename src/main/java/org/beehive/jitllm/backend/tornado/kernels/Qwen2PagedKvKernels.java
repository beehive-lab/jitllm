package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Qwen2 and Qwen2-MoE KV kernels that address the cache through a block table.
 *
 * <p>The {@code Paged} twins of {@link Qwen2Kernels#processHeadsFlashAttention} and {@link
 * Qwen2MoEBatchKernels#batchedRopeWithKVCacheQwen2}, differing from them in the KV index and
 * nothing else. Qwen2's other KV kernels are shared with Qwen3 and live in {@link
 * Qwen3PagedKvKernels}, which is where their twins are.
 *
 * <p>Addressing is defined once, in {@link KvBlockAddress}.
 */
public class Qwen2PagedKvKernels {

    public Qwen2PagedKvKernels() {}

    public static void processHeadsFlashAttentionPaged(
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
            int blockStride) {

        // Thread and workgroup information
        int globalTid = context.globalIdx;
        int localTid = context.localIdx;
        int localSize = context.localGroupSizeX;
        int workgroupId = context.groupIdx;

        // Calculate which head this workgroup processes
        int h = workgroupId;

        // Early exit if beyond head count
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        int BLOCK_SIZE_C = 8;

        // Allocate shared memory for tiled computation
        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_max = context.allocateFloatLocalArray(1);
        float[] output_shared = context.allocateFloatLocalArray(headSize);

        // Each thread accumulates only its own output dimensions.
        for (int d = localTid; d < headSize; d += localSize) {
            output_shared[d] = 0.0f;
        }
        context.localBarrier();

        // Thread-local accumulators for online softmax
        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        // Cooperatively load query vector into shared memory
        for (int i = localTid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }
        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = 0; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Cooperatively load key and value vectors for this tile
            for (int tIdxInSeq = tileC + localTid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC;
                int tileMemOffset = k_v_idx_in_tile * headSize;

                // The block walk depends on the position, not on d, so it is computed once
                // per position. Keeping it inside the loop — where the legacy line that
                // computed loff + pos*kvDim + d sat — costs a table load and an integer
                // divide per element, which measured -7.3% on Qwen2 Q8_0.
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

            // Cooperatively compute attention scores for this tile
            for (int tIdxInSeq = tileC + localTid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC;

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                score /= TornadoMath.sqrt(headSize);
                s_tile[score_idx_in_tile] = score;
            }
            context.localBarrier();

            // Find max score in this tile using reduction
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) {
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            // Thread 0 broadcasts the max
            if (localTid == 0) {
                shared_max[0] = tileLocalMax;
            }
            context.localBarrier();
            float currentTileMax = shared_max[0];

            // Update global max and rescale if needed
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = localTid; d < headSize; d += localSize) {
                    output_shared[d] *= scale;
                }
            }
            maxScore = newMax;

            // Process each key-value pair in the tile
            for (int t_idx_in_s_tile = 0; t_idx_in_s_tile <= tileEnd - tileC; t_idx_in_s_tile++) {
                float expScore = TornadoMath.exp(s_tile[t_idx_in_s_tile] - maxScore);
                sumExp += expScore;

                // Accumulate weighted values
                for (int d = localTid; d < headSize; d += localSize) {
                    output_shared[d] += expScore * v_tile[t_idx_in_s_tile * headSize + d];
                }
            }
            context.localBarrier();
        }

        // Normalize and cooperatively write final results
        float normFactor = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (int d = localTid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output_shared[d] * normFactor);
        }
    }

    // FP16-cache twin of processHeadsFlashAttentionPaged: identical arithmetic, the cache stored in
    // binary16.
    public static void processHeadsFlashAttentionFP16Paged(
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
            int blockStride) {

        // Thread and workgroup information
        int globalTid = context.globalIdx;
        int localTid = context.localIdx;
        int localSize = context.localGroupSizeX;
        int workgroupId = context.groupIdx;

        // Calculate which head this workgroup processes
        int h = workgroupId;

        // Early exit if beyond head count
        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHeadIdx = h / kvMul;
        int BLOCK_SIZE_C = 8;

        // Allocate shared memory for tiled computation
        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_max = context.allocateFloatLocalArray(1);

        // Per-thread output accumulation
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        // Thread-local accumulators for online softmax
        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        // Cooperatively load query vector into shared memory
        for (int i = localTid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }
        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = 0; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Cooperatively load key and value vectors for this tile
            for (int tIdxInSeq = tileC + localTid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC;
                int tileMemOffset = k_v_idx_in_tile * headSize;

                // The block walk depends on the position, not on d, so it is computed once
                // per position. Keeping it inside the loop — where the legacy line that
                // computed loff + pos*kvDim + d sat — costs a table load and an integer
                // divide per element, which measured -7.3% on Qwen2 Q8_0.
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

            // Cooperatively compute attention scores for this tile
            for (int tIdxInSeq = tileC + localTid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC;

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                score /= TornadoMath.sqrt(headSize);
                s_tile[score_idx_in_tile] = score;
            }
            context.localBarrier();

            // Find max score in this tile using reduction
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) {
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            // Thread 0 broadcasts the max
            if (localTid == 0) {
                shared_max[0] = tileLocalMax;
            }
            context.localBarrier();
            float currentTileMax = shared_max[0];

            // Update global max and rescale if needed
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
            }
            maxScore = newMax;

            // Process each key-value pair in the tile
            for (int t_idx_in_s_tile = 0; t_idx_in_s_tile <= tileEnd - tileC; t_idx_in_s_tile++) {
                float expScore = TornadoMath.exp(s_tile[t_idx_in_s_tile] - maxScore);
                sumExp += expScore;

                // Accumulate weighted values
                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * v_tile[t_idx_in_s_tile * headSize + d];
                }
            }
            context.localBarrier();
        }

        // Normalize and cooperatively write final results
        float normFactor = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (int d = localTid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output[d] * normFactor);
        }
    }

    public static void batchedRopeWithKVCacheQwen2Paged(
            KernelContext context,
            IntArray batchStartPosHolder,
            IntArray activeBatchSizeHolder,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            int kvDim,
            int headSize,
            int layerIndex,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int dim,
            float ropeTheta) {

        int index = context.globalIdx;
        int pairsPerToken = dim / 2;
        int token = index / pairsPerToken;
        int pair = index % pairsPerToken;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int position = batchStartPosHolder.get(0) + token;
        int slot = batchStartPosHolder.get(2);
        int halfHeadSize = headSize / 2;
        int component = pair % halfHeadSize;
        int head = pair / halfHeadSize;

        float frequency = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * component / (float) headSize);
        float angle = position * frequency;
        float cosine = TornadoMath.cos(angle);
        float sine = TornadoMath.sin(angle);

        int qHeadOffset = token * dim + head * headSize;
        float q0 = qBatch.get(qHeadOffset + component);
        float q1 = qBatch.get(qHeadOffset + component + halfHeadSize);
        qBatch.set(qHeadOffset + component, q0 * cosine - q1 * sine);
        qBatch.set(qHeadOffset + component + halfHeadSize, q0 * sine + q1 * cosine);

        if (pair < kvDim / 2) {
            int kvHead = pair / halfHeadSize;
            int kHeadOffset = token * kvDim + kvHead * headSize;
            float k0 = kBatch.get(kHeadOffset + component);
            float k1 = kBatch.get(kHeadOffset + component + halfHeadSize);
            float rotatedK0 = k0 * cosine - k1 * sine;
            float rotatedK1 = k0 * sine + k1 * cosine;

            kBatch.set(kHeadOffset + component, rotatedK0);
            kBatch.set(kHeadOffset + component + halfHeadSize, rotatedK1);

            int cacheOffset =
                    KvBlockAddress.offset(
                                    blockTable,
                                    slot,
                                    position,
                                    KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg),
                                    kvDim,
                                    blockCfg,
                                    blockStride)
                            + kvHead * headSize;
            keyCache.set(cacheOffset + component, rotatedK0);
            keyCache.set(cacheOffset + component + halfHeadSize, rotatedK1);
            valueCache.set(cacheOffset + component, vBatch.get(kHeadOffset + component));
            valueCache.set(
                    cacheOffset + component + halfHeadSize,
                    vBatch.get(kHeadOffset + component + halfHeadSize));
        }
    }
}
