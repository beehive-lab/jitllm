package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for batched prefill.
 *
 * <p>Each kernel processes {@code batchSize} tokens simultaneously. Batch tensors are flat: element
 * [b][i] lives at index {@code b*stride + i}. Worker-grid sizes are scaled by {@code batchSize} vs
 * the single-token kernels.
 *
 * <p>These kernels are meant to be registered in {@link
 * org.beehive.jllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode} TaskGraphs.
 */
public final class TransformerBatchPrefillKernels {

    // @formatter:off
    private TransformerBatchPrefillKernels() {}

    // ── RMS Norm (attention) ─────────────────────────────────────────────────

    /**
     * Sequential RMS reduction — one thread per batch item.
     *
     * <p>Each thread computes the RMS scale factor for its token: {@code scale[b] = 1 / sqrt(
     * mean(x[b]²) + eps)} Worker: batchSize global threads, localSize=1.
     */
    public static void batchedRmsReduce(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray attnScaleBatch,
            int dim,
            float eps) {
        int b = context.globalIdx;
        int base = b * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = wrapXBatch.get(base + i);
            ss += val * val;
        }
        ss /= dim;
        ss += eps;
        attnScaleBatch.set(b, 1.0f / TornadoMath.sqrt(ss));
    }

    /**
     * Applies RMS normalization and FP16-quantizes the result.
     *
     * <p>{@code xbFP16Batch[b*dim+i] = FP16(rmsWeights[i] * scale[b] * x[b*dim+i])} Worker: B*dim
     * global threads, localSize=256.
     */
    public static void batchedRmsApplyFP16(
            KernelContext context,
            HalfFloatArray xbFP16Batch,
            FloatArray wrapXBatch,
            FloatArray rmsWeights,
            FloatArray attnScaleBatch,
            int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        float scale = attnScaleBatch.get(b);
        float result = rmsWeights.get(i) * scale * wrapXBatch.get(gid);
        xbFP16Batch.set(gid, new HalfFloat(result));
    }

    // ── QKV Projection ────────────────────────────────────────────────────────

    /**
     * Fused batched QKV projection (FP16 weights, FP16 input).
     *
     * <p>One workgroup per (batchIdx, outputRow) pair. globalGroupIdx = batchIdx * (dim + 2*kvDim)
     * + rowIdx. Worker: B*(dim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmul(
            KernelContext context,
            HalfFloatArray xbFP16Batch,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int totalRows = dim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx = groupId % totalRows;
        int inputOff = batchIdx * dim;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < dim) {
            int rowOff = rowIdx * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial +=
                        wq.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapQBatch.set(batchIdx * dim + rowIdx, localSum[0]);
            }

        } else if (rowIdx < dim + kvDim) {
            int kRow = rowIdx - dim;
            int rowOff = kRow * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial +=
                        wk.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);
            }

        } else {
            int vRow = rowIdx - dim - kvDim;
            int rowOff = vRow * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial +=
                        wv.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
            }
        }
    }

    /**
     * Batched DECODE flash attention: B independent sequences, one query token each. Identical
     * online-softmax math to {@link #batchedFlashAttention}, but each batch slot has its OWN KV
     * cache region and its OWN position, so slot {@code b} attends positions {@code
     * 0.seqPositions[b]} of its own cache — the shape produced by batching B concurrent decode
     * requests.
     *
     * <p>KV cache layout: one contiguous region of {@code numLayers * contextLength * kvDim} per
     * slot, so the base for slot b, layer L is {@code b*numLayers*contextLength*kvDim +
     * L*contextLength*kvDim}.
     *
     * <p>One workgroup per (batchIdx, head): {@code groupId = batchIdx*nHeads + h}.
     */
    public static void batchedDecodeAttention(
            KernelContext context,
            IntArray seqPositions,
            FloatArray wrapQBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            FloatArray wrapXbBatch,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int contextLength,
            int dim) {
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = seqPositions.get(batchIdx); // per-slot position
        int loff =
                batchIdx * (numLayers * contextLength * kvDim)
                        + layerIndex * contextLength * kvDim; // per-slot KV base
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);
        float[] maxHolder = context.allocateFloatLocalArray(1);

        int qOffset = batchIdx * dim + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = wrapQBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                int tileMOff = tInTile * headSize;
                for (int d = 0; d < headSize; d++) {
                    int kvOff = loff + t * kvDim + kvHeadIdx * headSize + d;
                    kTile[tileMOff + d] = wrapKeyCache.get(kvOff);
                    vTile[tileMOff + d] = wrapValueCache.get(kvOff);
                }
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t <= tileEnd - tileC; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }
            if (tid == 0) {
                maxHolder[0] = tileMax;
            }
            context.localBarrier();
            float curTileMax = maxHolder[0];

            float newMax = Math.max(maxScore, curTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
            }
            maxScore = newMax;

            for (int t = 0; t <= tileEnd - tileC; t++) {
                float expScore = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += expScore;
                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * vTile[t * headSize + d];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int xbOffset = batchIdx * dim + h * headSize;
        for (int d = tid; d < headSize; d += localSz) {
            wrapXbBatch.set(xbOffset + d, output[d] * norm);
        }
    }

    // ── Output / FFN Projections ─────────────────────────────────────────────

    /**
     * Batched matrix-vector multiply with residual add.
     *
     * <p>Used for both the attention output projection (Wo) and the FFN down projection (W2). One
     * workgroup per (batchIdx, outputRow): {@code groupIdx = batchIdx * d + rowIdx}.
     *
     * <ul>
     *   <li>Wo: inputBatch=xbBatch (B×dim), outputBatch=xBatch (B×dim), n=dim, d=dim
     *   <li>W2: inputBatch=hbBatch (B×hiddenDim), outputBatch=xBatch (B×dim), n=hiddenDim, d=dim
     * </ul>
     *
     * Worker: B*d workgroups × localWorkGroupSize threads.
     */
    public static void batchedMatVecWithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            HalfFloatArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / d;
        int rowIdx = groupId % d;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            partial += w.get(rowOff + j).getFloat32() * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            int outIdx = batchIdx * d + rowIdx;
            outputBatch.set(outIdx, outputBatch.get(outIdx) + localSum[0]);
        }
    }

    // ── FFN RMS Norm ─────────────────────────────────────────────────────────

    /**
     * Sequential FFN RMS reduction — one thread per batch item. Worker: batchSize global threads,
     * localSize=1.
     */
    public static void batchedFFNRmsReduce(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray ffnScaleBatch,
            int dim,
            float eps) {
        int b = context.globalIdx;
        int base = b * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = wrapXBatch.get(base + i);
            ss += val * val;
        }
        ss /= dim;
        ss += eps;
        ffnScaleBatch.set(b, 1.0f / TornadoMath.sqrt(ss));
    }

    // ── FFN SwiGLU ───────────────────────────────────────────────────────────

    /**
     * Batched fused RMS-apply + W1/W3 gate-up projections + SiLU + GLU.
     *
     * <p>One workgroup per (batchIdx, hiddenRow): {@code groupIdx = batchIdx * hiddenDim + rowIdx}.
     * Worker: B*hiddenDim workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedRmsNormFFNGateUp(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray wrapHbBatch,
            FloatArray rmsFFNWeights,
            FloatArray ffnScaleBatch,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / hiddenDim;
        int rowIdx = groupId % hiddenDim;

        float scale = ffnScaleBatch.get(batchIdx);
        int inputOff = batchIdx * dim;
        int rowOff = rowIdx * dim;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        // W1 matmul with inline RMS apply
        float sum1 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum1 += w1.get(rowOff + j).getFloat32() * normed;
        }
        localSum[localId] = sum1;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];

        // W3 matmul with inline RMS apply
        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum3 += w3.get(rowOff + j).getFloat32() * normed;
        }
        localSum[localId] = sum3;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        // SiLU(W1·x) × (W3·x)
        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            wrapHbBatch.set(batchIdx * hiddenDim + rowIdx, silu * result3);
        }
    }

    // ── Q8_0 Batch Kernels ───────────────────────────────────────────────────

    /**
     * No-op kernel for Q8_0 batch activation graph. The host fills wrapXBatch with dequantized FP32
     * embeddings before execution. Worker: 1 global thread.
     */
    public static void batchPassthrough(KernelContext context, FloatArray wrapXBatch) {
        if (context.globalIdx == 0) {
            wrapXBatch.set(0, wrapXBatch.get(0));
        }
    }

    /**
     * Applies RMS normalization to FP32 — Q8_0 variant. Writes normalized FP32 to wrapXbBatch
     * (reused as xb intermediate before QKV). Worker: B*dim global threads, localSize=256.
     */
    public static void batchedRmsApplyFP32(
            KernelContext context,
            FloatArray wrapXbBatch,
            FloatArray wrapXBatch,
            FloatArray rmsWeights,
            FloatArray attnScaleBatch,
            int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        wrapXbBatch.set(gid, rmsWeights.get(i) * attnScaleBatch.get(b) * wrapXBatch.get(gid));
    }

    /**
     * Fused batched QKV projection with Q8_0 weight dequantization. Input wrapXbBatch is FP32
     * (written by batchedRmsApplyFP32). groupIdx = batchIdx * (dim + 2*kvDim) + rowIdx. Worker:
     * B*(dim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmulQ8(
            KernelContext context,
            FloatArray wrapXbBatch,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int totalRows = dim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx = groupId % totalRows;
        int inputOff = batchIdx * dim;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < dim) {
            int rowBlockOffset = rowIdx * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wq.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wq.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapQBatch.set(batchIdx * dim + rowIdx, localSum[0]);
            }

        } else if (rowIdx < dim + kvDim) {
            int kRow = rowIdx - dim;
            int rowBlockOffset = kRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wk.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wk.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);
            }

        } else {
            int vRow = rowIdx - dim - kvDim;
            int rowBlockOffset = vRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wv.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wv.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
            }
        }
    }

    /**
     * Batched matrix-vector multiply with residual add (Q8_0 weights). Used for attention output
     * (Wo) and FFN down (W2) projections. groupIdx = batchIdx * d + rowIdx. Worker: B*d workgroups
     * × localWorkGroupSize threads.
     */
    public static void batchedMatVecWithResidualQ8(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / d;
        int rowIdx = groupId % d;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (n + blockSize - 1) / blockSize;
        int rowBlockOffset = rowIdx * blocksPerRow;
        int inputOff = batchIdx * n;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float scale = w.getHalfFloat(blockByteOffset).getFloat32();
            float quant = w.get(blockByteOffset + 2 + j % blockSize);
            partial += quant * scale * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            int outIdx = batchIdx * d + rowIdx;
            outputBatch.set(outIdx, outputBatch.get(outIdx) + localSum[0]);
        }
    }

    /**
     * Batched fused RMS-apply + W1/W3 gate-up projections + SiLU + GLU (Q8_0 weights). groupIdx =
     * batchIdx * hiddenDim + rowIdx. Worker: B*hiddenDim workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedRmsNormFFNGateUpQ8(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray wrapHbBatch,
            FloatArray rmsFFNWeights,
            FloatArray ffnScaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / hiddenDim;
        int rowIdx = groupId % hiddenDim;

        float scale = ffnScaleBatch.get(batchIdx);
        int inputOff = batchIdx * dim;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;
        int rowBlockOffset = rowIdx * blocksPerRow;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        float sum1 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float w1Scale = w1.getHalfFloat(blockByteOffset).getFloat32();
            float w1Quant = w1.get(blockByteOffset + 2 + j % blockSize);
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum1 += w1Quant * w1Scale * normed;
        }
        localSum[localId] = sum1;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];

        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float w3Scale = w3.getHalfFloat(blockByteOffset).getFloat32();
            float w3Quant = w3.get(blockByteOffset + 2 + j % blockSize);
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum3 += w3Quant * w3Scale * normed;
        }
        localSum[localId] = sum3;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            wrapHbBatch.set(batchIdx * hiddenDim + rowIdx, silu * result3);
        }
    }

    /**
     * RMS-apply for FFN, writing FP16. Mirrors batchedRmsApplyFP16 but pulls scale from
     * ffnScaleBatch. Output is the A operand for the W1/W3 MMA tasks.
     *
     * <p>Worker: B*dim global threads, localSize=256.
     */
    public static void batchedFFNRmsApplyFP16(
            KernelContext context,
            HalfFloatArray normedXFFNFP16,
            FloatArray wrapXBatch,
            FloatArray rmsFFNWeights,
            FloatArray ffnScaleBatch,
            int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        float scale = ffnScaleBatch.get(b);
        float result = rmsFFNWeights.get(i) * scale * wrapXBatch.get(gid);
        normedXFFNFP16.set(gid, new HalfFloat(result));
    }

    private static final int WARP_SIZE = 32;
    private static final int BM = 128, BN = 128, BK = 16;
    private static final int WARPS_M = 4, WARPS_N = 2;
    private static final int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
    private static final int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    private static final int WM = BM / WARPS_M;
    private static final int WN = BN / WARPS_N;
    private static final int B_SUBTILE_BYTES = 256;

    /**
     * Packs two consecutive FP16 values into one int (lo | hi<<16) for the shared-memory ldmatrix
     * tiles. Leaf helper; inlined by the Tornado JIT.
     */
    private static int packHalves(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    /**
     * Tensor-core GEMM: C[M,N] (FP32) = A[M,K] (FP16, row-major) × B[N,K] (FP16, row-major).
     *
     * <p>Software-pipelined: each thread stages the NEXT K-step's A/B elements in registers while
     * the CURRENT step's ldmatrix+MMA execute, so global-memory latency is hidden behind
     * tensor-core compute. Shared memory stays single-buffered; the two block barriers per step
     * preserve correctness (read-complete before overwrite, write-complete before next ldmatrix).
     *
     * <p>Requires M % 128 == 0, N % 128 == 0, K % 16 == 0, SM 8.0+. Worker:
     * WorkerGrid2D((M/128)*256, N/128), local (256,1,1).
     */
    public static void gemmMMA(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + K);
        int bReg1 = packHalves(B, gB1, gB1 + K);
        int bReg2 = packHalves(B, gB2, gB2 + K);
        int bReg3 = packHalves(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    // ── Residual add (FP32) ───────────────────────────────────────────────
    // gemmMMA overwrites C, but Wo and W2 both need x = x + W·a.
    // Worker: B*dim global threads (valid rows only), localSize=256.
    public static void batchedResidualAddFP32(
            KernelContext context,
            FloatArray residual, // x (in/out)
            FloatArray delta) { // GEMM output
        int gid = context.globalIdx;
        residual.set(gid, residual.get(gid) + delta.get(gid));
    }

    // ── Fused MMA projections ─────────────────────────────────────────────────
    //
    // Q, K, V (and gate/up) share the same A operand and the same K dimension,
    // so they are fused into ONE kernel launch each. The N grid spans the packed
    // output [dim | kvDim | kvDim] (resp. [hidDim | hidDim]); each thread block
    // selects its weight matrix from groupIdy — a block-uniform branch, so there
    // is zero divergence and no weight duplication in memory. This restores the
    // A-reuse of the old fused matvec kernels AND fixes grid starvation for the
    // skinny GQA projections (kvDim/128 blocks alone cannot fill the SMs).

    /**
     * Fused QKV tensor-core GEMM into a PACKED output: qkvOut[M, dim+2*kvDim] = A[M,K] × [Wq | Wk |
     * Wv] (each [N_i, K] row-major).
     *
     * <p>Layout of a row of qkvOut: [ q(0.dim) | k(0.kvDim) | v(0.kvDim) ]. Requires dim % 128 == 0
     * and kvDim % 128 == 0. Worker: WorkerGrid2D((M/128)*256, (dim+2*kvDim)/128), local (256,1,1).
     */
    public static void gemmMMAQKV(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            FloatArray qkvOut,
            int M,
            int dim,
            int kvDim,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy; // column in the packed output
        int qkvStride = dim + 2 * kvDim;

        // Column base inside the segment's own weight matrix (block-uniform).
        int wColBase = blockCol;
        if (blockCol >= dim) wColBase -= dim;
        if (blockCol >= dim + kvDim) wColBase -= kvDim;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 = (wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ──
        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < dim) {
            bReg0 = packHalves(wq, gB0, gB0 + K);
            bReg1 = packHalves(wq, gB1, gB1 + K);
            bReg2 = packHalves(wq, gB2, gB2 + K);
            bReg3 = packHalves(wq, gB3, gB3 + K);
        } else if (blockCol < dim + kvDim) {
            bReg0 = packHalves(wk, gB0, gB0 + K);
            bReg1 = packHalves(wk, gB1, gB1 + K);
            bReg2 = packHalves(wk, gB2, gB2 + K);
            bReg3 = packHalves(wk, gB3, gB3 + K);
        } else {
            bReg0 = packHalves(wv, gB0, gB0 + K);
            bReg1 = packHalves(wv, gB1, gB1 + K);
            bReg2 = packHalves(wv, gB2, gB2 + K);
            bReg3 = packHalves(wv, gB3, gB3 + K);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < dim) {
                    bReg0 = packHalves(wq, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wq, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wq, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wq, gB3 + kOff, gB3 + kOff + K);
                } else if (blockCol < dim + kvDim) {
                    bReg0 = packHalves(wk, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wk, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wk, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wk, gB3 + kOff, gB3 + kOff + K);
                } else {
                    bReg0 = packHalves(wv, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wv, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wv, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wv, gB3 + kOff, gB3 + kOff + K);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        // Stores are uniform: packed column == blockCol-relative column,
        // stride is the packed row width.
        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, qkvOut, rBase + 0, cBase + 0, qkvStride);
        ctx.mmaStore(c01, qkvOut, rBase + 0, cBase + 8, qkvStride);
        ctx.mmaStore(c02, qkvOut, rBase + 0, cBase + 16, qkvStride);
        ctx.mmaStore(c03, qkvOut, rBase + 0, cBase + 24, qkvStride);
        ctx.mmaStore(c04, qkvOut, rBase + 0, cBase + 32, qkvStride);
        ctx.mmaStore(c05, qkvOut, rBase + 0, cBase + 40, qkvStride);
        ctx.mmaStore(c06, qkvOut, rBase + 0, cBase + 48, qkvStride);
        ctx.mmaStore(c07, qkvOut, rBase + 0, cBase + 56, qkvStride);
        ctx.mmaStore(c10, qkvOut, rBase + 16, cBase + 0, qkvStride);
        ctx.mmaStore(c11, qkvOut, rBase + 16, cBase + 8, qkvStride);
        ctx.mmaStore(c12, qkvOut, rBase + 16, cBase + 16, qkvStride);
        ctx.mmaStore(c13, qkvOut, rBase + 16, cBase + 24, qkvStride);
        ctx.mmaStore(c14, qkvOut, rBase + 16, cBase + 32, qkvStride);
        ctx.mmaStore(c15, qkvOut, rBase + 16, cBase + 40, qkvStride);
        ctx.mmaStore(c16, qkvOut, rBase + 16, cBase + 48, qkvStride);
        ctx.mmaStore(c17, qkvOut, rBase + 16, cBase + 56, qkvStride);
    }

    /**
     * Fused W1/W3 (gate/up) tensor-core GEMM into a PACKED output: gateUpOut[M, 2*hidDim] = A[M,K]
     * × [W1 | W3] (each [hidDim, K] row-major).
     *
     * <p>Layout of a row: [ gate(0.hidDim) | up(0.hidDim) ]. Requires hidDim % 128 == 0. Worker:
     * WorkerGrid2D((M/128)*256, (2*hidDim)/128), local (256,1,1).
     */
    public static void gemmMMAGateUp(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray w1,
            HalfFloatArray w3,
            FloatArray gateUpOut,
            int M,
            int hidDim,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy; // column in the packed output
        int outStride = 2 * hidDim;

        int wColBase = (blockCol < hidDim) ? blockCol : (blockCol - hidDim);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 = (wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < hidDim) {
            bReg0 = packHalves(w1, gB0, gB0 + K);
            bReg1 = packHalves(w1, gB1, gB1 + K);
            bReg2 = packHalves(w1, gB2, gB2 + K);
            bReg3 = packHalves(w1, gB3, gB3 + K);
        } else {
            bReg0 = packHalves(w3, gB0, gB0 + K);
            bReg1 = packHalves(w3, gB1, gB1 + K);
            bReg2 = packHalves(w3, gB2, gB2 + K);
            bReg3 = packHalves(w3, gB3, gB3 + K);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < hidDim) {
                    bReg0 = packHalves(w1, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(w1, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(w1, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(w1, gB3 + kOff, gB3 + kOff + K);
                } else {
                    bReg0 = packHalves(w3, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(w3, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(w3, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(w3, gB3 + kOff, gB3 + kOff + K);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, gateUpOut, rBase + 0, cBase + 0, outStride);
        ctx.mmaStore(c01, gateUpOut, rBase + 0, cBase + 8, outStride);
        ctx.mmaStore(c02, gateUpOut, rBase + 0, cBase + 16, outStride);
        ctx.mmaStore(c03, gateUpOut, rBase + 0, cBase + 24, outStride);
        ctx.mmaStore(c04, gateUpOut, rBase + 0, cBase + 32, outStride);
        ctx.mmaStore(c05, gateUpOut, rBase + 0, cBase + 40, outStride);
        ctx.mmaStore(c06, gateUpOut, rBase + 0, cBase + 48, outStride);
        ctx.mmaStore(c07, gateUpOut, rBase + 0, cBase + 56, outStride);
        ctx.mmaStore(c10, gateUpOut, rBase + 16, cBase + 0, outStride);
        ctx.mmaStore(c11, gateUpOut, rBase + 16, cBase + 8, outStride);
        ctx.mmaStore(c12, gateUpOut, rBase + 16, cBase + 16, outStride);
        ctx.mmaStore(c13, gateUpOut, rBase + 16, cBase + 24, outStride);
        ctx.mmaStore(c14, gateUpOut, rBase + 16, cBase + 32, outStride);
        ctx.mmaStore(c15, gateUpOut, rBase + 16, cBase + 40, outStride);
        ctx.mmaStore(c16, gateUpOut, rBase + 16, cBase + 48, outStride);
        ctx.mmaStore(c17, gateUpOut, rBase + 16, cBase + 56, outStride);
    }

    // ── Parallel RMS reductions ───────────────────────────────────────────────
    // Replace the localSize=1 sequential reductions (one thread walking `dim`
    // elements alone) with one 256-thread workgroup per token and a shared-memory
    // tree reduction.

    /**
     * Parallel RMS square-sum reduction. One workgroup per batch token.
     *
     * <p>Worker: B workgroups × localSize threads (localSize=256).
     */
    public static void batchedRmsReduceParallel(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray scaleBatch,
            int dim,
            float eps,
            int localSize) {
        int tid = context.localIdx;
        int b = context.groupIdx;
        int localSz = context.localGroupSizeX;
        float[] partial = context.allocateFloatLocalArray(localSize);

        int base = b * dim;
        float ss = 0.0f;
        for (int i = tid; i < dim; i += localSz) {
            float v = wrapXBatch.get(base + i);
            ss += v * v;
        }
        partial[tid] = ss;
        context.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] += partial[tid + s];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            float m = partial[0] / dim + eps;
            scaleBatch.set(b, 1.0f / TornadoMath.sqrt(m));
        }
    }

    /**
     * Parallel RMS reduction FUSED with the pending residual add: x[b,i] += delta[b,i] first, then
     * square-sum over the updated row. Replaces the separate woResid task + FFN RMS reduce (each
     * element is visited exactly once, so the in-place update is race-free).
     *
     * <p>Worker: B workgroups × localSize threads (localSize=256).
     */
    public static void batchedRmsReduceFusedResidual(
            KernelContext context,
            FloatArray wrapXBatch,
            FloatArray delta,
            FloatArray scaleBatch,
            int dim,
            float eps,
            int localSize) {
        int tid = context.localIdx;
        int b = context.groupIdx;
        int localSz = context.localGroupSizeX;
        float[] partial = context.allocateFloatLocalArray(localSize);

        int base = b * dim;
        float ss = 0.0f;
        for (int i = tid; i < dim; i += localSz) {
            float v = wrapXBatch.get(base + i) + delta.get(base + i);
            wrapXBatch.set(base + i, v);
            ss += v * v;
        }
        partial[tid] = ss;
        context.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] += partial[tid + s];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            float m = partial[0] / dim + eps;
            scaleBatch.set(b, 1.0f / TornadoMath.sqrt(m));
        }
    }

    // ── Batched DECODE variants (per-slot KV cache + per-slot position) ──────
    //
    // These two kernels are the only semantic delta between batched PREFILL (B
    // tokens of ONE sequence, shared causal KV) and batched DECODE (B independent
    // sequences, each with its own KV region and its own position). The math is
    // identical to the *Packed / *FP16Out prefill kernels above; only the KV
    // addressing changes:
    //   pos  = seqPositions[batchIdx]                                  (per slot)
    //   base = batchIdx*(numLayers*ctx*kvDim) + layer*ctx*kvDim        (per slot)
    // The KV cache is therefore sized B*numLayers*contextLength*kvDim.

    /**
     * Per-slot RoPE + KV-cache write over the packed QKV buffer (decode).
     *
     * <p>Fork of {@link #batchedRopeWithKVCachePacked}: each batch slot rotates at its own position
     * {@code seqPositions[batchIdx]} and writes K/V into its own KV region ({@code batchIdx} stride
     * = {@code numLayers*contextLength*kvDim}).
     */
    public static void batchedDecodeRopeWithKVCachePacked(
            KernelContext context,
            IntArray seqPositions,
            FloatArray qkvBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            int numLayers,
            int contextLength,
            int dim) {
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int pairIdx = globalIdx % halfDim;
        int i = pairIdx * 2;
        int qkvStride = dim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);
        int qOffset = batchIdx * qkvStride;
        int kOffset = batchIdx * qkvStride + dim;
        int vOffset = batchIdx * qkvStride + dim + kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            float v0q = qkvBatch.get(qOffset + i);
            float v1q = qkvBatch.get(qOffset + i + 1);
            qkvBatch.set(qOffset + i, v0q * fcr - v1q * fci);
            qkvBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            if (i + 1 < kvDim) {
                float v0k = qkvBatch.get(kOffset + i);
                float v1k = qkvBatch.get(kOffset + i + 1);
                float rotK0 = v0k * fcr - v1k * fci;
                float rotK1 = v0k * fci + v1k * fcr;

                int slotBase = batchIdx * (numLayers * contextLength * kvDim);
                int cacheOff = slotBase + layerIndex * contextLength * kvDim + pos * kvDim;
                wrapKeyCache.set(cacheOff + i, rotK0);
                wrapKeyCache.set(cacheOff + i + 1, rotK1);
                wrapValueCache.set(cacheOff + i, qkvBatch.get(vOffset + i));
                wrapValueCache.set(cacheOff + i + 1, qkvBatch.get(vOffset + i + 1));
            }
        }
    }

    /**
     * Per-slot flash attention over the packed QKV buffer, FP16 output (decode).
     *
     * <p>Fork of {@link #batchedFlashAttentionFP16Out}: each batch slot attends over {@code
     * 0.seqPositions[batchIdx]} of its OWN KV region. Same register-partitioned P·V accumulation
     * and FP16 emission.
     *
     * <p>Requires headSize <= 2*localSz (localSz = min(headSize, 128)).
     */
    public static void batchedDecodeAttentionFP16Out(
            KernelContext context,
            IntArray seqPositions,
            FloatArray qkvBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            HalfFloatArray attnOutFP16,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int contextLength,
            int dim) {
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = seqPositions.get(batchIdx);
        int loff =
                batchIdx * (numLayers * contextLength * kvDim) + layerIndex * contextLength * kvDim;
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = batchIdx * qkvStride + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = qkvBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int d1 = tid + localSz;

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int d = idx % headSize;
                int kvOff = loff + (tileC + tInTile) * kvDim + kvHeadIdx * headSize + d;
                kTile[tInTile * headSize + d] = wrapKeyCache.get(kvOff);
                vTile[tInTile * headSize + d] = wrapValueCache.get(kvOff);
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr;
                acc1 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                acc0 += p * vTile[t * headSize + tid];
                if (d1 < headSize) {
                    acc1 += p * vTile[t * headSize + d1];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outOffset = batchIdx * dim + h * headSize;
        attnOutFP16.set(outOffset + tid, new HalfFloat(acc0 * norm));
        if (d1 < headSize) {
            attnOutFP16.set(outOffset + d1, new HalfFloat(acc1 * norm));
        }
    }

    // ── Paged KV variants (block-table indirection) ─────────────────────────
    //
    // KV lives in a global pool of fixed-size blocks. A block holds `blockSize`
    // consecutive positions of ONE sequence across ALL layers:
    //   pool[ physBlock*(numLayers*blockSize*kvDim) + layer*(blockSize*kvDim)
    //         + (pos % blockSize)*kvDim + c ]
    // The per-slot block table maps a logical block to a physical one:
    //   physBlock = blockTable[batchIdx*maxBlocksPerSlot + pos/blockSize]
    // This removes the fixed per-slot context reservation of the contiguous cache:
    // slots draw blocks from a shared pool only for the tokens they actually hold,
    // so the pool can be far smaller than B*ctx (and blocks can be shared for
    // prefix caching).

    /**
     * Paged per-slot RoPE + KV write (Llama adjacent-pair). {@code blockCfg} packs {@code blockSize
     * | (maxBlocksPerSlot << 16)} to stay within the task arg limit.
     */
    public static void batchedDecodePagedRopeWithKVCachePacked(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            FloatArray keyPool,
            FloatArray valuePool,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int dim) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int pairIdx = globalIdx % halfDim;
        int i = pairIdx * 2;
        int qkvStride = dim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);
        int qOffset = batchIdx * qkvStride;
        int kOffset = batchIdx * qkvStride + dim;
        int vOffset = batchIdx * qkvStride + dim + kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            float v0q = qkvBatch.get(qOffset + i);
            float v1q = qkvBatch.get(qOffset + i + 1);
            qkvBatch.set(qOffset + i, v0q * fcr - v1q * fci);
            qkvBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            if (i + 1 < kvDim) {
                float v0k = qkvBatch.get(kOffset + i);
                float v1k = qkvBatch.get(kOffset + i + 1);
                float rotK0 = v0k * fcr - v1k * fci;
                float rotK1 = v0k * fci + v1k * fcr;

                int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + pos / blockSize);
                int slotInBlock = pos % blockSize;
                int cacheOff =
                        physBlock * (numLayers * blockSize * kvDim)
                                + layerIndex * (blockSize * kvDim)
                                + slotInBlock * kvDim;
                keyPool.set(cacheOff + i, rotK0);
                keyPool.set(cacheOff + i + 1, rotK1);
                valuePool.set(cacheOff + i, qkvBatch.get(vOffset + i));
                valuePool.set(cacheOff + i + 1, qkvBatch.get(vOffset + i + 1));
            }
        }
    }

    // FP16-cache twin of batchedDecodePagedRopeWithKVCachePacked: identical arithmetic, the cache
    // stored in binary16.
    /**
     * Paged per-slot RoPE + KV write (Llama adjacent-pair). {@code blockCfg} packs {@code blockSize
     * | (maxBlocksPerSlot << 16)} to stay within the task arg limit.
     */
    public static void batchedDecodePagedRopeWithKVCachePackedKVFP16(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int dim) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int pairIdx = globalIdx % halfDim;
        int i = pairIdx * 2;
        int qkvStride = dim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);
        int qOffset = batchIdx * qkvStride;
        int kOffset = batchIdx * qkvStride + dim;
        int vOffset = batchIdx * qkvStride + dim + kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            float v0q = qkvBatch.get(qOffset + i);
            float v1q = qkvBatch.get(qOffset + i + 1);
            qkvBatch.set(qOffset + i, v0q * fcr - v1q * fci);
            qkvBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            if (i + 1 < kvDim) {
                float v0k = qkvBatch.get(kOffset + i);
                float v1k = qkvBatch.get(kOffset + i + 1);
                float rotK0 = v0k * fcr - v1k * fci;
                float rotK1 = v0k * fci + v1k * fcr;

                int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + pos / blockSize);
                int slotInBlock = pos % blockSize;
                int cacheOff =
                        physBlock * (numLayers * blockSize * kvDim)
                                + layerIndex * (blockSize * kvDim)
                                + slotInBlock * kvDim;
                keyPool.set(cacheOff + i, new HalfFloat(rotK0));
                keyPool.set(cacheOff + i + 1, new HalfFloat(rotK1));
                valuePool.set(cacheOff + i, new HalfFloat(qkvBatch.get(vOffset + i)));
                valuePool.set(cacheOff + i + 1, new HalfFloat(qkvBatch.get(vOffset + i + 1)));
            }
        }
    }

    /**
     * Paged per-slot flash attention, FP16 output (Llama; {@code dim = nHeads*headSize}). {@code
     * blockCfg} packs {@code blockSize | (maxBlocksPerSlot << 16)}.
     */
    public static void batchedDecodePagedAttentionFP16Out(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            FloatArray keyPool,
            FloatArray valuePool,
            HalfFloatArray attnOutFP16,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = seqPositions.get(batchIdx);
        int layerOff = layerIndex * (blockSize * kvDim);
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = batchIdx * qkvStride + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = qkvBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int d1 = tid + localSz;

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int d = idx % headSize;
                int t = tileC + tInTile;
                int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + t / blockSize);
                int kvOff =
                        physBlock * blockStride
                                + layerOff
                                + (t % blockSize) * kvDim
                                + kvHeadIdx * headSize
                                + d;
                kTile[tInTile * headSize + d] = keyPool.get(kvOff);
                vTile[tInTile * headSize + d] = valuePool.get(kvOff);
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr;
                acc1 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                acc0 += p * vTile[t * headSize + tid];
                if (d1 < headSize) {
                    acc1 += p * vTile[t * headSize + d1];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outOffset = batchIdx * dim + h * headSize;
        attnOutFP16.set(outOffset + tid, new HalfFloat(acc0 * norm));
        if (d1 < headSize) {
            attnOutFP16.set(outOffset + d1, new HalfFloat(acc1 * norm));
        }
    }

    // FP16-cache twin of batchedDecodePagedAttentionFP16Out: identical arithmetic, the cache stored
    // in binary16.
    /**
     * Paged per-slot flash attention, FP16 output (Llama; {@code dim = nHeads*headSize}). {@code
     * blockCfg} packs {@code blockSize | (maxBlocksPerSlot << 16)}.
     */
    public static void batchedDecodePagedAttentionFP16OutKVFP16(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            HalfFloatArray attnOutFP16,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = seqPositions.get(batchIdx);
        int layerOff = layerIndex * (blockSize * kvDim);
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = batchIdx * qkvStride + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = qkvBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int d1 = tid + localSz;

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int d = idx % headSize;
                int t = tileC + tInTile;
                int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + t / blockSize);
                int kvOff =
                        physBlock * blockStride
                                + layerOff
                                + (t % blockSize) * kvDim
                                + kvHeadIdx * headSize
                                + d;
                kTile[tInTile * headSize + d] = keyPool.get(kvOff).getFloat32();
                vTile[tInTile * headSize + d] = valuePool.get(kvOff).getFloat32();
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr;
                acc1 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                acc0 += p * vTile[t * headSize + tid];
                if (d1 < headSize) {
                    acc1 += p * vTile[t * headSize + d1];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outOffset = batchIdx * dim + h * headSize;
        attnOutFP16.set(outOffset + tid, new HalfFloat(acc0 * norm));
        if (d1 < headSize) {
            attnOutFP16.set(outOffset + d1, new HalfFloat(acc1 * norm));
        }
    }

    // ── On-device greedy sampling (argmax) ──────────────────────────────────

    /**
     * Per-row argmax over the batched logits: one workgroup per row reduces over the whole vocab
     * and writes the winning token id to {@code outTokens[b]}. Keeps the full logits tensor on the
     * GPU — only B integers cross to the host, instead of the paddedB×vocab (~65–78 MB) D2H copy +
     * a CPU scan every step.
     *
     * <p>Worker: B workgroups × localSize threads (localSize a power of two, e.g. 256).
     */
    public static void batchedArgmaxLogits(
            KernelContext context, FloatArray logits, IntArray outTokens, int vocab) {
        int b = context.groupIdx;
        int tid = context.localIdx;
        int localSz = context.localGroupSizeX;
        float[] vals = context.allocateFloatLocalArray(256);
        int[] idxs = context.allocateIntLocalArray(256);

        int base = b * vocab;
        float best = Float.NEGATIVE_INFINITY;
        int bestIdx = 0;
        for (int i = tid; i < vocab; i += localSz) {
            float v = logits.get(base + i);
            if (v > best) {
                best = v;
                bestIdx = i;
            }
        }
        vals[tid] = best;
        idxs[tid] = bestIdx;
        context.localBarrier();

        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid + s] > vals[tid]) {
                    vals[tid] = vals[tid + s];
                    idxs[tid] = idxs[tid + s];
                }
            }
            context.localBarrier();
        }
        if (tid == 0) {
            outTokens.set(b, idxs[0]);
        }
    }

    // ── FP32 → FP16 cast ─────────────────────────────────────────────────────
    /**
     * Elementwise cast of a batch buffer to FP16, for a family whose next kernel takes
     * half-precision input that no fused producer already writes (Gemma4's per-layer embedding
     * scale).
     *
     * <p>Worker: B*dim global threads, localSize=256.
     */
    public static void batchedConvertFP32toFP16(
            KernelContext context, FloatArray in, HalfFloatArray out) {
        int gid = context.globalIdx;
        out.set(gid, new HalfFloat(in.get(gid)));
    }

    // ── SwiGLU over the packed gate/up buffer, emitting FP16 ─────────────────

    /**
     * Fused SiLU(gate) * up over the PACKED [gate | up] GEMM output, emitting FP16 (the A operand
     * of the W2 GEMM).
     *
     * <p>Worker: B*hiddenDim global threads, localSize=256.
     */
    public static void batchedFFNSwiGLUFP16Packed(
            KernelContext context,
            HalfFloatArray wrapHbFP16Batch,
            FloatArray gateUpResult,
            int hiddenDim) {
        int gid = context.globalIdx;
        int b = gid / hiddenDim;
        int i = gid % hiddenDim;
        int rowBase = b * 2 * hiddenDim;
        float g = gateUpResult.get(rowBase + i);
        float u = gateUpResult.get(rowBase + hiddenDim + i);
        float silu = g / (1.0f + TornadoMath.exp(-g));
        wrapHbFP16Batch.set(gid, new HalfFloat(silu * u));
    }

    // ── Q8_0 tensor-core GEMMs (W8A16) ───────────────────────────────────────
    //
    // Q8_0 weights stay quantized in global memory (34-byte GGUF blocks:
    // FP16 scale + 32 int8 quants) and are dequantized to FP16 *in the
    // register-staging step* of the software pipeline, then flow through the
    // identical ldmatrix + m16n8k16 path as the FP16 GEMMs. This halves the
    // weight-side memory traffic relative to FP16 while reusing the proven
    // FP16 tensor-core pipeline. A true INT8 (m16n8k32) path would need
    // per-k-block accumulator rescaling — blocked on fragment-level access
    // in the TornadoVM intrinsics; see paper future work.
    //
    // Note: BK = 16, so a K-step never straddles a Q8_0 block boundary
    // (32 | K), and each staged pair reads exactly one scale per column.

    /**
     * The IEEE half-precision bit pattern of {@code v}, round-to-nearest-even, computed with plain
     * arithmetic.
     *
     * <p>Deliberately not {@code new HalfFloat(v).getHalfFloatValue()}. That is what the CUDA
     * backend cannot lower inside the tensor-core kernels below — "unimplemented: address origin
     * unimplemented: ...calc.MulNode" in {@code CUDAAddressLowering.lower} — which is why Q8_0
     * batched prefill failed on CUDA while the same kernels compiled elsewhere. The constructor
     * lowers fine in the simpler kernels that store straight into a {@code HalfFloatArray}; here
     * its result is read back as a short and packed into a register pair, and the object does not
     * escape-analyse away.
     *
     * <p>Only what a dequantized Q8_0 weight can be is handled: finite values, including subnormals
     * and zero. Infinities and NaNs are not, because a block scale times an int8 is neither.
     * Negative zero comes back as positive zero — distinguishing it needs a reciprocal in the inner
     * loop to buy a value no Q8_0 product produces, and the two are interchangeable in the GEMM.
     * Overflow beyond the half-precision range saturates to the largest finite half rather than
     * producing an infinity, which no Q8_0 weight reaches in practice.
     */
    static int fp16BitsOf(float v) {
        int sign = 0;
        float a = v;
        if (a < 0.0f) {
            sign = 0x8000;
            a = -a;
        }
        if (a < 2.9802322E-8f) { // below half the smallest subnormal: rounds to zero
            return sign;
        }
        if (a >= 65520.0f) { // rounds above the largest finite half
            return sign | 0x7BFF;
        }
        // Binary search for the unbiased exponent e with 2^e <= a < 2^(e+1), over the range a
        // subnormal-to-max half can occupy. Fixed comparisons rather than a loop: a data-dependent
        // loop compiles but is far slower on the device, and this sits in the inner K-loop.
        int e = -24;
        float p = 5.9604645E-8f; // 2^-24
        if (a >= p * 4294967296.0f) { // 2^32
            e += 32;
            p *= 4294967296.0f;
        }
        if (a >= p * 65536.0f) {
            e += 16;
            p *= 65536.0f;
        }
        if (a >= p * 256.0f) {
            e += 8;
            p *= 256.0f;
        }
        if (a >= p * 16.0f) {
            e += 4;
            p *= 16.0f;
        }
        if (a >= p * 4.0f) {
            e += 2;
            p *= 4.0f;
        }
        if (a >= p * 2.0f) {
            e += 1;
            p *= 2.0f;
        }
        // Scale so that the value to round is an integer count of half-precision steps: normals
        // carry an implicit leading 1 and 10 stored mantissa bits, subnormals a fixed 2^-24 step.
        float scaled;
        if (e < -14) {
            scaled = a * 16777216.0f; // a / 2^-24
        } else {
            scaled = a / p * 1024.0f;
        }
        int unit = (int) scaled;
        float frac = scaled - unit;
        if (frac > 0.5f) {
            unit++;
        } else if (frac == 0.5f && (unit & 1) != 0) {
            unit++; // ties to even
        }
        if (e < -14) {
            return sign | unit; // subnormal, or the carry into the smallest normal, which the
            // encoding already places correctly
        }
        int mantissa = unit - 1024;
        int exponent = e + 15;
        if (mantissa == 1024) { // rounding carried into the next binade
            mantissa = 0;
            exponent++;
        }
        if (exponent >= 31) {
            return sign | 0x7BFF;
        }
        return sign | (exponent << 10) | mantissa;
    }

    /**
     * Dequantizes and packs two vertically-adjacent (col, col+1) Q8_0 weight elements at depth k
     * into one int of two FP16 values, matching the bTile layout expected by mmaLoadB. Leaf helper;
     * inlined by the JIT.
     */
    private static int packQ8Halves(ByteArray w, int col, int k, int blocksPerRow) {
        int kBlock = k >>> 5; // k / 32
        int kIn = k & 31; // k % 32
        int off0 = (col * blocksPerRow + kBlock) * 34;
        int off1 = off0 + blocksPerRow * 34; // column col+1, same k-block
        float v0 = w.getHalfFloat(off0).getFloat32() * w.get(off0 + 2 + kIn);
        float v1 = w.getHalfFloat(off1).getFloat32() * w.get(off1 + 2 + kIn);
        int lo = fp16BitsOf(v0);
        int hi = fp16BitsOf(v1);
        return lo | (hi << 16);
    }

    /**
     * Tensor-core GEMM with Q8_0 weights: C[M,N] (FP32) = A[M,K] (FP16) × B[N,K] (Q8_0 blocks,
     * row-major). Same tiling, pipeline, and constraints as {@link #gemmMMA}.
     */
    public static void gemmMMAQ8(
            KernelContext ctx, HalfFloatArray A, ByteArray B, FloatArray C, int M, int N, int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int blocksPerRow = K / 32;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B staging keeps (col, k) coordinates explicit for block-offset math.
        int bIdx0 = tid;
        int bCol0 = blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1);
        int bK0 = (bIdx0 & 63) >>> 2;
        int bIdx1 = tid + 256;
        int bCol1 = blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1);
        int bK1 = (bIdx1 & 63) >>> 2;
        int bIdx2 = tid + 512;
        int bCol2 = blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1);
        int bK2 = (bIdx2 & 63) >>> 2;
        int bIdx3 = tid + 768;
        int bCol3 = blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1);
        int bK3 = (bIdx3 & 63) >>> 2;

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packQ8Halves(B, bCol0, bK0, blocksPerRow);
        int bReg1 = packQ8Halves(B, bCol1, bK1, blocksPerRow);
        int bReg2 = packQ8Halves(B, bCol2, bK2, blocksPerRow);
        int bReg3 = packQ8Halves(B, bCol3, bK3, blocksPerRow);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packQ8Halves(B, bCol0, kOff + bK0, blocksPerRow);
                bReg1 = packQ8Halves(B, bCol1, kOff + bK1, blocksPerRow);
                bReg2 = packQ8Halves(B, bCol2, kOff + bK2, blocksPerRow);
                bReg3 = packQ8Halves(B, bCol3, kOff + bK3, blocksPerRow);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    /**
     * Fused QKV tensor-core GEMM with Q8_0 weights into the PACKED output: qkvOut[M, dim+2*kvDim] =
     * A[M,K] (FP16) × [Wq | Wk | Wv] (Q8_0, [N_i,K] row-major). Same layout, fusion, and
     * constraints as {@link #gemmMMAQKV}.
     */
    public static void gemmMMAQKVQ8(
            KernelContext ctx,
            HalfFloatArray A,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            FloatArray qkvOut,
            int M,
            int dim,
            int kvDim,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int qkvStride = dim + 2 * kvDim;
        int blocksPerRow = K / 32;

        int wColBase = blockCol;
        if (blockCol >= dim) wColBase -= dim;
        if (blockCol >= dim + kvDim) wColBase -= kvDim;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int bCol0 = wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1);
        int bK0 = (bIdx0 & 63) >>> 2;
        int bIdx1 = tid + 256;
        int bCol1 = wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1);
        int bK1 = (bIdx1 & 63) >>> 2;
        int bIdx2 = tid + 512;
        int bCol2 = wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1);
        int bK2 = (bIdx2 & 63) >>> 2;
        int bIdx3 = tid + 768;
        int bCol3 = wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1);
        int bK3 = (bIdx3 & 63) >>> 2;

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < dim) {
            bReg0 = packQ8Halves(wq, bCol0, bK0, blocksPerRow);
            bReg1 = packQ8Halves(wq, bCol1, bK1, blocksPerRow);
            bReg2 = packQ8Halves(wq, bCol2, bK2, blocksPerRow);
            bReg3 = packQ8Halves(wq, bCol3, bK3, blocksPerRow);
        } else if (blockCol < dim + kvDim) {
            bReg0 = packQ8Halves(wk, bCol0, bK0, blocksPerRow);
            bReg1 = packQ8Halves(wk, bCol1, bK1, blocksPerRow);
            bReg2 = packQ8Halves(wk, bCol2, bK2, blocksPerRow);
            bReg3 = packQ8Halves(wk, bCol3, bK3, blocksPerRow);
        } else {
            bReg0 = packQ8Halves(wv, bCol0, bK0, blocksPerRow);
            bReg1 = packQ8Halves(wv, bCol1, bK1, blocksPerRow);
            bReg2 = packQ8Halves(wv, bCol2, bK2, blocksPerRow);
            bReg3 = packQ8Halves(wv, bCol3, bK3, blocksPerRow);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < dim) {
                    bReg0 = packQ8Halves(wq, bCol0, kOff + bK0, blocksPerRow);
                    bReg1 = packQ8Halves(wq, bCol1, kOff + bK1, blocksPerRow);
                    bReg2 = packQ8Halves(wq, bCol2, kOff + bK2, blocksPerRow);
                    bReg3 = packQ8Halves(wq, bCol3, kOff + bK3, blocksPerRow);
                } else if (blockCol < dim + kvDim) {
                    bReg0 = packQ8Halves(wk, bCol0, kOff + bK0, blocksPerRow);
                    bReg1 = packQ8Halves(wk, bCol1, kOff + bK1, blocksPerRow);
                    bReg2 = packQ8Halves(wk, bCol2, kOff + bK2, blocksPerRow);
                    bReg3 = packQ8Halves(wk, bCol3, kOff + bK3, blocksPerRow);
                } else {
                    bReg0 = packQ8Halves(wv, bCol0, kOff + bK0, blocksPerRow);
                    bReg1 = packQ8Halves(wv, bCol1, kOff + bK1, blocksPerRow);
                    bReg2 = packQ8Halves(wv, bCol2, kOff + bK2, blocksPerRow);
                    bReg3 = packQ8Halves(wv, bCol3, kOff + bK3, blocksPerRow);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, qkvOut, rBase + 0, cBase + 0, qkvStride);
        ctx.mmaStore(c01, qkvOut, rBase + 0, cBase + 8, qkvStride);
        ctx.mmaStore(c02, qkvOut, rBase + 0, cBase + 16, qkvStride);
        ctx.mmaStore(c03, qkvOut, rBase + 0, cBase + 24, qkvStride);
        ctx.mmaStore(c04, qkvOut, rBase + 0, cBase + 32, qkvStride);
        ctx.mmaStore(c05, qkvOut, rBase + 0, cBase + 40, qkvStride);
        ctx.mmaStore(c06, qkvOut, rBase + 0, cBase + 48, qkvStride);
        ctx.mmaStore(c07, qkvOut, rBase + 0, cBase + 56, qkvStride);
        ctx.mmaStore(c10, qkvOut, rBase + 16, cBase + 0, qkvStride);
        ctx.mmaStore(c11, qkvOut, rBase + 16, cBase + 8, qkvStride);
        ctx.mmaStore(c12, qkvOut, rBase + 16, cBase + 16, qkvStride);
        ctx.mmaStore(c13, qkvOut, rBase + 16, cBase + 24, qkvStride);
        ctx.mmaStore(c14, qkvOut, rBase + 16, cBase + 32, qkvStride);
        ctx.mmaStore(c15, qkvOut, rBase + 16, cBase + 40, qkvStride);
        ctx.mmaStore(c16, qkvOut, rBase + 16, cBase + 48, qkvStride);
        ctx.mmaStore(c17, qkvOut, rBase + 16, cBase + 56, qkvStride);
    }

    /**
     * Fused W1/W3 (gate/up) tensor-core GEMM with Q8_0 weights into the PACKED output gateUpOut[M,
     * 2*hidDim]. Same layout and constraints as {@link #gemmMMAGateUp}.
     */
    public static void gemmMMAGateUpQ8(
            KernelContext ctx,
            HalfFloatArray A,
            ByteArray w1,
            ByteArray w3,
            FloatArray gateUpOut,
            int M,
            int hidDim,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int outStride = 2 * hidDim;
        int blocksPerRow = K / 32;

        int wColBase = (blockCol < hidDim) ? blockCol : (blockCol - hidDim);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int bCol0 = wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1);
        int bK0 = (bIdx0 & 63) >>> 2;
        int bIdx1 = tid + 256;
        int bCol1 = wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1);
        int bK1 = (bIdx1 & 63) >>> 2;
        int bIdx2 = tid + 512;
        int bCol2 = wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1);
        int bK2 = (bIdx2 & 63) >>> 2;
        int bIdx3 = tid + 768;
        int bCol3 = wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1);
        int bK3 = (bIdx3 & 63) >>> 2;

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < hidDim) {
            bReg0 = packQ8Halves(w1, bCol0, bK0, blocksPerRow);
            bReg1 = packQ8Halves(w1, bCol1, bK1, blocksPerRow);
            bReg2 = packQ8Halves(w1, bCol2, bK2, blocksPerRow);
            bReg3 = packQ8Halves(w1, bCol3, bK3, blocksPerRow);
        } else {
            bReg0 = packQ8Halves(w3, bCol0, bK0, blocksPerRow);
            bReg1 = packQ8Halves(w3, bCol1, bK1, blocksPerRow);
            bReg2 = packQ8Halves(w3, bCol2, bK2, blocksPerRow);
            bReg3 = packQ8Halves(w3, bCol3, bK3, blocksPerRow);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < hidDim) {
                    bReg0 = packQ8Halves(w1, bCol0, kOff + bK0, blocksPerRow);
                    bReg1 = packQ8Halves(w1, bCol1, kOff + bK1, blocksPerRow);
                    bReg2 = packQ8Halves(w1, bCol2, kOff + bK2, blocksPerRow);
                    bReg3 = packQ8Halves(w1, bCol3, kOff + bK3, blocksPerRow);
                } else {
                    bReg0 = packQ8Halves(w3, bCol0, kOff + bK0, blocksPerRow);
                    bReg1 = packQ8Halves(w3, bCol1, kOff + bK1, blocksPerRow);
                    bReg2 = packQ8Halves(w3, bCol2, kOff + bK2, blocksPerRow);
                    bReg3 = packQ8Halves(w3, bCol3, kOff + bK3, blocksPerRow);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, gateUpOut, rBase + 0, cBase + 0, outStride);
        ctx.mmaStore(c01, gateUpOut, rBase + 0, cBase + 8, outStride);
        ctx.mmaStore(c02, gateUpOut, rBase + 0, cBase + 16, outStride);
        ctx.mmaStore(c03, gateUpOut, rBase + 0, cBase + 24, outStride);
        ctx.mmaStore(c04, gateUpOut, rBase + 0, cBase + 32, outStride);
        ctx.mmaStore(c05, gateUpOut, rBase + 0, cBase + 40, outStride);
        ctx.mmaStore(c06, gateUpOut, rBase + 0, cBase + 48, outStride);
        ctx.mmaStore(c07, gateUpOut, rBase + 0, cBase + 56, outStride);
        ctx.mmaStore(c10, gateUpOut, rBase + 16, cBase + 0, outStride);
        ctx.mmaStore(c11, gateUpOut, rBase + 16, cBase + 8, outStride);
        ctx.mmaStore(c12, gateUpOut, rBase + 16, cBase + 16, outStride);
        ctx.mmaStore(c13, gateUpOut, rBase + 16, cBase + 24, outStride);
        ctx.mmaStore(c14, gateUpOut, rBase + 16, cBase + 32, outStride);
        ctx.mmaStore(c15, gateUpOut, rBase + 16, cBase + 40, outStride);
        ctx.mmaStore(c16, gateUpOut, rBase + 16, cBase + 48, outStride);
        ctx.mmaStore(c17, gateUpOut, rBase + 16, cBase + 56, outStride);
    }

    // @formatter:on

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for <b>F32</b> weights, one workgroup per (row, output
     * row).
     *
     * <p>The batched counterpart of {@code TransformerComputeKernelsLayered.matrixVectorGeneric}
     * over a float weight matrix — what an SSM projection needs, whose weights are F32 in every
     * file that carries them.
     */
    // @formatter:on
    public static void batchedMatVecF32(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            FloatArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / d;
        int rowIdx = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            partial += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            outputBatch.set(batchIdx * d + rowIdx, localSum[0]);
        }
    }

    // @formatter:off
    /**
     * {@link #batchedMatVecF32} with one warp per output instead of a 128-lane workgroup: four
     * warps a block, each computing one (batch row, output row), no shared reduction.
     *
     * <p>Lane {@code l} of the warp plays the control's lanes {@code l, l + 32, l + 64, l + 96}:
     * four partial sums, each over the input indices that lane walked ({@code l + 128 i}, {@code l
     * + 32 + 128 i}, ...), each from zero with the same multiply-add expression, so each partial
     * carries the bits the control's lane produced. The control then folds its 128 partials as a
     * shared tree: stride 64 forms {@code p[i] + p[i + 64]}, stride 32 forms {@code (p[i] + p[i +
     * 64]) + (p[i + 32] + p[i + 96])} for {@code i < 32}, and strides 16, 8, 4, 2, 1 add entry
     * {@code i + stride} into entry {@code i}. Here {@code combined = (partial0 + partial2) +
     * (partial1 + partial3)} reproduces the first two steps in the same operand order, and five
     * shuffle-down additions (offsets 16, 8, 4, 2, 1, lane {@code i} adding lane {@code i +
     * offset}) reproduce the rest; lane zero holds the control's {@code localSum[0]} and writes it.
     * Every operand FP32; the output is raw-bit equal to the control's on the CUDA backend, where
     * the shuffle is verified.
     *
     * <p>Worker: {@code activeRowsPadded * d * 32} lanes, local 128 (four outputs per block).
     */
    // @formatter:on
    public static void batchedMatVecF32Warp(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            FloatArray w,
            int n,
            int d,
            int activeRows) {
        int lane = context.localIdx & 31;
        int output = (context.groupIdx << 2) + (context.localIdx >> 5);
        int batchIdx = output / d;
        int rowIdx = output - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        for (int j = lane; j < n; j += 128) {
            partial0 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 32; j < n; j += 128) {
            partial1 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 64; j < n; j += 128) {
            partial2 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        for (int j = lane + 96; j < n; j += 128) {
            partial3 += w.get(rowOff + j) * inputBatch.get(inputOff + j);
        }
        float combined = (partial0 + partial2) + (partial1 + partial3);
        combined += context.simdShuffleDown(combined, 16);
        combined += context.simdShuffleDown(combined, 8);
        combined += context.simdShuffleDown(combined, 4);
        combined += context.simdShuffleDown(combined, 2);
        combined += context.simdShuffleDown(combined, 1);
        if (lane == 0) {
            outputBatch.set(batchIdx * d + rowIdx, combined);
        }
    }

    /** Rows and outputs of one warp's tile in {@link #batchedMatVecF32WarpTile}. */
    public static final int MATVEC_TILE = 4;

    // @formatter:off
    /**
     * {@link #batchedMatVecF32Warp} with a warp computing a {@link #MATVEC_TILE} x {@link
     * #MATVEC_TILE} tile of (batch row, output row) pairs instead of one, so each input value and
     * each weight a lane loads serves four multiply-adds instead of one.
     *
     * <p>Every (row, output) pair keeps the warp kernel's arithmetic exactly: lane {@code l}
     * accumulates the same four partial sums over the same input indices in the same order, from
     * zero, with the same multiply-add expression — only interleaved across the tile's sixteen
     * pairs inside the {@code i} loop — then the same {@code (p0 + p2) + (p1 + p3)} and the same
     * five shuffle-down additions. The outputs are raw-bit equal to the warp kernel's (asserted by
     * the test). Requires {@code d % 4 == 0}; rows past {@code activeRows} are neither read nor
     * written, and a warp whose whole row tile is past them returns.
     *
     * <p>Worker: {@code ceil(activeRowsPadded / 4) * (d / 4) * 32} lanes, local 128 (four tiles per
     * block).
     */
    // @formatter:on
    public static void batchedMatVecF32WarpTile(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            FloatArray w,
            int n,
            int d,
            int activeRows) {
        int lane = context.localIdx & 31;
        int tile = (context.groupIdx << 2) + (context.localIdx >> 5);
        int outputTiles = d / MATVEC_TILE;
        int rowTile = tile / outputTiles;
        int row0 = rowTile * MATVEC_TILE;
        int out0 = (tile - rowTile * outputTiles) * MATVEC_TILE;
        if (row0 >= activeRows) {
            return;
        }
        // Rows past the active count read row zero (finite, in bounds) and are not written.
        int r1 = row0 + 1 < activeRows ? row0 + 1 : row0;
        int r2 = row0 + 2 < activeRows ? row0 + 2 : row0;
        int r3 = row0 + 3 < activeRows ? row0 + 3 : row0;
        int in0 = row0 * n;
        int in1 = r1 * n;
        int in2 = r2 * n;
        int in3 = r3 * n;
        int w0 = out0 * n;
        int w1 = w0 + n;
        int w2 = w1 + n;
        int w3 = w2 + n;

        float p00a = 0.0f, p00b = 0.0f, p00c = 0.0f, p00d = 0.0f;
        float p01a = 0.0f, p01b = 0.0f, p01c = 0.0f, p01d = 0.0f;
        float p02a = 0.0f, p02b = 0.0f, p02c = 0.0f, p02d = 0.0f;
        float p03a = 0.0f, p03b = 0.0f, p03c = 0.0f, p03d = 0.0f;
        float p10a = 0.0f, p10b = 0.0f, p10c = 0.0f, p10d = 0.0f;
        float p11a = 0.0f, p11b = 0.0f, p11c = 0.0f, p11d = 0.0f;
        float p12a = 0.0f, p12b = 0.0f, p12c = 0.0f, p12d = 0.0f;
        float p13a = 0.0f, p13b = 0.0f, p13c = 0.0f, p13d = 0.0f;
        float p20a = 0.0f, p20b = 0.0f, p20c = 0.0f, p20d = 0.0f;
        float p21a = 0.0f, p21b = 0.0f, p21c = 0.0f, p21d = 0.0f;
        float p22a = 0.0f, p22b = 0.0f, p22c = 0.0f, p22d = 0.0f;
        float p23a = 0.0f, p23b = 0.0f, p23c = 0.0f, p23d = 0.0f;
        float p30a = 0.0f, p30b = 0.0f, p30c = 0.0f, p30d = 0.0f;
        float p31a = 0.0f, p31b = 0.0f, p31c = 0.0f, p31d = 0.0f;
        float p32a = 0.0f, p32b = 0.0f, p32c = 0.0f, p32d = 0.0f;
        float p33a = 0.0f, p33b = 0.0f, p33c = 0.0f, p33d = 0.0f;

        // Partial a: indices lane + 128 i; b: lane + 32 + 128 i; c: + 64; d: + 96 — the warp
        // kernel's four loops, fused over i, each pair's product added to its own partial.
        for (int j = lane; j < n; j += 128) {
            float wa0 = w.get(w0 + j);
            float wa1 = w.get(w1 + j);
            float wa2 = w.get(w2 + j);
            float wa3 = w.get(w3 + j);
            float x0 = inputBatch.get(in0 + j);
            float x1 = inputBatch.get(in1 + j);
            float x2 = inputBatch.get(in2 + j);
            float x3 = inputBatch.get(in3 + j);
            p00a += wa0 * x0;
            p01a += wa1 * x0;
            p02a += wa2 * x0;
            p03a += wa3 * x0;
            p10a += wa0 * x1;
            p11a += wa1 * x1;
            p12a += wa2 * x1;
            p13a += wa3 * x1;
            p20a += wa0 * x2;
            p21a += wa1 * x2;
            p22a += wa2 * x2;
            p23a += wa3 * x2;
            p30a += wa0 * x3;
            p31a += wa1 * x3;
            p32a += wa2 * x3;
            p33a += wa3 * x3;
        }
        for (int j = lane + 32; j < n; j += 128) {
            float wa0 = w.get(w0 + j);
            float wa1 = w.get(w1 + j);
            float wa2 = w.get(w2 + j);
            float wa3 = w.get(w3 + j);
            float x0 = inputBatch.get(in0 + j);
            float x1 = inputBatch.get(in1 + j);
            float x2 = inputBatch.get(in2 + j);
            float x3 = inputBatch.get(in3 + j);
            p00b += wa0 * x0;
            p01b += wa1 * x0;
            p02b += wa2 * x0;
            p03b += wa3 * x0;
            p10b += wa0 * x1;
            p11b += wa1 * x1;
            p12b += wa2 * x1;
            p13b += wa3 * x1;
            p20b += wa0 * x2;
            p21b += wa1 * x2;
            p22b += wa2 * x2;
            p23b += wa3 * x2;
            p30b += wa0 * x3;
            p31b += wa1 * x3;
            p32b += wa2 * x3;
            p33b += wa3 * x3;
        }
        for (int j = lane + 64; j < n; j += 128) {
            float wa0 = w.get(w0 + j);
            float wa1 = w.get(w1 + j);
            float wa2 = w.get(w2 + j);
            float wa3 = w.get(w3 + j);
            float x0 = inputBatch.get(in0 + j);
            float x1 = inputBatch.get(in1 + j);
            float x2 = inputBatch.get(in2 + j);
            float x3 = inputBatch.get(in3 + j);
            p00c += wa0 * x0;
            p01c += wa1 * x0;
            p02c += wa2 * x0;
            p03c += wa3 * x0;
            p10c += wa0 * x1;
            p11c += wa1 * x1;
            p12c += wa2 * x1;
            p13c += wa3 * x1;
            p20c += wa0 * x2;
            p21c += wa1 * x2;
            p22c += wa2 * x2;
            p23c += wa3 * x2;
            p30c += wa0 * x3;
            p31c += wa1 * x3;
            p32c += wa2 * x3;
            p33c += wa3 * x3;
        }
        for (int j = lane + 96; j < n; j += 128) {
            float wa0 = w.get(w0 + j);
            float wa1 = w.get(w1 + j);
            float wa2 = w.get(w2 + j);
            float wa3 = w.get(w3 + j);
            float x0 = inputBatch.get(in0 + j);
            float x1 = inputBatch.get(in1 + j);
            float x2 = inputBatch.get(in2 + j);
            float x3 = inputBatch.get(in3 + j);
            p00d += wa0 * x0;
            p01d += wa1 * x0;
            p02d += wa2 * x0;
            p03d += wa3 * x0;
            p10d += wa0 * x1;
            p11d += wa1 * x1;
            p12d += wa2 * x1;
            p13d += wa3 * x1;
            p20d += wa0 * x2;
            p21d += wa1 * x2;
            p22d += wa2 * x2;
            p23d += wa3 * x2;
            p30d += wa0 * x3;
            p31d += wa1 * x3;
            p32d += wa2 * x3;
            p33d += wa3 * x3;
        }

        float s00 = warpTreeF32(context, (p00a + p00c) + (p00b + p00d));
        float s01 = warpTreeF32(context, (p01a + p01c) + (p01b + p01d));
        float s02 = warpTreeF32(context, (p02a + p02c) + (p02b + p02d));
        float s03 = warpTreeF32(context, (p03a + p03c) + (p03b + p03d));
        float s10 = warpTreeF32(context, (p10a + p10c) + (p10b + p10d));
        float s11 = warpTreeF32(context, (p11a + p11c) + (p11b + p11d));
        float s12 = warpTreeF32(context, (p12a + p12c) + (p12b + p12d));
        float s13 = warpTreeF32(context, (p13a + p13c) + (p13b + p13d));
        float s20 = warpTreeF32(context, (p20a + p20c) + (p20b + p20d));
        float s21 = warpTreeF32(context, (p21a + p21c) + (p21b + p21d));
        float s22 = warpTreeF32(context, (p22a + p22c) + (p22b + p22d));
        float s23 = warpTreeF32(context, (p23a + p23c) + (p23b + p23d));
        float s30 = warpTreeF32(context, (p30a + p30c) + (p30b + p30d));
        float s31 = warpTreeF32(context, (p31a + p31c) + (p31b + p31d));
        float s32 = warpTreeF32(context, (p32a + p32c) + (p32b + p32d));
        float s33 = warpTreeF32(context, (p33a + p33c) + (p33b + p33d));
        if (lane == 0) {
            int o0 = row0 * d + out0;
            outputBatch.set(o0, s00);
            outputBatch.set(o0 + 1, s01);
            outputBatch.set(o0 + 2, s02);
            outputBatch.set(o0 + 3, s03);
            if (row0 + 1 < activeRows) {
                int o1 = o0 + d;
                outputBatch.set(o1, s10);
                outputBatch.set(o1 + 1, s11);
                outputBatch.set(o1 + 2, s12);
                outputBatch.set(o1 + 3, s13);
            }
            if (row0 + 2 < activeRows) {
                int o2 = o0 + 2 * d;
                outputBatch.set(o2, s20);
                outputBatch.set(o2 + 1, s21);
                outputBatch.set(o2 + 2, s22);
                outputBatch.set(o2 + 3, s23);
            }
            if (row0 + 3 < activeRows) {
                int o3 = o0 + 3 * d;
                outputBatch.set(o3, s30);
                outputBatch.set(o3 + 1, s31);
                outputBatch.set(o3 + 2, s32);
                outputBatch.set(o3 + 3, s33);
            }
        }
    }

    /** The warp kernel's five shuffle-down additions; lane zero holds the sum. */
    private static float warpTreeF32(KernelContext context, float combined) {
        combined += context.simdShuffleDown(combined, 16);
        combined += context.simdShuffleDown(combined, 8);
        combined += context.simdShuffleDown(combined, 4);
        combined += context.simdShuffleDown(combined, 2);
        combined += context.simdShuffleDown(combined, 1);
        return combined;
    }
}
