package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

// @formatter:off
public class Qwen3Kernels {

    /**
     * For explicit copy out useful in debugging. With this kernel we can store the values of an
     * array to a tmp buffer at a timing of interest. In the end of the taskgraph we copy out the
     * tmp buffer to inspect the array values at the timing of interest.
     *
     * @param srcBuffer the array we want to inspect.
     * @param dstBuffer the tmp buffer.
     */
    public static void dbgCopy(FloatArray srcBuffer, FloatArray dstBuffer) {
        for (@Parallel int i = 0; i < srcBuffer.getSize(); i++) {
            dstBuffer.set(i, srcBuffer.get(i));
        }
    }

    /**
     * RmsNorm with parallel offset: The following 3 kernels implement rmsnorm in offset range in
     * parallel for qCur and Kcur rmsnorm calculations.
     *
     * <p>Step 1: Reduction. This kernel implements rmsnorm in offset range in parallel for qCur and
     * Kcur rmsnorm calculations.
     */
    public static void rmsnormReductionWithParallelOffset(
            KernelContext context, FloatArray output, FloatArray x, int localMemSize) {

        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId, localX[0]);
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * <p>Step 2: Combines partial reduction outputs and computes final normalization.
     */
    public static void rmsnormFinalNormalizationWithParallelOffset(
            KernelContext context,
            FloatArray output, // size should be related to offsetIndex
            int offsetIndex, // = config.numberOfHeads()
            int size,
            float ermsNorm) {

        int gid = context.globalIdx;

        // Only the index threads need to perform this calculation
        if (gid < offsetIndex) {
            // Combine partial sums from all workgroups
            float ss = output.get(gid);

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            // in place
            output.set(gid, ss); // Store the final scale factor
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * <p>Step 3: perform mapIndex operation.
     */
    public static void rmsnormMapIndexInPlaceWithParallelOffset(
            KernelContext context, FloatArray out, FloatArray weights, int size, FloatArray ss) {

        int gid = context.globalIdx;
        int groupId = context.groupIdx;

        float finalss = ss.get(groupId);

        if (gid < out.getSize()) { // TODO: check if redundant
            float a = weights.get(gid % size);
            float b = finalss * out.get(gid);
            out.set(gid, a * b);
        }
    }

    /**
     * Fused Q/K/V matrix-vector multiplication for Qwen3 GQA. Q has full head dimension, K/V have
     * reduced KV head dimension.
     *
     * <p>Workgroup assignment: - rowId [0, qDim): Q projection - rowId [qDim, qDim+kvDim): K
     * projection - rowId [qDim+kvDim, qDim+2*kvDim): V projection
     */
    public static void fusedQKVMatmul(
            KernelContext context,
            FloatArray x, // input vector
            FloatArray q, // output Q
            FloatArray k, // output K
            FloatArray v, // output V
            HalfFloatArray wq, // Q weight matrix
            HalfFloatArray wk, // K weight matrix
            HalfFloatArray wv, // V weight matrix
            int inputDim, // input dimension (config.dim())
            int qDim, // Q output dimension
            int kvDim, // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection ==========
            int rowOffset = rowId * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                Half2 pair = wq.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * x.get(j);
                partialSum += Half2.highFloat(pair) * x.get(j + 1);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSum[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection ==========
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                Half2 pair = wk.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * x.get(j);
                partialSum += Half2.highFloat(pair) * x.get(j + 1);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSum[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection ==========
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                Half2 pair = wv.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * x.get(j);
                partialSum += Half2.highFloat(pair) * x.get(j + 1);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSum[0]);
            }
        }
    }

    /**
     * Fused RMSNorm apply + Q/K/V projection for Qwen3 GQA. Eliminates intermediate wrapXb buffer
     * write/read.
     */
    public static void fusedRmsNormQKVMatmul(
            KernelContext context,
            FloatArray x, // raw input (FP32)
            FloatArray q, // output Q
            FloatArray k, // output K
            FloatArray v, // output V
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            HalfFloatArray wq, // Q weight matrix
            HalfFloatArray wk, // K weight matrix
            HalfFloatArray wv, // V weight matrix
            int inputDim, // input dimension (config.dim())
            int qDim, // Q output dimension
            int kvDim, // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float scale = rmsScale.get(0);

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection with inline normalization ==========
            int rowOffset = rowId * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                float normalized0 = rmsWeights.get(j) * scale * x.get(j);
                float normalized1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                Half2 pair = wq.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * normalized0;
                partialSum += Half2.highFloat(pair) * normalized1;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSum[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection with inline normalization ==========
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                float normalized0 = rmsWeights.get(j) * scale * x.get(j);
                float normalized1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                Half2 pair = wk.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * normalized0;
                partialSum += Half2.highFloat(pair) * normalized1;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSum[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection with inline normalization ==========
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads (single 32-bit load per pair); inputDim is even.
            for (int j = localId * 2; j < inputDim; j += localWorkGroupSize * 2) {
                float normalized0 = rmsWeights.get(j) * scale * x.get(j);
                float normalized1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                Half2 pair = wv.getHalf2(rowOffset + j);
                partialSum += Half2.lowFloat(pair) * normalized0;
                partialSum += Half2.highFloat(pair) * normalized1;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSum[0]);
            }
        }
    }

    /**
     * Warp-shuffle variant of {@link #fusedRmsNormQKVMatmul}. Assumes a 32-lane workgroup per
     * output row. Reduces each row's dot product with {@code simdShuffleDown} (no shared-memory
     * barriers), matching the reduction strategy of llama.cpp's {@code mul_mat_vec}. Used on CUDA
     * only (see {@code SchedulerDetectionService.isWarpShuffleSupported}); OpenCL miscompiles the
     * shuffle.
     */
    public static void fusedRmsNormQKVMatmulWarp(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        float scale = rmsScale.get(0);

        // Three independent blocks (mirrors the shared-memory variant): each does its own loop +
        // warp reduction + write. Keeping the reduction inside each branch avoids carrying the
        // partial across control flow, which produced incorrect PTX for the shuffle.
        if (rowId < qDim) {
            int rowOffset = rowId * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                partialSum +=
                        wq.get(rowOffset + j).getFloat32() * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                q.set(rowId, partialSum);
            }
        } else if (rowId < qDim + kvDim) {
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                partialSum +=
                        wk.get(rowOffset + j).getFloat32() * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                k.set(kRow, partialSum);
            }
        } else if (rowId < qDim + 2 * kvDim) {
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                partialSum +=
                        wv.get(rowOffset + j).getFloat32() * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                v.set(vRow, partialSum);
            }
        }
    }

    /**
     * Fused RMSNorm apply + Q/K/V projection for Qwen3 GQA with Q8_0 quantized weights. Uses the
     * same Q8_0 block structure as matrixVectorRowMajorOptimizedQ8_0Byte.
     */
    public static void fusedRmsNormQKVMatmulQ8_0(
            KernelContext context,
            FloatArray x, // raw input (FP32)
            FloatArray q, // output Q
            FloatArray k, // output K
            FloatArray v, // output V
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            ByteArray wq, // Q weight matrix (Q8_0)
            ByteArray wk, // K weight matrix (Q8_0)
            ByteArray wv, // V weight matrix (Q8_0)
            int inputDim, // input dimension (config.dim())
            int qDim, // Q output dimension
            int kvDim, // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection with inline normalization ==========
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = rowId * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                // Load scale for this block
                HalfFloat blockScale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                // Load 4 consecutive quantized values
                int quantsOffset = blockByteOffset + 2 + withinBlockIdx; // Skip 2-byte scale
                byte quant1 = wq.get(quantsOffset);
                byte quant2 = wq.get(quantsOffset + 1);
                byte quant3 = wq.get(quantsOffset + 2);
                byte quant4 = wq.get(quantsOffset + 3);

                // Apply RMS normalization inline and compute dot product
                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            // Handle remaining elements
            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wq.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSums[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection with inline normalization ==========
            int kRow = rowId - qDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = kRow * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wk.get(quantsOffset);
                byte quant2 = wk.get(quantsOffset + 1);
                byte quant3 = wk.get(quantsOffset + 2);
                byte quant4 = wk.get(quantsOffset + 3);

                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wk.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSums[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection with inline normalization ==========
            int vRow = rowId - qDim - kvDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = vRow * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wv.get(quantsOffset);
                byte quant2 = wv.get(quantsOffset + 1);
                byte quant3 = wv.get(quantsOffset + 2);
                byte quant4 = wv.get(quantsOffset + 3);

                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wv.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSums[0]);
            }
        }
    }

    /**
     * Warp-shuffle variant of {@link #fusedRmsNormQKVMatmulQ8_0}: one 32-lane warp per output row,
     * reducing via {@code simdShuffleDown} instead of a shared-memory tree. Q8_0 byte layout
     * (34-byte blocks: 2-byte half scale + 32 int8 quants). Three independent branches (q/k/v) each
     * do their own loop + warp reduction + write; keeping the reduction inside each branch avoids
     * carrying the partial across control flow, which produced incorrect PTX for the shuffle.
     */
    public static void fusedRmsNormQKVMatmulQ8_0Warp(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2-byte scale + 32 int8 quants
        int blocksPerRow = (inputDim + blockSize - 1) / blockSize;

        if (rowId < qDim) {
            int rowBlockOffset = rowId * blocksPerRow;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j - blockIdx * blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float s = wq.getHalfFloat(blockByteOffset).getFloat32();
                byte quant = wq.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * s) * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                q.set(rowId, partialSum);
            }
        } else if (rowId < qDim + kvDim) {
            int kRow = rowId - qDim;
            int rowBlockOffset = kRow * blocksPerRow;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j - blockIdx * blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float s = wk.getHalfFloat(blockByteOffset).getFloat32();
                byte quant = wk.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * s) * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                k.set(kRow, partialSum);
            }
        } else if (rowId < qDim + 2 * kvDim) {
            int vRow = rowId - qDim - kvDim;
            int rowBlockOffset = vRow * blocksPerRow;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += 32) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j - blockIdx * blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float s = wv.getHalfFloat(blockByteOffset).getFloat32();
                byte quant = wv.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * s) * (rmsWeights.get(j) * scale * x.get(j));
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                v.set(vRow, partialSum);
            }
        }
    }

    /**
     * Fused Q and K RMSNorm for Qwen3. Combines rmsnormReduction + rmsnormMapIndexInPlace for both
     * Q and K into one kernel.
     *
     * <p>Workgroup assignment: - Workgroups [0, nHeads): Process Q heads - Workgroups [nHeads,
     * nHeads + nHeadKv): Process K heads
     */
    public static void fusedQKRmsNorm(
            KernelContext context,
            FloatArray q, // Q vector (in/out)
            FloatArray k, // K vector (in/out)
            FloatArray qWeights, // Q RMS norm weights
            FloatArray kWeights, // K RMS norm weights
            int nHeads, // number of Q heads
            int nHeadKv, // number of K heads
            int nEmbdHead, // head dimension
            int localMemSize, // local memory size (must be fixed)
            float rmsNormEps) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        // Allocate local memory with FIXED size parameter
        float[] localSum = context.allocateFloatLocalArray(localMemSize);

        if (groupId < nHeads) {
            // === Process Q head ===
            int headOffset = groupId * nEmbdHead;

            // Step 1: Compute sum of squares (reduction)
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = q.get(headOffset + i);
                partialSum += val * val;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            // Compute normalization factor
            float ss = localSum[0];
            ss = ss / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);

            context.localBarrier();

            // Step 2: Apply normalization with weights (in-place)
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float normalized = ss * q.get(headOffset + i);
                q.set(headOffset + i, qWeights.get(i) * normalized);
            }

        } else if (groupId < nHeads + nHeadKv) {
            // === Process K head ===
            int headIdx = groupId - nHeads;
            int headOffset = headIdx * nEmbdHead;

            // Step 1: Compute sum of squares (reduction)
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = k.get(headOffset + i);
                partialSum += val * val;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            // Compute normalization factor
            float ss = localSum[0];
            ss = ss / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);

            context.localBarrier();

            // Step 2: Apply normalization with weights (in-place)
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float normalized = ss * k.get(headOffset + i);
                k.set(headOffset + i, kWeights.get(i) * normalized);
            }
        }
    }

    // ── Batch prefill kernels ────────────────────────────────────────────────

    /**
     * Batched fused Q/K/V projection for Qwen3 GQA (FP16 weights, FP16 input).
     *
     * <p>Like {@code TransformerBatchPrefillKernels.batchedFusedQKVMatmul} but uses separate {@code
     * qDim} (Q output rows) and {@code kvDim} (K/V output rows). Row layout: [0, qDim) → Q; [qDim,
     * qDim+kvDim) → K; [qDim+kvDim, qDim+2*kvDim) → V. Q output stride per batch = qDim; K/V output
     * stride = kvDim. Worker: B*(qDim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmulFP16(
            KernelContext context,
            HalfFloatArray xbFP16Batch,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int groupId = context.globalIdx / localWorkGroupSize;
        int localId = context.localIdx;
        int totalRows = qDim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx = groupId % totalRows;
        int inputOff = batchIdx * inputDim;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < qDim) {
            int rowOff = rowIdx * inputDim;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partial +=
                        wq.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapQBatch.set(batchIdx * qDim + rowIdx, localSum[0]);

        } else if (rowIdx < qDim + kvDim) {
            int kRow = rowIdx - qDim;
            int rowOff = kRow * inputDim;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partial +=
                        wk.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);

        } else {
            int vRow = rowIdx - qDim - kvDim;
            int rowOff = vRow * inputDim;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partial +=
                        wv.get(rowOff + j).getFloat32()
                                * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
        }
    }

    /**
     * Batched fused Q/K/V projection for Qwen3 GQA (Q8_0 weights, FP32 input).
     *
     * <p>Worker: B*(qDim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmulQ8_0(
            KernelContext context,
            FloatArray wrapXbBatch,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray wrapVBatch,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int groupId = context.globalIdx / localWorkGroupSize;
        int localId = context.localIdx;
        int totalRows = qDim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx = groupId % totalRows;
        int inputOff = batchIdx * inputDim;

        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < qDim) {
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOff = rowIdx * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlock = j % blockSize;
                int blockByteOff = (rowBlockOff + blockIdx) * Q8_0_BLOCK_BYTES;
                HalfFloat sc = wq.getHalfFloat(blockByteOff);
                byte q8 = wq.get(blockByteOff + 2 + withinBlock);
                partial += ((float) q8 * sc.getFloat32()) * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapQBatch.set(batchIdx * qDim + rowIdx, localSum[0]);

        } else if (rowIdx < qDim + kvDim) {
            int kRow = rowIdx - qDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOff = kRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlock = j % blockSize;
                int blockByteOff = (rowBlockOff + blockIdx) * Q8_0_BLOCK_BYTES;
                HalfFloat sc = wk.getHalfFloat(blockByteOff);
                byte q8 = wk.get(blockByteOff + 2 + withinBlock);
                partial += ((float) q8 * sc.getFloat32()) * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);

        } else {
            int vRow = rowIdx - qDim - kvDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOff = vRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlock = j % blockSize;
                int blockByteOff = (rowBlockOff + blockIdx) * Q8_0_BLOCK_BYTES;
                HalfFloat sc = wv.getHalfFloat(blockByteOff);
                byte q8 = wv.get(blockByteOff + 2 + withinBlock);
                partial += ((float) q8 * sc.getFloat32()) * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
        }
    }

    /**
     * Batched fused Q/K RMSNorm for Qwen3 GQA.
     *
     * <p>Workgroup layout: B*(nHeads+nHeadKv) groups × nEmbdHead local threads. Groups [0,
     * B*nHeads) normalize Q; groups [B*nHeads, B*(nHeads+nHeadKv)) normalize K. groupIdx =
     * batchIdx*(nHeads+nHeadKv) + headSlot. Worker: B*(nHeads+nHeadKv) workgroups × nEmbdHead
     * threads.
     */
    public static void batchedFusedQKRmsNorm(
            KernelContext context,
            FloatArray wrapQBatch,
            FloatArray wrapKBatch,
            FloatArray qWeights,
            FloatArray kWeights,
            int nHeads,
            int nHeadKv,
            int nEmbdHead,
            int qDim,
            int kvDim,
            float rmsNormEps) {

        int groupId = context.globalIdx / nEmbdHead;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int totalHeadsPerBatch = nHeads + nHeadKv;

        int batchIdx = groupId / totalHeadsPerBatch;
        int headSlot = groupId % totalHeadsPerBatch;

        float[] localSum = context.allocateFloatLocalArray(nEmbdHead);

        if (headSlot < nHeads) {
            // Q head
            int headOffset = batchIdx * qDim + headSlot * nEmbdHead;
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = wrapQBatch.get(headOffset + i);
                partialSum += val * val;
            }
            localSum[localId] = partialSum;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) localSum[localId] += localSum[localId + stride];
                context.localBarrier();
            }
            float ss = localSum[0] / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);
            context.localBarrier();
            for (int i = localId; i < nEmbdHead; i += localSize) {
                wrapQBatch.set(
                        headOffset + i, qWeights.get(i) * ss * wrapQBatch.get(headOffset + i));
            }
        } else {
            // K head
            int kHeadIdx = headSlot - nHeads;
            int headOffset = batchIdx * kvDim + kHeadIdx * nEmbdHead;
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = wrapKBatch.get(headOffset + i);
                partialSum += val * val;
            }
            localSum[localId] = partialSum;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) localSum[localId] += localSum[localId + stride];
                context.localBarrier();
            }
            float ss = localSum[0] / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);
            context.localBarrier();
            for (int i = localId; i < nEmbdHead; i += localSize) {
                wrapKBatch.set(
                        headOffset + i, kWeights.get(i) * ss * wrapKBatch.get(headOffset + i));
            }
        }
    }

    // ── Packed-QKV variants for the tensor-core batch prefill path ───────────
    // These mirror batchedFusedQKRmsNorm / batchedRopeWithKVCacheQwen3 but read
    // and write the packed [q | k | v] buffer produced by gemmMMAQKV
    // (row stride qDim + 2*kvDim), so no separate Q/K/V buffers are needed.

    public static void batchedFusedQKRmsNormPacked(
            KernelContext context,
            FloatArray qkvBatch,
            FloatArray qWeights,
            FloatArray kWeights,
            int nHeads,
            int nHeadKv,
            int nEmbdHead,
            int qDim,
            int kvDim,
            float rmsNormEps) {

        int groupId = context.globalIdx / nEmbdHead;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int totalHeadsPerBatch = nHeads + nHeadKv;
        int qkvStride = qDim + 2 * kvDim;

        int batchIdx = groupId / totalHeadsPerBatch;
        int headSlot = groupId % totalHeadsPerBatch;

        float[] localSum = context.allocateFloatLocalArray(nEmbdHead);

        if (headSlot < nHeads) {
            // Q head (packed offset 0)
            int headOffset = batchIdx * qkvStride + headSlot * nEmbdHead;
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = qkvBatch.get(headOffset + i);
                partialSum += val * val;
            }
            localSum[localId] = partialSum;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }
            float ss = localSum[0] / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);
            context.localBarrier();
            for (int i = localId; i < nEmbdHead; i += localSize) {
                qkvBatch.set(headOffset + i, qWeights.get(i) * ss * qkvBatch.get(headOffset + i));
            }
        } else {
            // K head (packed offset qDim)
            int kHeadIdx = headSlot - nHeads;
            int headOffset = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = qkvBatch.get(headOffset + i);
                partialSum += val * val;
            }
            localSum[localId] = partialSum;
            context.localBarrier();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }
            float ss = localSum[0] / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);
            context.localBarrier();
            for (int i = localId; i < nEmbdHead; i += localSize) {
                qkvBatch.set(headOffset + i, kWeights.get(i) * ss * qkvBatch.get(headOffset + i));
            }
        }
    }

    /**
     * Per-slot Qwen3 split-half RoPE + KV-cache write (batched DECODE).
     *
     * <p>Fork of {@link #batchedRopeWithKVCacheQwen3Packed}: each slot rotates at its own position
     * {@code seqPositions[batchIdx]} and writes K/V into its own KV region ({@code batchIdx} stride
     * = {@code numLayers*contextLength*kvDim}).
     */
    public static void batchedDecodeRopeWithKVCacheQwen3Packed(
            KernelContext context,
            IntArray seqPositions,
            FloatArray qkvBatch,
            FloatArray wrapKeyCache,
            FloatArray wrapValueCache,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            int numLayers,
            int contextLength,
            int qDim) {

        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;
        int qkvStride = qDim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);

        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int qHeadBase = batchIdx * qkvStride + headIdx * nEmbdHead;
        float v0q = qkvBatch.get(qHeadBase + ic);
        float v1q = qkvBatch.get(qHeadBase + ic + halfEmbdHead);
        qkvBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        qkvBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            int vHeadBase = batchIdx * qkvStride + qDim + kvDim + kHeadIdx * nEmbdHead;
            float v0k = qkvBatch.get(kHeadBase + ic);
            float v1k = qkvBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;

            int slotBase = batchIdx * (numLayers * contextLength * kvDim);
            int cacheOff =
                    slotBase
                            + layerIndex * contextLength * kvDim
                            + pos * kvDim
                            + kHeadIdx * nEmbdHead;
            wrapKeyCache.set(cacheOff + ic, rotK0);
            wrapKeyCache.set(cacheOff + ic + halfEmbdHead, rotK1);
            wrapValueCache.set(cacheOff + ic, qkvBatch.get(vHeadBase + ic));
            wrapValueCache.set(
                    cacheOff + ic + halfEmbdHead, qkvBatch.get(vHeadBase + ic + halfEmbdHead));
        }
    }

    /**
     * Paged Qwen3 split-half RoPE + KV write (block-table indirection). {@code blockCfg} packs
     * {@code blockSize | (maxBlocksPerSlot << 16)}. Paired flash attention is the shared {@code
     * batchedDecodePagedAttentionFP16Out} (qDim == nHeads*nEmbdHead).
     */
    public static void batchedDecodePagedRopeWithKVCacheQwen3Packed(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            FloatArray keyPool,
            FloatArray valuePool,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int qDim) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;
        int qkvStride = qDim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);

        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int qHeadBase = batchIdx * qkvStride + headIdx * nEmbdHead;
        float v0q = qkvBatch.get(qHeadBase + ic);
        float v1q = qkvBatch.get(qHeadBase + ic + halfEmbdHead);
        qkvBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        qkvBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            int vHeadBase = batchIdx * qkvStride + qDim + kvDim + kHeadIdx * nEmbdHead;
            float v0k = qkvBatch.get(kHeadBase + ic);
            float v1k = qkvBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;

            int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + pos / blockSize);
            int cacheOff =
                    physBlock * (numLayers * blockSize * kvDim)
                            + layerIndex * (blockSize * kvDim)
                            + (pos % blockSize) * kvDim
                            + kHeadIdx * nEmbdHead;
            keyPool.set(cacheOff + ic, rotK0);
            keyPool.set(cacheOff + ic + halfEmbdHead, rotK1);
            valuePool.set(cacheOff + ic, qkvBatch.get(vHeadBase + ic));
            valuePool.set(
                    cacheOff + ic + halfEmbdHead, qkvBatch.get(vHeadBase + ic + halfEmbdHead));
        }
    }

    // FP16-cache twin of batchedDecodePagedRopeWithKVCacheQwen3Packed: identical arithmetic, the
    // cache stored in binary16.
    /**
     * Paged Qwen3 split-half RoPE + KV write (block-table indirection). {@code blockCfg} packs
     * {@code blockSize | (maxBlocksPerSlot << 16)}. Paired flash attention is the shared {@code
     * batchedDecodePagedAttentionFP16Out} (qDim == nHeads*nEmbdHead).
     */
    public static void batchedDecodePagedRopeWithKVCacheQwen3PackedKVFP16(
            KernelContext context,
            IntArray seqPositions,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            float ropeTheta,
            int kvDim,
            int nEmbdHead,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int qDim) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int globalIdx = context.globalIdx;
        int halfQDim = qDim / 2;
        int batchIdx = globalIdx / halfQDim;
        int pairIdx = globalIdx % halfQDim;
        int qkvStride = qDim + 2 * kvDim;

        int pos = seqPositions.get(batchIdx);

        int halfEmbdHead = nEmbdHead / 2;
        int ic = pairIdx % halfEmbdHead;
        int headIdx = pairIdx / halfEmbdHead;

        float freq = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * ic / (float) nEmbdHead);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int qHeadBase = batchIdx * qkvStride + headIdx * nEmbdHead;
        float v0q = qkvBatch.get(qHeadBase + ic);
        float v1q = qkvBatch.get(qHeadBase + ic + halfEmbdHead);
        qkvBatch.set(qHeadBase + ic, v0q * fcr - v1q * fci);
        qkvBatch.set(qHeadBase + ic + halfEmbdHead, v0q * fci + v1q * fcr);

        if (pairIdx < kvDim / 2) {
            int kHeadIdx = pairIdx / halfEmbdHead;
            int kHeadBase = batchIdx * qkvStride + qDim + kHeadIdx * nEmbdHead;
            int vHeadBase = batchIdx * qkvStride + qDim + kvDim + kHeadIdx * nEmbdHead;
            float v0k = qkvBatch.get(kHeadBase + ic);
            float v1k = qkvBatch.get(kHeadBase + ic + halfEmbdHead);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;

            int physBlock = blockTable.get(batchIdx * maxBlocksPerSlot + pos / blockSize);
            int cacheOff =
                    physBlock * (numLayers * blockSize * kvDim)
                            + layerIndex * (blockSize * kvDim)
                            + (pos % blockSize) * kvDim
                            + kHeadIdx * nEmbdHead;
            keyPool.set(cacheOff + ic, new HalfFloat(rotK0));
            keyPool.set(cacheOff + ic + halfEmbdHead, new HalfFloat(rotK1));
            valuePool.set(cacheOff + ic, new HalfFloat(qkvBatch.get(vHeadBase + ic)));
            valuePool.set(
                    cacheOff + ic + halfEmbdHead,
                    new HalfFloat(qkvBatch.get(vHeadBase + ic + halfEmbdHead)));
        }
    }
}
// @formatter:on
