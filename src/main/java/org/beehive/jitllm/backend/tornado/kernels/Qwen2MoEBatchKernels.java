package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** GPU kernels used by Qwen2-MoE batch prefill. */
public final class Qwen2MoEBatchKernels {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_REPACK_GROUP_BLOCKS = 16;
    private static final int Q8_0_REPACK_SCALE_BYTES = Q8_0_REPACK_GROUP_BLOCKS * 2;
    private static final int Q8_0_REPACK_GROUP_BYTES =
            Q8_0_REPACK_SCALE_BYTES + Q8_0_REPACK_GROUP_BLOCKS * Q8_0_BLOCK_SIZE;

    private Qwen2MoEBatchKernels() {}

    /** Computes one router score for every token-expert pair. */
    public static void batchedRouterProjection(
            KernelContext context,
            FloatArray input,
            FloatArray routerLogits,
            FloatArray routerWeights,
            IntArray activeBatchSizeHolder,
            int dim,
            int numberOfExperts,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;

        // One work-group handles one (token, expert) pair.
        // For 60 experts, groups 0.59 process token 0, groups 60.119
        // process token 1, and so on.

        int token = groupId / numberOfExperts;
        int expert = groupId % numberOfExperts;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int inputOffset = token * dim;
        int weightOffset = expert * dim;

        // Each local thread computes a strided part of the dot product.
        float partialSum = 0.0f;

        for (int column = localId; column < dim; column += localWorkGroupSize) {

            partialSum +=
                    input.get(inputOffset + column) * routerWeights.get(weightOffset + column);
        }

        // Store every thread's partial sum in local memory.
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        localSums[localId] = partialSum;
        context.localBarrier();

        // Reduce the partial sums inside this work-group.
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {

            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }

            context.localBarrier();
        }

        // Local thread 0 writes routerLogits[token, expert].
        if (localId == 0) {
            int outputOffset = token * numberOfExperts + expert;
            routerLogits.set(outputOffset, localSums[0]);
        }
    }

    /** Adds Qwen2's Q, K, and V biases independently for every active token. */
    public static void batchedQKVBias(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray qBias,
            FloatArray kBias,
            FloatArray vBias,
            IntArray activeBatchSizeHolder,
            int dim,
            int kvDim) {

        int index = context.globalIdx;
        int rowsPerToken = dim + 2 * kvDim;
        int token = index / rowsPerToken;
        int row = index % rowsPerToken;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        if (row < dim) {
            int qIndex = token * dim + row;
            qBatch.set(qIndex, qBatch.get(qIndex) + qBias.get(row));
        } else if (row < dim + kvDim) {
            int kRow = row - dim;
            int kIndex = token * kvDim + kRow;
            kBatch.set(kIndex, kBatch.get(kIndex) + kBias.get(kRow));
        } else {
            int vRow = row - dim - kvDim;
            int vIndex = token * kvDim + vRow;
            vBatch.set(vIndex, vBatch.get(vIndex) + vBias.get(vRow));
        }
    }

    /** Applies softmax and selects Top-K experts independently for each token. */
    public static void batchedSoftmaxAndTopK(
            KernelContext context,
            FloatArray routerLogits,
            IntArray selectedExperts,
            FloatArray routingWeights,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK) {

        // One GPU thread handles the complete routing result for one token.
        int token = context.globalIdx;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int logitsOffset = token * numberOfExperts;
        int assignmentOffset = token * topK;

        // Find this token's maximum router logit.
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = logitsOffset; i < logitsOffset + numberOfExperts; i++) {
            maxLogit = Math.max(maxLogit, routerLogits.get(i));
        }

        // Compute the stable softmax denominator.
        float sumExp = 0.0f;

        for (int expert = 0; expert < numberOfExperts; expert++) {
            float logit = routerLogits.get(logitsOffset + expert);
            sumExp += TornadoMath.exp(logit - maxLogit);
        }

        // Convert this token's logits to probabilities.
        for (int expert = 0; expert < numberOfExperts; expert++) {
            int index = logitsOffset + expert;
            float logit = routerLogits.get(index);

            float probability = TornadoMath.exp(logit - maxLogit) / sumExp;

            routerLogits.set(index, probability);
        }

        // Select Top-K expert IDs and their routing weights.
        for (int slot = 0; slot < topK; slot++) {
            float currentMax = Float.NEGATIVE_INFINITY;
            int selectedExpert = -1;

            for (int expert = 0; expert < numberOfExperts; expert++) {
                float probability = routerLogits.get(logitsOffset + expert);

                if (probability > currentMax) {
                    currentMax = probability;
                    selectedExpert = expert;
                }
            }

            selectedExperts.set(assignmentOffset + slot, selectedExpert);
            routingWeights.set(assignmentOffset + slot, currentMax);

            routerLogits.set(logitsOffset + selectedExpert, Float.NEGATIVE_INFINITY);
        }
    }

    /** Groups assignments and builds fixed-size tiles containing one expert's tokens. */
    public static void groupAssignmentsAndBuildExpertTiles(
            KernelContext context,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            IntArray groupedPositionByAssignment,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK,
            int expertTileSize) {

        if (context.globalIdx != 0) {
            return;
        }

        int numberOfAssignments = activeBatchSizeHolder.get(0) * topK;
        int groupedPosition = 0;
        int tilePosition = 0;

        for (int expert = 0; expert < numberOfExperts; expert++) {
            int expertStart = groupedPosition;
            for (int assignment = 0; assignment < numberOfAssignments; assignment++) {
                if (selectedExperts.get(assignment) == expert) {
                    groupedAssignmentIds.set(groupedPosition, assignment);
                    groupedPositionByAssignment.set(assignment, groupedPosition);
                    groupedPosition++;
                }
            }

            for (int tileStart = expertStart;
                    tileStart < groupedPosition;
                    tileStart += expertTileSize) {
                int tileCount = groupedPosition - tileStart;
                if (tileCount > expertTileSize) {
                    tileCount = expertTileSize;
                }
                expertTileIds.set(tilePosition, expert);
                expertTileStarts.set(tilePosition, tileStart);
                expertTileCounts.set(tilePosition, tileCount);
                tilePosition++;
            }
        }

        expertTileCountHolder.set(0, tilePosition);
    }

    /** Computes a 2-token by 4-output-row Gate/Up tile in one work-group. */
    public static void tiled2DRoutedExpertsGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray groupedAssignmentIds,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray groupedExpertHidden,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (moeHiddenDim + rowsPerGroup - 1) / rowsPerGroup;
        int tilePosition = context.groupIdx / outputRowTiles;

        if (tilePosition >= expertTileCountHolder.get(0)) {
            return;
        }

        int outputRowTile = context.groupIdx % outputRowTiles;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < moeHiddenDim;
        int expert = expertTileIds.get(tilePosition);
        int tileStart = expertTileStarts.get(tilePosition);
        int tileCount = expertTileCounts.get(tilePosition);
        int token0 = groupedAssignmentIds.get(tileStart) / topK;
        int token1 = token0;
        if (tileCount > 1) {
            token1 = groupedAssignmentIds.get(tileStart + 1) / topK;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        float gatePartial0 = 0.0f;
        float upPartial0 = 0.0f;
        float gatePartial1 = 0.0f;
        float upPartial1 = 0.0f;
        if (activeRow) {
            int rowBlockOffset = (expert * moeHiddenDim + rowId) * blocksPerRow;
            int inputOffset0 = token0 * dim;
            int inputOffset1 = token1 * dim;

            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= dim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                int scaleOffset = repackedScaleOffset(blockIndex);

                float gateWeight =
                        gateExperts.get(quantIndex)
                                * gateExperts.getHalfFloat(scaleOffset).getFloat32();
                float upWeight =
                        upExperts.get(quantIndex)
                                * upExperts.getHalfFloat(scaleOffset).getFloat32();

                float input0 = inputBatch.get(inputOffset0 + column);
                gatePartial0 += gateWeight * input0;
                upPartial0 += upWeight * input0;
                if (tileCount > 1) {
                    float input1 = inputBatch.get(inputOffset1 + column);
                    gatePartial1 += gateWeight * input1;
                    upPartial1 += upWeight * input1;
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(4 * localWorkGroupSize);
        localPartials[localId] = gatePartial0;
        localPartials[localWorkGroupSize + localId] = upPartial0;
        localPartials[2 * localWorkGroupSize + localId] = gatePartial1;
        localPartials[3 * localWorkGroupSize + localId] = upPartial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
                localPartials[2 * localWorkGroupSize + target] +=
                        localPartials[2 * localWorkGroupSize + source];
                localPartials[3 * localWorkGroupSize + target] +=
                        localPartials[3 * localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            float gate0 = localPartials[subgroupBase];
            float up0 = localPartials[localWorkGroupSize + subgroupBase];
            float siluGate0 = gate0 / (1.0f + TornadoMath.exp(-gate0));
            groupedExpertHidden.set(tileStart * moeHiddenDim + rowId, siluGate0 * up0);
            if (tileCount > 1) {
                float gate1 = localPartials[2 * localWorkGroupSize + subgroupBase];
                float up1 = localPartials[3 * localWorkGroupSize + subgroupBase];
                float siluGate1 = gate1 / (1.0f + TornadoMath.exp(-gate1));
                groupedExpertHidden.set((tileStart + 1) * moeHiddenDim + rowId, siluGate1 * up1);
            }
        }
    }

    /** Computes a 2-token by 4-output-row Down tile in one work-group. */
    public static void tiled2DRoutedExpertsDownQ8_0(
            KernelContext context,
            FloatArray groupedExpertHidden,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            ByteArray downExperts,
            FloatArray groupedExpertDown,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (dim + rowsPerGroup - 1) / rowsPerGroup;
        int tilePosition = context.groupIdx / outputRowTiles;

        if (tilePosition >= expertTileCountHolder.get(0)) {
            return;
        }

        int outputRowTile = context.groupIdx % outputRowTiles;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < dim;
        int expert = expertTileIds.get(tilePosition);
        int tileStart = expertTileStarts.get(tilePosition);
        int tileCount = expertTileCounts.get(tilePosition);
        int hiddenOffset0 = tileStart * moeHiddenDim;
        int hiddenOffset1 = (tileStart + 1) * moeHiddenDim;
        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        if (activeRow) {
            int rowBlockOffset = (expert * dim + rowId) * blocksPerRow;

            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= moeHiddenDim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                float weight =
                        downExperts.get(quantIndex)
                                * downExperts
                                        .getHalfFloat(repackedScaleOffset(blockIndex))
                                        .getFloat32();

                partial0 += weight * groupedExpertHidden.get(hiddenOffset0 + column);
                if (tileCount > 1) {
                    partial1 += weight * groupedExpertHidden.get(hiddenOffset1 + column);
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(2 * localWorkGroupSize);
        localPartials[localId] = partial0;
        localPartials[localWorkGroupSize + localId] = partial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            groupedExpertDown.set(tileStart * dim + rowId, localPartials[subgroupBase]);
            if (tileCount > 1) {
                groupedExpertDown.set(
                        (tileStart + 1) * dim + rowId,
                        localPartials[localWorkGroupSize + subgroupBase]);
            }
        }
    }

    /** Adds the weighted routed-expert results back to each token's residual. */
    public static void accumulateGroupedRoutedExperts(
            KernelContext context,
            FloatArray groupedExpertDown,
            IntArray groupedPositionByAssignment,
            FloatArray routingWeights,
            FloatArray residualBatch,
            IntArray activeBatchSizeHolder,
            int dim,
            int topK) {

        int index = context.globalIdx;
        int token = index / dim;
        int rowId = index % dim;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        float result = residualBatch.get(index);
        int assignmentOffset = token * topK;
        for (int slot = 0; slot < topK; slot++) {
            int assignment = assignmentOffset + slot;
            int groupedPosition = groupedPositionByAssignment.get(assignment);
            int downOffset = groupedPosition * dim + rowId;
            result += routingWeights.get(assignment) * groupedExpertDown.get(downOffset);
        }

        residualBatch.set(index, result);
    }

    /** Computes two shared-expert token rows and four output rows per work-group. */
    public static void tiled2SharedExpertGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHiddenBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (sharedExpertHiddenDim + rowsPerGroup - 1) / rowsPerGroup;
        int tokenTile = context.groupIdx / outputRowTiles;
        int outputRowTile = context.groupIdx % outputRowTiles;
        int token0 = tokenTile * 2;
        int token1 = token0 + 1;
        int activeBatchSize = activeBatchSizeHolder.get(0);
        boolean active0 = token0 < activeBatchSize;
        boolean active1 = token1 < activeBatchSize;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < sharedExpertHiddenDim;

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        float gatePartial0 = 0.0f;
        float upPartial0 = 0.0f;
        float gatePartial1 = 0.0f;
        float upPartial1 = 0.0f;

        if (activeRow && (active0 || active1)) {
            int inputOffset0 = token0 * dim;
            int inputOffset1 = token1 * dim;
            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= dim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                int scaleOffset = repackedScaleOffset(blockIndex);
                float gateWeight =
                        sharedGate.get(quantIndex)
                                * sharedGate.getHalfFloat(scaleOffset).getFloat32();
                float upWeight =
                        sharedUp.get(quantIndex) * sharedUp.getHalfFloat(scaleOffset).getFloat32();

                if (active0) {
                    float input0 = inputBatch.get(inputOffset0 + column);
                    gatePartial0 += gateWeight * input0;
                    upPartial0 += upWeight * input0;
                }
                if (active1) {
                    float input1 = inputBatch.get(inputOffset1 + column);
                    gatePartial1 += gateWeight * input1;
                    upPartial1 += upWeight * input1;
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(4 * localWorkGroupSize);
        localPartials[localId] = gatePartial0;
        localPartials[localWorkGroupSize + localId] = upPartial0;
        localPartials[2 * localWorkGroupSize + localId] = gatePartial1;
        localPartials[3 * localWorkGroupSize + localId] = upPartial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
                localPartials[2 * localWorkGroupSize + target] +=
                        localPartials[2 * localWorkGroupSize + source];
                localPartials[3 * localWorkGroupSize + target] +=
                        localPartials[3 * localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            if (active0) {
                float gate0 = localPartials[subgroupBase];
                float up0 = localPartials[localWorkGroupSize + subgroupBase];
                float siluGate0 = gate0 / (1.0f + TornadoMath.exp(-gate0));
                sharedHiddenBatch.set(token0 * sharedExpertHiddenDim + rowId, siluGate0 * up0);
            }
            if (active1) {
                float gate1 = localPartials[2 * localWorkGroupSize + subgroupBase];
                float up1 = localPartials[3 * localWorkGroupSize + subgroupBase];
                float siluGate1 = gate1 / (1.0f + TornadoMath.exp(-gate1));
                sharedHiddenBatch.set(token1 * sharedExpertHiddenDim + rowId, siluGate1 * up1);
            }
        }
    }

    /** Computes four shared-expert token rows and four output rows per work-group. */
    public static void tiled4SharedExpertGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHiddenBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (sharedExpertHiddenDim + rowsPerGroup - 1) / rowsPerGroup;
        int tokenTile = context.groupIdx / outputRowTiles;
        int outputRowTile = context.groupIdx % outputRowTiles;
        int token0 = tokenTile * 4;
        int token1 = token0 + 1;
        int token2 = token0 + 2;
        int token3 = token0 + 3;
        int activeBatchSize = activeBatchSizeHolder.get(0);
        boolean active0 = token0 < activeBatchSize;
        boolean active1 = token1 < activeBatchSize;
        boolean active2 = token2 < activeBatchSize;
        boolean active3 = token3 < activeBatchSize;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < sharedExpertHiddenDim;

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        float gatePartial0 = 0.0f;
        float upPartial0 = 0.0f;
        float gatePartial1 = 0.0f;
        float upPartial1 = 0.0f;
        float gatePartial2 = 0.0f;
        float upPartial2 = 0.0f;
        float gatePartial3 = 0.0f;
        float upPartial3 = 0.0f;

        if (activeRow && (active0 || active1 || active2 || active3)) {
            int inputOffset0 = token0 * dim;
            int inputOffset1 = token1 * dim;
            int inputOffset2 = token2 * dim;
            int inputOffset3 = token3 * dim;
            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= dim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                int scaleOffset = repackedScaleOffset(blockIndex);
                float gateWeight =
                        sharedGate.get(quantIndex)
                                * sharedGate.getHalfFloat(scaleOffset).getFloat32();
                float upWeight =
                        sharedUp.get(quantIndex) * sharedUp.getHalfFloat(scaleOffset).getFloat32();

                if (active0) {
                    float input0 = inputBatch.get(inputOffset0 + column);
                    gatePartial0 += gateWeight * input0;
                    upPartial0 += upWeight * input0;
                }
                if (active1) {
                    float input1 = inputBatch.get(inputOffset1 + column);
                    gatePartial1 += gateWeight * input1;
                    upPartial1 += upWeight * input1;
                }
                if (active2) {
                    float input2 = inputBatch.get(inputOffset2 + column);
                    gatePartial2 += gateWeight * input2;
                    upPartial2 += upWeight * input2;
                }
                if (active3) {
                    float input3 = inputBatch.get(inputOffset3 + column);
                    gatePartial3 += gateWeight * input3;
                    upPartial3 += upWeight * input3;
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(8 * localWorkGroupSize);
        localPartials[localId] = gatePartial0;
        localPartials[localWorkGroupSize + localId] = upPartial0;
        localPartials[2 * localWorkGroupSize + localId] = gatePartial1;
        localPartials[3 * localWorkGroupSize + localId] = upPartial1;
        localPartials[4 * localWorkGroupSize + localId] = gatePartial2;
        localPartials[5 * localWorkGroupSize + localId] = upPartial2;
        localPartials[6 * localWorkGroupSize + localId] = gatePartial3;
        localPartials[7 * localWorkGroupSize + localId] = upPartial3;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                for (int segment = 0; segment < 8; segment++) {
                    int segmentOffset = segment * localWorkGroupSize;
                    localPartials[segmentOffset + target] += localPartials[segmentOffset + source];
                }
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            if (active0) {
                float gate0 = localPartials[subgroupBase];
                float up0 = localPartials[localWorkGroupSize + subgroupBase];
                float siluGate0 = gate0 / (1.0f + TornadoMath.exp(-gate0));
                sharedHiddenBatch.set(token0 * sharedExpertHiddenDim + rowId, siluGate0 * up0);
            }
            if (active1) {
                float gate1 = localPartials[2 * localWorkGroupSize + subgroupBase];
                float up1 = localPartials[3 * localWorkGroupSize + subgroupBase];
                float siluGate1 = gate1 / (1.0f + TornadoMath.exp(-gate1));
                sharedHiddenBatch.set(token1 * sharedExpertHiddenDim + rowId, siluGate1 * up1);
            }
            if (active2) {
                float gate2 = localPartials[4 * localWorkGroupSize + subgroupBase];
                float up2 = localPartials[5 * localWorkGroupSize + subgroupBase];
                float siluGate2 = gate2 / (1.0f + TornadoMath.exp(-gate2));
                sharedHiddenBatch.set(token2 * sharedExpertHiddenDim + rowId, siluGate2 * up2);
            }
            if (active3) {
                float gate3 = localPartials[6 * localWorkGroupSize + subgroupBase];
                float up3 = localPartials[7 * localWorkGroupSize + subgroupBase];
                float siluGate3 = gate3 / (1.0f + TornadoMath.exp(-gate3));
                sharedHiddenBatch.set(token3 * sharedExpertHiddenDim + rowId, siluGate3 * up3);
            }
        }
    }

    /** Computes the sigmoid gate that scales each token's shared-expert output. */
    public static void batchedSharedExpertGateWeight(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray sharedGateInput,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            int dim,
            int localWorkGroupSize) {

        int token = context.groupIdx;
        int localId = context.localIdx;
        boolean active = token < activeBatchSizeHolder.get(0);
        int inputOffset = token * dim;

        float partialScore = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {
                partialScore += sharedGateInput.get(column) * inputBatch.get(inputOffset + column);
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialScore;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            float sharedWeight = 1.0f / (1.0f + TornadoMath.exp(-localSums[0]));
            sharedWeightBatch.set(token, sharedWeight);
        }
    }

    /** Down-projects two shared-expert token rows and four output rows per work-group. */
    public static void tiled2SharedExpertDownAndAccumulateQ8_0(
            KernelContext context,
            FloatArray sharedHiddenBatch,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedDown,
            FloatArray residualBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (dim + rowsPerGroup - 1) / rowsPerGroup;
        int tokenTile = context.groupIdx / outputRowTiles;
        int outputRowTile = context.groupIdx % outputRowTiles;
        int token0 = tokenTile * 2;
        int token1 = token0 + 1;
        int activeBatchSize = activeBatchSizeHolder.get(0);
        boolean active0 = token0 < activeBatchSize;
        boolean active1 = token1 < activeBatchSize;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < dim;

        int blocksPerRow = (sharedExpertHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        float partial0 = 0.0f;
        float partial1 = 0.0f;

        if (activeRow && (active0 || active1)) {
            int hiddenOffset0 = token0 * sharedExpertHiddenDim;
            int hiddenOffset1 = token1 * sharedExpertHiddenDim;
            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= sharedExpertHiddenDim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                float weight =
                        sharedDown.get(quantIndex)
                                * sharedDown
                                        .getHalfFloat(repackedScaleOffset(blockIndex))
                                        .getFloat32();
                if (active0) {
                    partial0 += weight * sharedHiddenBatch.get(hiddenOffset0 + column);
                }
                if (active1) {
                    partial1 += weight * sharedHiddenBatch.get(hiddenOffset1 + column);
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(2 * localWorkGroupSize);
        localPartials[localId] = partial0;
        localPartials[localWorkGroupSize + localId] = partial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            if (active0) {
                int output0 = token0 * dim + rowId;
                residualBatch.set(
                        output0,
                        residualBatch.get(output0)
                                + sharedWeightBatch.get(token0) * localPartials[subgroupBase]);
            }
            if (active1) {
                int output1 = token1 * dim + rowId;
                residualBatch.set(
                        output1,
                        residualBatch.get(output1)
                                + sharedWeightBatch.get(token1)
                                        * localPartials[localWorkGroupSize + subgroupBase]);
            }
        }
    }

    /** Down-projects four shared-expert token rows and four output rows per work-group. */
    public static void tiled4SharedExpertDownAndAccumulateQ8_0(
            KernelContext context,
            FloatArray sharedHiddenBatch,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedDown,
            FloatArray residualBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (dim + rowsPerGroup - 1) / rowsPerGroup;
        int tokenTile = context.groupIdx / outputRowTiles;
        int outputRowTile = context.groupIdx % outputRowTiles;
        int token0 = tokenTile * 4;
        int token1 = token0 + 1;
        int token2 = token0 + 2;
        int token3 = token0 + 3;
        int activeBatchSize = activeBatchSizeHolder.get(0);
        boolean active0 = token0 < activeBatchSize;
        boolean active1 = token1 < activeBatchSize;
        boolean active2 = token2 < activeBatchSize;
        boolean active3 = token3 < activeBatchSize;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < dim;

        int blocksPerRow = (sharedExpertHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;

        if (activeRow && (active0 || active1 || active2 || active3)) {
            int hiddenOffset0 = token0 * sharedExpertHiddenDim;
            int hiddenOffset1 = token1 * sharedExpertHiddenDim;
            int hiddenOffset2 = token2 * sharedExpertHiddenDim;
            int hiddenOffset3 = token3 * sharedExpertHiddenDim;
            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= sharedExpertHiddenDim) {
                    continue;
                }
                int blockIndex = rowBlockOffset + block;
                int quantIndex = repackedQuantOffset(blockIndex) + lane;
                float weight =
                        sharedDown.get(quantIndex)
                                * sharedDown
                                        .getHalfFloat(repackedScaleOffset(blockIndex))
                                        .getFloat32();
                if (active0) {
                    partial0 += weight * sharedHiddenBatch.get(hiddenOffset0 + column);
                }
                if (active1) {
                    partial1 += weight * sharedHiddenBatch.get(hiddenOffset1 + column);
                }
                if (active2) {
                    partial2 += weight * sharedHiddenBatch.get(hiddenOffset2 + column);
                }
                if (active3) {
                    partial3 += weight * sharedHiddenBatch.get(hiddenOffset3 + column);
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(4 * localWorkGroupSize);
        localPartials[localId] = partial0;
        localPartials[localWorkGroupSize + localId] = partial1;
        localPartials[2 * localWorkGroupSize + localId] = partial2;
        localPartials[3 * localWorkGroupSize + localId] = partial3;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                for (int segment = 0; segment < 4; segment++) {
                    int segmentOffset = segment * localWorkGroupSize;
                    localPartials[segmentOffset + target] += localPartials[segmentOffset + source];
                }
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            if (active0) {
                int output0 = token0 * dim + rowId;
                residualBatch.set(
                        output0,
                        residualBatch.get(output0)
                                + sharedWeightBatch.get(token0) * localPartials[subgroupBase]);
            }
            if (active1) {
                int output1 = token1 * dim + rowId;
                residualBatch.set(
                        output1,
                        residualBatch.get(output1)
                                + sharedWeightBatch.get(token1)
                                        * localPartials[localWorkGroupSize + subgroupBase]);
            }
            if (active2) {
                int output2 = token2 * dim + rowId;
                residualBatch.set(
                        output2,
                        residualBatch.get(output2)
                                + sharedWeightBatch.get(token2)
                                        * localPartials[2 * localWorkGroupSize + subgroupBase]);
            }
            if (active3) {
                int output3 = token3 * dim + rowId;
                residualBatch.set(
                        output3,
                        residualBatch.get(output3)
                                + sharedWeightBatch.get(token3)
                                        * localPartials[3 * localWorkGroupSize + subgroupBase]);
            }
        }
    }

    private static int repackedScaleOffset(int blockIndex) {
        int group = blockIndex / Q8_0_REPACK_GROUP_BLOCKS;
        int blockInGroup = blockIndex % Q8_0_REPACK_GROUP_BLOCKS;
        return group * Q8_0_REPACK_GROUP_BYTES + blockInGroup * 2;
    }

    private static int repackedQuantOffset(int blockIndex) {
        int group = blockIndex / Q8_0_REPACK_GROUP_BLOCKS;
        int blockInGroup = blockIndex % Q8_0_REPACK_GROUP_BLOCKS;
        return group * Q8_0_REPACK_GROUP_BYTES
                + Q8_0_REPACK_SCALE_BYTES
                + blockInGroup * Q8_0_BLOCK_SIZE;
    }
}
