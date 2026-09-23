package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** GPU kernels used by Qwen2-MoE batch prefill. */
public final class Qwen2MoEBatchKernels {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

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

    /** Groups token-expert assignments by expert. */
    public static void groupAssignmentsByExpert(
            KernelContext context,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            IntArray groupedPositionByAssignment,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK) {

        // Start with one thread for a simple, deterministic implementation.
        if (context.globalIdx != 0) {
            return;
        }

        int numberOfAssignments = activeBatchSizeHolder.get(0) * topK;
        int groupedPosition = 0;

        for (int expert = 0; expert < numberOfExperts; expert++) {
            for (int assignment = 0; assignment < numberOfAssignments; assignment++) {
                int selectedExpert = selectedExperts.get(assignment);
                if (selectedExpert == expert) {
                    groupedAssignmentIds.set(groupedPosition, assignment);
                    groupedPositionByAssignment.set(assignment, groupedPosition);
                    groupedPosition++;
                }
            }
        }
    }

    /** Computes routed Gate/Up projections in expert-grouped assignment order. */
    public static void groupedRoutedExpertsGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            IntArray activeBatchSizeHolder,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray groupedExpertHidden,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int groupedPosition = flatGroupId / moeHiddenDim;
        int rowId = flatGroupId % moeHiddenDim;
        int numberOfAssignments = activeBatchSizeHolder.get(0) * topK;

        boolean active = groupedPosition < numberOfAssignments;
        int assignment = 0;
        int token = 0;
        int expert = 0;
        if (active) {
            assignment = groupedAssignmentIds.get(groupedPosition);
            token = assignment / topK;
            expert = selectedExperts.get(assignment);
            active = expert >= 0 && expert < numberOfExperts;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * moeHiddenDim + rowId) * blocksPerRow;
        int inputOffset = token * dim;

        float gatePartialSum = 0.0f;
        float upPartialSum = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float inputValue = inputBatch.get(inputOffset + column);
                float gateScale = gateExperts.getHalfFloat(blockByteOffset).getFloat32();
                float upScale = upExperts.getHalfFloat(blockByteOffset).getFloat32();

                gatePartialSum += gateExperts.get(quantOffset) * gateScale * inputValue;
                upPartialSum += upExperts.get(quantOffset) * upScale * inputValue;
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        localSums[localId] = gatePartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        float gate = localSums[0];

        localSums[localId] = upPartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            float up = localSums[0];
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            int outputOffset = groupedPosition * moeHiddenDim + rowId;
            groupedExpertHidden.set(outputOffset, siluGate * up);
        }
    }

    /** Down-projects every routed assignment in expert-grouped order. */
    public static void groupedRoutedExpertsDownQ8_0(
            KernelContext context,
            FloatArray groupedExpertHidden,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            IntArray activeBatchSizeHolder,
            ByteArray downExperts,
            FloatArray groupedExpertDown,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int groupedPosition = flatGroupId / dim;
        int rowId = flatGroupId % dim;
        int numberOfAssignments = activeBatchSizeHolder.get(0) * topK;

        boolean active = groupedPosition < numberOfAssignments;
        int assignment = 0;
        int expert = 0;
        if (active) {
            assignment = groupedAssignmentIds.get(groupedPosition);
            expert = selectedExperts.get(assignment);
            active = expert >= 0 && expert < numberOfExperts;
        }

        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * dim + rowId) * blocksPerRow;
        int hiddenOffset = groupedPosition * moeHiddenDim;

        float partialSum = 0.0f;
        if (active) {
            for (int column = localId; column < moeHiddenDim; column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float weight =
                        downExperts.get(quantOffset)
                                * downExperts.getHalfFloat(blockByteOffset).getFloat32();
                partialSum += weight * groupedExpertHidden.get(hiddenOffset + column);
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialSum;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            int outputOffset = groupedPosition * dim + rowId;
            groupedExpertDown.set(outputOffset, localSums[0]);
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

    /** Computes the shared expert Gate/Up result independently for each token. */
    public static void batchedSharedExpertGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHiddenBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int token = flatGroupId / sharedExpertHiddenDim;
        int rowId = flatGroupId % sharedExpertHiddenDim;
        boolean active = token < activeBatchSizeHolder.get(0);

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        int inputOffset = token * dim;

        float gatePartialSum = 0.0f;
        float upPartialSum = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float inputValue = inputBatch.get(inputOffset + column);
                float gateScale = sharedGate.getHalfFloat(blockByteOffset).getFloat32();
                float upScale = sharedUp.getHalfFloat(blockByteOffset).getFloat32();

                gatePartialSum += sharedGate.get(quantOffset) * gateScale * inputValue;
                upPartialSum += sharedUp.get(quantOffset) * upScale * inputValue;
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        localSums[localId] = gatePartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        float gate = localSums[0];

        localSums[localId] = upPartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            float up = localSums[0];
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            int outputOffset = token * sharedExpertHiddenDim + rowId;
            sharedHiddenBatch.set(outputOffset, siluGate * up);
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

    /** Down-projects the shared expert and adds it to each token's residual. */
    public static void batchedSharedExpertDownAndAccumulateQ8_0(
            KernelContext context,
            FloatArray sharedHiddenBatch,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedDown,
            FloatArray residualBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int token = flatGroupId / dim;
        int rowId = flatGroupId % dim;
        boolean active = token < activeBatchSizeHolder.get(0);

        int blocksPerRow = (sharedExpertHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        int hiddenOffset = token * sharedExpertHiddenDim;

        float partialSum = 0.0f;
        if (active) {
            for (int column = localId;
                    column < sharedExpertHiddenDim;
                    column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float weight =
                        sharedDown.get(quantOffset)
                                * sharedDown.getHalfFloat(blockByteOffset).getFloat32();
                partialSum += weight * sharedHiddenBatch.get(hiddenOffset + column);
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialSum;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            int outputOffset = token * dim + rowId;
            float weightedOutput = sharedWeightBatch.get(token) * localSums[0];
            residualBatch.set(outputOffset, residualBatch.get(outputOffset) + weightedOutput);
        }
    }
}
