package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** TornadoVM kernels specific to Qwen2-MoE / Qwen1.5-MoE inference. */
public final class Qwen2MoEKernels {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private Qwen2MoEKernels() {}

    /**
     * Converts router scores to probabilities and selects the highest-scoring {@code topK} experts
     * for one token.
     *
     * <p>Inputs and outputs are GPU-resident TornadoVM arrays:
     *
     * <ul>
     *   <li>{@code routerLogits}: one raw score per expert
     *   <li>{@code selectedExperts}: output expert indices, length {@code topK}
     *   <li>{@code routingWeights}: output probabilities, length {@code topK}
     * </ul>
     *
     * <p>The first implementation deliberately uses one GPU thread for this small operation (60
     * experts in the target model). A later optimization can parallelize the reductions within one
     * workgroup.
     */
    public static void softmaxAndTopK(
            KernelContext context,
            FloatArray routerLogits,
            IntArray selectedExperts,
            FloatArray routingWeights,
            int numberOfExperts,
            int topK) {

        // The first correctness-oriented implementation is serial. Without
        // this guard, every thread would race to overwrite the same buffers.
        if (context.groupIdx != 0 || context.localIdx != 0) {
            return;
        }

        // Find the maximum first to keep the softmax numerically stable.
        float maxLogit = Float.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            float logit = routerLogits.get(expert);
            if (logit > maxLogit) {
                maxLogit = logit;
                maxIndex = expert;
            }
        }

        // Convert every router score to a probability over all experts.
        float sumExp = 0.0f;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            sumExp += TornadoMath.exp(routerLogits.get(expert) - maxLogit);
        }
        for (int expert = 0; expert < numberOfExperts; expert++) {
            float probability = TornadoMath.exp(routerLogits.get(expert) - maxLogit) / sumExp;
            routerLogits.set(expert, probability);
        }

        // Select the top-K probabilities without renormalizing their sum.
        selectedExperts.set(0, maxIndex);
        routingWeights.set(0, routerLogits.get(maxIndex));
        routerLogits.set(maxIndex, Float.NEGATIVE_INFINITY);

        for (int slot = 1; slot < topK; slot++) {
            maxLogit = Float.NEGATIVE_INFINITY;
            maxIndex = -1;
            for (int expert = 0; expert < numberOfExperts; expert++) {
                if (routerLogits.get(expert) > maxLogit) {
                    maxLogit = routerLogits.get(expert);
                    maxIndex = expert;
                }
            }
            selectedExperts.set(slot, maxIndex);
            routingWeights.set(slot, routerLogits.get(maxIndex));
            routerLogits.set(maxIndex, Float.NEGATIVE_INFINITY);
        }
    }

    /**
     * Gate/Up + SiLU for <b>all</b> routed slots in a single launch.
     *
     * <p>The slot index is folded into the work-group id, so all top-K slots execute in one launch.
     * Each slot writes its own {@code moeHiddenDim}-sized window of {@code expertHidden}, so the
     * slots never alias.
     */
    public static void fusedRoutedExpertsGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray input,
            IntArray selectedExperts,
            int expertsUsed,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray expertHidden,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int slot = flatGroupId / moeHiddenDim;
        int rowId = flatGroupId - slot * moeHiddenDim;

        // A work-group whose slot or row falls outside the launch still has to reach the
        // barriers below, so the guard only suppresses the memory accesses and the store.
        boolean active = slot < expertsUsed && rowId < moeHiddenDim;
        int expert = 0;
        if (active) {
            expert = selectedExperts.get(slot);
            active = expert >= 0 && expert < numberOfExperts;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * moeHiddenDim + rowId) * blocksPerRow;

        float gatePartialSum = 0.0f;
        float upPartialSum = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {
                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float inputValue = input.get(column);
                float gateScale = gateExperts.getHalfFloat(blockByteOffset).getFloat32();
                float upScale = upExperts.getHalfFloat(blockByteOffset).getFloat32();

                gatePartialSum += (gateExperts.get(quantOffset) * gateScale) * inputValue;
                upPartialSum += (upExperts.get(quantOffset) * upScale) * inputValue;
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
            expertHidden.set(slot * moeHiddenDim + rowId, siluGate * up);
        }
    }

    /**
     * Down-projects <b>all</b> routed slots and accumulates them into the residual in one launch.
     *
     * <p>Each slot retains the original reduction tree and is accumulated in slot order. The
     * optimization removes the per-slot kernel launches and writes the final residual once.
     */
    public static void routedExpertsDownProjectAndAccumulateQ8_0(
            KernelContext context,
            FloatArray expertHidden,
            FloatArray residual,
            IntArray selectedExperts,
            FloatArray routingWeights,
            int expertsUsed,
            ByteArray downExperts,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        boolean active = rowId < dim;

        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        // Lane 0 carries the running residual across slots. Each slot is reduced with the same
        // tree and folded in the same order as the per-slot kernels, so the result is bit-identical
        // to launching them separately - only the launch count and the residual write change.
        float running = 0.0f;
        if (localId == 0 && active) {
            running = residual.get(rowId);
        }

        for (int slot = 0; slot < expertsUsed; slot++) {
            int expert = selectedExperts.get(slot);
            boolean slotActive = active && expert >= 0 && expert < numberOfExperts;

            float partialSum = 0.0f;
            if (slotActive) {
                int rowBlockOffset = (expert * dim + rowId) * blocksPerRow;
                int hiddenBase = slot * moeHiddenDim;
                for (int column = localId; column < moeHiddenDim; column += localWorkGroupSize) {
                    int blockByteOffset =
                            (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                    int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                    float weight =
                            downExperts.get(quantOffset)
                                    * downExperts.getHalfFloat(blockByteOffset).getFloat32();
                    partialSum += weight * expertHidden.get(hiddenBase + column);
                }
            }

            context.localBarrier();
            localSums[localId] = partialSum;
            context.localBarrier();
            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0 && slotActive) {
                running += routingWeights.get(slot) * localSums[0];
            }
        }

        if (localId == 0 && active) {
            residual.set(rowId, running);
        }
    }

    /** Computes {@code SiLU(sharedGate * input) * (sharedUp * input)}. */
    public static void sharedExpertGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray input,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHidden,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        // One workgroup computes one shared-expert hidden output row.
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= sharedExpertHiddenDim) {
            return;
        }

        // Number of Q8_0 blocks needed for one row of dim weights.
        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

        // Index of this row's first Q8_0 block in the shared weight array.
        int rowBlockOffset = rowId * blocksPerRow;

        float gatePartialSum = 0.0f;
        float upPartialSum = 0.0f;

        for (int column = localId; column < dim; column += localWorkGroupSize) {

            //  Byte offset of the first byte of the Q8_0 block that contains this column.
            // Each block occupies 34 bytes: a 2-byte FP16 scale plus 32 int8 quants.
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;

            // Skip the 2-byte scale at the block start and locate this column's int8 quant.
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

            float inputValue = input.get(column);

            // getHalfFloat reads the FP16 scale from the first two block bytes, then converts it to
            // FP32.
            float gateScale = sharedGate.getHalfFloat(blockByteOffset).getFloat32();
            float upScale = sharedUp.getHalfFloat(blockByteOffset).getFloat32();

            byte gateQuant = sharedGate.get(quantOffset);
            byte upQuant = sharedUp.get(quantOffset);

            float gateWeight = gateQuant * gateScale;
            float upWeight = upQuant * upScale;

            gatePartialSum += gateWeight * inputValue;
            upPartialSum += upWeight * inputValue;
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

        // Reuse local memory to sum the partial up values.
        localSums[localId] = upPartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        // One thread writes this output row after both reductions are complete.
        if (localId == 0) {
            float up = localSums[0];
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            sharedHidden.set(rowId, siluGate * up);
        }
    }

    /** Down-projects the shared expert hidden vector into the model dimension. */
    public static void sharedExpertDownProjectQ8_0(
            KernelContext context,
            FloatArray sharedHidden,
            ByteArray sharedDown,
            FloatArray sharedOutput,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= dim) {
            return;
        }

        // Number of Q8_0 blocks needed for one row of dim weights.
        int blocksPerRow = (sharedExpertHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

        // Index of this row's first Q8_0 block in the shared weight array.
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int column = localId; column < sharedExpertHiddenDim; column += localWorkGroupSize) {
            // The start byte of the Q8_0 block holding this down-projection weight.
            // Block layout: a 2-byte FP16 scale followed by 32 int8 quants.
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;

            // Quants begin immediately after the scale; column % 32 is the index within this block.
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

            float weight =
                    sharedDown.get(quantOffset)
                            * sharedDown.getHalfFloat(blockByteOffset).getFloat32();
            partialSum += weight * sharedHidden.get(column);
        }

        // Combine all thread-local partial sums into the completed output row.
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0) {
            float outputValue = localSums[0];
            sharedOutput.set(rowId, outputValue);
        }
    }

    /** Computes the shared gate sigmoid and adds the weighted shared output to the residual. */
    public static void sharedExpertGateAndAccumulate(
            KernelContext context,
            FloatArray input,
            FloatArray sharedGateInput,
            FloatArray sharedOutput,
            FloatArray residual,
            int dim,
            int localWorkGroupSize) {
        int localId = context.localIdx;

        float partialScore = 0.0f;

        for (int column = localId; column < dim; column += localWorkGroupSize) {
            partialScore += sharedGateInput.get(column) * input.get(column);
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
        float gateScore = localSums[0];
        float sharedWeight = 1.0f / (1.0f + TornadoMath.exp(-gateScore));

        for (int index = localId; index < dim; index += localWorkGroupSize) {
            residual.set(index, residual.get(index) + sharedWeight * sharedOutput.get(index));
        }
    }
}
