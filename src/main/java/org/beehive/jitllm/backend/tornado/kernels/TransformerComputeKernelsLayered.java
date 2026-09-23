package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

public class TransformerComputeKernelsLayered {

    /** Default constructor for the TransformerComputeKernelsLayered class. */
    public TransformerComputeKernelsLayered() {}

    public static void fusedQKvBiasAddition(
            KernelContext context,
            FloatArray q_out,
            FloatArray k_out,
            FloatArray qBias,
            FloatArray v_out,
            FloatArray kBias,
            FloatArray vBias,
            int dimQ,
            int dimKV) {

        int gid = context.globalIdx;

        if (gid < dimQ) {
            // 1. Add Q bias
            q_out.set(gid, q_out.get(gid) + qBias.get(gid));

            // 2. Conditionally Add K and V Bias
            if (gid < dimKV) {
                k_out.set(gid, k_out.get(gid) + kBias.get(gid));
                v_out.set(gid, v_out.get(gid) + vBias.get(gid));
            }
        }
    }

    public static void fusedRmsNormFFNGateUp(
            KernelContext context,
            FloatArray x, // raw input (FP32)
            FloatArray hb, // output
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim, // input dimension
            int hiddenDim, // output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);

        // Allocate shared memory for normalized input (reused for both W1 and W3)
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        int rowOffsetW1 = rowId * dim;
        int rowOffsetW3 = rowId * dim;

        // === W1 matmul with inline normalization ===
        // Weights are read as packed FP16 pairs (single 32-bit loads); dim is even for all
        // supported models, so the pair indices stay even as getHalf2 requires.
        float sum1 = 0.0f;
        for (int j = localId * 2; j < dim; j += localWorkGroupSize * 2) {
            float normalized0 = rmsWeights.get(j) * scale * x.get(j);
            float normalized1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            Half2 pairW1 = w1.getHalf2(rowOffsetW1 + j);
            sum1 += Half2.lowFloat(pairW1) * normalized0;
            sum1 += Half2.highFloat(pairW1) * normalized1;
        }

        localSum[localId] = sum1;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];

        // === W3 matmul with inline normalization (same computation) ===
        float sum3 = 0.0f;
        for (int j = localId * 2; j < dim; j += localWorkGroupSize * 2) {
            float normalized0 = rmsWeights.get(j) * scale * x.get(j);
            float normalized1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            Half2 pairW3 = w3.getHalf2(rowOffsetW3 + j);
            sum3 += Half2.lowFloat(pairW3) * normalized0;
            sum3 += Half2.highFloat(pairW3) * normalized1;
        }

        localSum[localId] = sum3;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        // === SiLU + GLU ===
        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            hb.set(rowId, silu * result3);
        }
    }

    /**
     * Warp-shuffle variant of {@link #fusedRmsNormFFNGateUp}. Assumes a 32-lane workgroup per
     * output row (the decode worker shape). Reduces each row's dot product with {@code
     * simdShuffleDown} instead of a shared-memory tree, eliminating the per-row barriers and shared
     * round-trip — the same reduction strategy as llama.cpp's {@code mul_mat_vec}. Used on CUDA
     * only (see {@code SchedulerDetectionService.isWarpShuffleSupported}); OpenCL miscompiles the
     * shuffle.
     */
    public static void fusedRmsNormFFNGateUpWarp(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);
        int rowOffset = rowId * dim;

        float sum1 = 0.0f;
        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += 32) {
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            sum1 += w1.get(rowOffset + j).getFloat32() * normalized;
            sum3 += w3.get(rowOffset + j).getFloat32() * normalized;
        }

        sum1 += context.simdShuffleDown(sum1, 16);
        sum1 += context.simdShuffleDown(sum1, 8);
        sum1 += context.simdShuffleDown(sum1, 4);
        sum1 += context.simdShuffleDown(sum1, 2);
        sum1 += context.simdShuffleDown(sum1, 1);

        sum3 += context.simdShuffleDown(sum3, 16);
        sum3 += context.simdShuffleDown(sum3, 8);
        sum3 += context.simdShuffleDown(sum3, 4);
        sum3 += context.simdShuffleDown(sum3, 2);
        sum3 += context.simdShuffleDown(sum3, 1);

        if (localId == 0) {
            float silu = sum1 / (1.0f + TornadoMath.exp(-sum1));
            hb.set(rowId, silu * sum3);
        }
    }

    /**
     * Warp-shuffle variant of {@link #fusedRmsNormFFNGateUpQ8_0}: one 32-lane warp per hidden row,
     * two independent {@code simdShuffleDown} reductions (gate W1 and up W3), no shared-memory
     * tree. Q8_0 byte layout (34-byte blocks: 2-byte half scale + 32 int8 quants).
     */
    public static void fusedRmsNormFFNGateUpQ8_0Warp(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            ByteArray w1,
            ByteArray w3,
            int inputDim,
            int hiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2-byte scale + 32 int8 quants
        int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow;

        float sum1 = 0.0f;
        float sum3 = 0.0f;
        for (int j = localId; j < inputDim; j += 32) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j - blockIdx * blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            float w1s = w1.getHalfFloat(blockByteOffset).getFloat32();
            float w3s = w3.getHalfFloat(blockByteOffset).getFloat32();
            byte w1q = w1.get(blockByteOffset + 2 + withinBlockIdx);
            byte w3q = w3.get(blockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * (scale * x.get(j));
            sum1 += ((float) w1q * w1s) * normalized;
            sum3 += ((float) w3q * w3s) * normalized;
        }

        sum1 += context.simdShuffleDown(sum1, 16);
        sum1 += context.simdShuffleDown(sum1, 8);
        sum1 += context.simdShuffleDown(sum1, 4);
        sum1 += context.simdShuffleDown(sum1, 2);
        sum1 += context.simdShuffleDown(sum1, 1);

        sum3 += context.simdShuffleDown(sum3, 16);
        sum3 += context.simdShuffleDown(sum3, 8);
        sum3 += context.simdShuffleDown(sum3, 4);
        sum3 += context.simdShuffleDown(sum3, 2);
        sum3 += context.simdShuffleDown(sum3, 1);

        if (localId == 0) {
            float silu = sum1 / (1.0f + TornadoMath.exp(-sum1));
            hb.set(rowId, silu * sum3);
        }
    }

    /**
     * Fused RMSNorm apply + Gate/Up projection + SiLU + GLU for Q8_0 weights. Combines:
     * reductionOneBlock2WithLayer + fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte
     */
    public static void fusedRmsNormFFNGateUpQ8_0(
            KernelContext context,
            FloatArray x, // raw input (FP32)
            FloatArray hb, // output: SiLU(x·W1) ⊙ (x·W3)
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // tempFFN[0] = scale factor
            ByteArray w1, // W1 (gate) Q8_0 weights
            ByteArray w3, // W3 (up) Q8_0 weights
            int inputDim, // input dimension
            int hiddenDim, // hidden dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        // Calculate block offsets for W1 and W3 matrices
        int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
        int w1RowBlockOffset = rowId * blocksPerRow;
        int w3RowBlockOffset = rowId * blocksPerRow;

        // ========== W1 computation with inline RMS normalization ==========
        float partialSum1_1 = 0.0f,
                partialSum1_2 = 0.0f,
                partialSum1_3 = 0.0f,
                partialSum1_4 = 0.0f;

        // Main loop with 4-way unrolling for W1
        for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            // W1 block access
            int w1BlockByteOffset = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w1Scale = w1.getHalfFloat(w1BlockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            int w1QuantsOffset = w1BlockByteOffset + 2 + withinBlockIdx;
            byte w1Quant1 = w1.get(w1QuantsOffset);
            byte w1Quant2 = w1.get(w1QuantsOffset + 1);
            byte w1Quant3 = w1.get(w1QuantsOffset + 2);
            byte w1Quant4 = w1.get(w1QuantsOffset + 3);

            // Apply RMS normalization inline (equivalent to reductionOneBlock2WithLayer)
            float norm1 = rmsWeights.get(j) * (scale * x.get(j));
            float norm2 = rmsWeights.get(j + 1) * (scale * x.get(j + 1));
            float norm3 = rmsWeights.get(j + 2) * (scale * x.get(j + 2));
            float norm4 = rmsWeights.get(j + 3) * (scale * x.get(j + 3));

            partialSum1_1 += ((float) w1Quant1 * w1ScaleFloat) * norm1;
            partialSum1_2 += ((float) w1Quant2 * w1ScaleFloat) * norm2;
            partialSum1_3 += ((float) w1Quant3 * w1ScaleFloat) * norm3;
            partialSum1_4 += ((float) w1Quant4 * w1ScaleFloat) * norm4;
        }

        float partialSum1 = partialSum1_1 + partialSum1_2 + partialSum1_3 + partialSum1_4;

        // Handle remaining elements for W1
        for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w1BlockByteOffset = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w1Scale = w1.getHalfFloat(w1BlockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            byte w1Quant = w1.get(w1BlockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * (scale * x.get(j));

            partialSum1 += ((float) w1Quant * w1ScaleFloat) * normalized;
        }

        localSums[localId] = partialSum1;
        context.localBarrier();

        // Parallel reduction for W1
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        float sum1 = localSums[0];

        // ========== W3 computation with inline RMS normalization ==========
        float partialSum3_1 = 0.0f,
                partialSum3_2 = 0.0f,
                partialSum3_3 = 0.0f,
                partialSum3_4 = 0.0f;

        // Main loop with 4-way unrolling for W3
        for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            // W3 block access
            int w3BlockByteOffset = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w3Scale = w3.getHalfFloat(w3BlockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            int w3QuantsOffset = w3BlockByteOffset + 2 + withinBlockIdx;
            byte w3Quant1 = w3.get(w3QuantsOffset);
            byte w3Quant2 = w3.get(w3QuantsOffset + 1);
            byte w3Quant3 = w3.get(w3QuantsOffset + 2);
            byte w3Quant4 = w3.get(w3QuantsOffset + 3);

            // Apply RMS normalization inline (same computation as W1)
            float norm1 = rmsWeights.get(j) * (scale * x.get(j));
            float norm2 = rmsWeights.get(j + 1) * (scale * x.get(j + 1));
            float norm3 = rmsWeights.get(j + 2) * (scale * x.get(j + 2));
            float norm4 = rmsWeights.get(j + 3) * (scale * x.get(j + 3));

            partialSum3_1 += ((float) w3Quant1 * w3ScaleFloat) * norm1;
            partialSum3_2 += ((float) w3Quant2 * w3ScaleFloat) * norm2;
            partialSum3_3 += ((float) w3Quant3 * w3ScaleFloat) * norm3;
            partialSum3_4 += ((float) w3Quant4 * w3ScaleFloat) * norm4;
        }

        float partialSum3 = partialSum3_1 + partialSum3_2 + partialSum3_3 + partialSum3_4;

        // Handle remaining elements for W3
        for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w3BlockByteOffset = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w3Scale = w3.getHalfFloat(w3BlockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            byte w3Quant = w3.get(w3BlockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * (scale * x.get(j));

            partialSum3 += ((float) w3Quant * w3ScaleFloat) * normalized;
        }

        localSums[localId] = partialSum3;
        context.localBarrier();

        // Parallel reduction for W3
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        float sum3 = localSums[0];

        // ========== SiLU + GLU (same as original) ==========
        if (localId == 0) {
            float silu = siluActivation(sum1);
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    /**
     * Performs RMS (Root Mean Square) normalization using parallel reduction. This is the first
     * phase of RMS normalization that computes the variance and scaling factor across all work
     * groups.
     *
     * <p>Algorithm: 1. Each thread computes square of its input element 2. Work group performs
     * parallel reduction of squares 3. Partial sums stored per work group 4. First thread combines
     * all partial sums and computes normalization factor
     *
     * @param context Kernel execution context
     * @param output Array to store partial sums and final normalization factor
     * @param x Input array to normalize
     * @param size Number of elements to process
     * @param ermsNorm Epsilon value squared for numerical stability
     * @param localMemSize Size of local memory allocation (must match work group size)
     */
    public static void reductionOneBlockWithLayer(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int size,
            float ermsNorm,
            int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

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
            output.set(groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= (size / localMemSize); i++) { // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss); // Store the final scale factor
        }
    }

    /**
     * Applies the computed normalization factor to input and weight elements. This is the second
     * phase of RMS normalization.
     *
     * <p>Formula: output[i] = weight[i] * (normalizationFactor * x[i])
     *
     * @param context Kernel execution context
     * @param output Array for normalized output
     * @param x Input values to normalize
     * @param weights Weight values for each element
     * @param temp Temporary array containing normalization factor at index 0
     */
    public static void reductionOneBlock2WithLayer(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            FloatArray weights,
            FloatArray temp) {
        int gid = context.globalIdx;

        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * x.get(gid)));
    }

    public static void splitQKV(
            FloatArray qkv, FloatArray q, FloatArray k, FloatArray v, int dimQ, int dimKV) {
        int totalSize = dimQ + 2 * dimKV;

        for (@Parallel int i = 0; i < totalSize; i++) {
            if (i < dimQ) {
                // Copy to Q
                q.set(i, qkv.get(i));
            } else if (i < dimQ + dimKV) {
                // Copy to K
                int kIndex = i - dimQ;
                k.set(kIndex, qkv.get(i));
            } else {
                // Copy to V
                int vIndex = i - dimQ - dimKV;
                v.set(vIndex, qkv.get(i));
            }
        }
    }

    public static void ropeRotationPhi3(
            KernelContext context,
            IntArray positionHolder,
            FloatArray sq,
            FloatArray sk,
            int kv_dim,
            int head_size) {
        int idx = context.globalIdx;

        // For Phi3, we process pairs with offset of head_size/2
        int dimHalf = head_size / 2;

        // Each thread processes one dimension pair
        if (idx >= dimHalf) {
            return;
        }

        int position = positionHolder.get(0);

        // Calculate frequency for this dimension
        float freq = 1.0f / TornadoMath.pow(10000.0f, (float) (idx * 2) / (float) head_size);
        float val = position * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Process all heads
        int totalDim = sq.getSize();
        for (int base = 0; base < totalDim; base += head_size) {
            // Skip if we're beyond the bounds
            if (base + idx >= totalDim || base + idx + dimHalf >= totalDim) {
                break;
            }

            // Rotate query
            float v0 = sq.get(base + idx);
            float v1 = sq.get(base + idx + dimHalf);
            sq.set(base + idx, v0 * fcr - v1 * fci);
            sq.set(base + idx + dimHalf, v0 * fci + v1 * fcr);

            // Rotate key if within kv_dim
            if (base < kv_dim && base + idx < sk.getSize() && base + idx + dimHalf < sk.getSize()) {
                float k0 = sk.get(base + idx);
                float k1 = sk.get(base + idx + dimHalf);
                sk.set(base + idx, k0 * fcr - k1 * fci);
                sk.set(base + idx + dimHalf, k0 * fci + k1 * fcr);
            }
        }
    }

    /**
     * Qwen3-family decode attention, split-KV (flash-decoding) phase 2: combine.
     *
     * <p>One workgroup per query head. Reads the {@code nSplits} partial states ({@code out_s, M_s,
     * L_s}) produced by {@link #processHeadsFlashAttentionSplitKV} from {@code att} ({@code
     * wrapAttSplit}) and merges them with the standard online-softmax rule: global max {@code Mg =
     * max_s M_s}, scale {@code f_s = exp(M_s - Mg)}, output {@code xb[d] = Σ_s f_s·out_s[d] / Σ_s
     * f_s·L_s}. Writes the final attention result to {@code xb}.
     */
    public static void combineSplitKVAttention(
            KernelContext context,
            FloatArray att,
            FloatArray xb,
            int nHeads,
            int headSize,
            int nSplits) {

        final int MAX_SPLITS = 64;

        int tid = context.localIdx;
        int h = context.groupIdx;
        int localSize = context.localGroupSizeX;

        if (h >= nHeads) {
            return;
        }

        // Must match the COMPACT layout written by processHeadsFlashAttentionSplitKV.
        int headBase = h * nSplits * (headSize + 2);
        int mBase = headBase + nSplits * headSize;
        int lBase = mBase + nSplits;

        float[] fShared = context.allocateFloatLocalArray(MAX_SPLITS);
        float[] bcast = context.allocateFloatLocalArray(1);

        // Global max over split maxima.
        if (tid == 0) {
            float gMax = Float.NEGATIVE_INFINITY;
            for (int s = 0; s < nSplits; s++) {
                float ms = att.get(mBase + s);
                if (ms > gMax) {
                    gMax = ms;
                }
            }
            bcast[0] = gMax;
        }
        context.localBarrier();
        float gMax = bcast[0];

        // Per-split scale factors f_s = exp(M_s - gMax).
        for (int s = tid; s < nSplits; s += localSize) {
            float ms = att.get(mBase + s);
            fShared[s] = (ms == Float.NEGATIVE_INFINITY) ? 0.0f : TornadoMath.exp(ms - gMax);
        }
        context.localBarrier();

        // Global denominator Σ_s f_s · L_s.
        if (tid == 0) {
            float denom = 0.0f;
            for (int s = 0; s < nSplits; s++) {
                denom += fShared[s] * att.get(lBase + s);
            }
            bcast[0] = denom;
        }
        context.localBarrier();
        float denom = bcast[0];
        float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        // Merge per-dimension numerators across splits and normalize.
        for (int d = tid; d < headSize; d += localSize) {
            float acc = 0.0f;
            for (int s = 0; s < nSplits; s++) {
                acc += fShared[s] * att.get(headBase + s * headSize + d);
            }
            xb.set(h * headSize + d, acc * inv);
        }
    }

    /**
     * Performs optimized matrix-vector multiplication where each work group processes one row of
     * the matrix.
     *
     * <p>Algorithm: 1. Each work group handles one output dimension 2. Threads in work group
     * compute partial dot products 3. Parallel reduction yields final row result
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Output vector
     * @param w Weight matrix (row-major)
     * @param n Input dimension
     * @param d Output dimension
     * @param localWorkGroupSize Number of threads per work group
     */
    public static void matrixVectorGeneric(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }
        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum);
        }
    }

    // @formatter:off
    public static void matrixVectorGeneric(
            KernelContext context,
            FloatArray x,
            FloatArray hb, // output
            HalfFloatArray w,
            int dim1, // inner loop
            int dim0, // outer loop
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= dim0) {
            return;
        }
        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, dim1);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum);
        }
    }

    /**
     * Fused Q/K/V matrix-vector multiplication. Reduces kernel launch overhead and improves input
     * vector cache utilization.
     *
     * <p>Workgroup assignment: - rowId [0, dim): Q projection - rowId [dim, dim+kvDim): K
     * projection - rowId [dim+kvDim, dim+2*kvDim): V projection
     */
    public static void fusedQKVMatmulX(
            KernelContext context,
            HalfFloatArray x, // input vector (FP16)
            FloatArray q, // output Q (FP32)
            FloatArray k, // output K (FP32)
            FloatArray v, // output V (FP32)
            HalfFloatArray wq, // Q weight matrix
            HalfFloatArray wk, // K weight matrix
            HalfFloatArray wv, // V weight matrix
            int dim, // model dimension (Q output size)
            int kvDim, // KV dimension (K/V output size)
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < dim) {
            // ========== Q projection ==========
            int rowOffset = rowId * dim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
            // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
            for (int j = localId * 2; j < dim; j += localWorkGroupSize * 2) {
                Half2 product = Half2.mult(wq.getHalf2(rowOffset + j), x.getHalf2(j));
                partialSum += Half2.lowFloat(product);
                partialSum += Half2.highFloat(product);
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

        } else if (rowId < dim + kvDim) {
            // ========== K projection ==========
            int kRow = rowId - dim;
            int rowOffset = kRow * dim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
            // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
            for (int j = localId * 2; j < dim; j += localWorkGroupSize * 2) {
                Half2 product = Half2.mult(wk.getHalf2(rowOffset + j), x.getHalf2(j));
                partialSum += Half2.lowFloat(product);
                partialSum += Half2.highFloat(product);
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

        } else if (rowId < dim + 2 * kvDim) {
            // ========== V projection ==========
            int vRow = rowId - dim - kvDim;
            int rowOffset = vRow * dim;

            float partialSum = 0.0f;
            // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
            // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
            for (int j = localId * 2; j < dim; j += localWorkGroupSize * 2) {
                Half2 product = Half2.mult(wv.getHalf2(rowOffset + j), x.getHalf2(j));
                partialSum += Half2.lowFloat(product);
                partialSum += Half2.highFloat(product);
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
     * {@link #fusedQKVMatmulX} with the products taken in FP32 instead of packed FP16.
     *
     * <p>Identical structure and identical work decomposition; the only difference is that each
     * FP16 pair is widened before it is multiplied, rather than multiplied by {@code __hmul2} and
     * widened afterwards. That rounds every product to FP16 before it reaches the FP32 accumulator,
     * once per term, and over a 2048-term row the loss is systematic rather than cancelling.
     *
     * <p>Selected where {@code DeviceCapability.PACKED_HALF2_MATH} is withheld. It is measurably
     * more accurate everywhere; it is not the default because the packed form is faster where it
     * has been shown to hold parity.
     */
    public static void fusedQKVMatmulXFp32Products(
            KernelContext context,
            HalfFloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < dim) {
            int rowOffset = rowId * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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

        } else if (rowId < dim + kvDim) {
            int kRow = rowId - dim;
            int rowOffset = kRow * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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

        } else if (rowId < dim + 2 * kvDim) {
            int vRow = rowId - dim - kvDim;
            int rowOffset = vRow * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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
     * CUDA/Metal SIMD variant of fusedQKVMatmulX for LLaMA FP16 decode. It assumes the existing
     * decode worker shape: one 32-lane workgroup per Q/K/V output row.
     */
    public static void fusedQKVMatmulXSimd32(
            KernelContext context,
            HalfFloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int kvDim) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId < dim) {
            int rowOffset = rowId * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += 32) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                q.set(rowId, partialSum);
            }

        } else if (rowId < dim + kvDim) {
            int kRow = rowId - dim;
            int rowOffset = kRow * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += 32) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
            }
            partialSum += context.simdShuffleDown(partialSum, 16);
            partialSum += context.simdShuffleDown(partialSum, 8);
            partialSum += context.simdShuffleDown(partialSum, 4);
            partialSum += context.simdShuffleDown(partialSum, 2);
            partialSum += context.simdShuffleDown(partialSum, 1);
            if (localId == 0) {
                k.set(kRow, partialSum);
            }

        } else if (rowId < dim + 2 * kvDim) {
            int vRow = rowId - dim - kvDim;
            int rowOffset = vRow * dim;
            float partialSum = 0.0f;
            for (int j = localId; j < dim; j += 32) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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

    /** Fused QKV matmul for FP16 models where Q output dim != input dim. */
    public static void fusedQKVMatmulXNonSquare(
            KernelContext context,
            HalfFloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            int rowOffset = rowId * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;
            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
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

    // @formatter:off
    public static void matrixVectorGeneric(
            KernelContext context,
            HalfFloatArray x,
            FloatArray hb, // output
            HalfFloatArray w,
            int dim1, // inner loop
            int dim0, // outer loop
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= dim0) {
            return;
        }
        float sum = matrixVectorRowMajorOptimizedSingle(context, localSize, x, w, dim1);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum);
        }
    }

    // @formatter:on

    /**
     * CUDA/Metal SIMD variant of {@link #matrixVectorGeneric(KernelContext, HalfFloatArray,
     * FloatArray, HalfFloatArray, int, int, int)} — same kernel shape as {@link
     * #matrixVectorGenericWithResidualSimd32}, minus the residual add: this call site (the
     * vocabulary projection) has no residual to accumulate into. Assumes the existing decode worker
     * shape: one 32-lane workgroup per output row.
     */
    public static void matrixVectorGenericSimd32(
            KernelContext context,
            HalfFloatArray x,
            FloatArray hb,
            HalfFloatArray w,
            int n,
            int d) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        int rowOffset = rowId * n;
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            partialSum += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);

        if (localId == 0) {
            hb.set(rowId, partialSum);
        }
    }

    /**
     * Matrix-vector multiplication with residual connection. Combines regular matrix multiplication
     * with addition of existing values.
     *
     * <p>Formula: hb[i] = hb[i] + w[i]·x
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Input/output vector (contains residual, receives result)
     * @param w Weight matrix
     * @param n Input dimension
     * @param d Output dimension
     * @param localWorkGroupSize Work group size
     */
    public static void matrixVectorGenericWithResidual(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            HalfFloatArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    /**
     * CUDA/Metal SIMD variant of matrixVectorGenericWithResidual for LLaMA FP16 decode. It assumes
     * the existing decode worker shape: one 32-lane workgroup per output row.
     */
    public static void matrixVectorGenericWithResidualSimd32(
            KernelContext context, FloatArray x, FloatArray hb, HalfFloatArray w, int n, int d) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        int rowOffset = rowId * n;
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            partialSum += w.get(rowOffset + j).getFloat32() * x.get(j);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);

        if (localId == 0) {
            hb.set(rowId, hb.get(rowId) + partialSum);
        }
    }

    public static void matrixVectorGenericWithResidual(
            KernelContext context,
            HalfFloatArray x,
            FloatArray hb,
            HalfFloatArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    /**
     * Fused feed-forward network with SiLU activation and GLU gating. Implements the SwiGLU variant
     * used in LLaMA-style models.
     *
     * <p>Formula: FFN(x) = SiLU(x·W1) ⊙ (x·W3) where ⊙ denotes element-wise multiplication
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Output buffer
     * @param w1 First feed-forward weight matrix
     * @param w3 Third feed-forward weight matrix (gate)
     * @param n Input dimension
     * @param d Hidden dimension
     * @param localWorkGroupSize Work group size
     */
    public static void fusedFeedForwardWithSiLUAndGLUActivation(
            KernelContext context,
            HalfFloatArray x,
            HalfFloatArray hb,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n);
        float sum3 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1); // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, new HalfFloat(result));
        }
    }

    public static void fusedFeedForwardWithSiLUAndGLUActivation(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n);
        float sum3 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1); // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    public static void fusedFeedForwardWithSiLUAndGLUActivation(
            KernelContext context,
            HalfFloatArray x,
            FloatArray hb,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        HalfFloat sum1 = matrixVectorRowMajorOptimizedFHF(context, localWorkGroupSize, x, w1, n);
        HalfFloat sum3 = matrixVectorRowMajorOptimizedFHF(context, localWorkGroupSize, x, w3, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1.getFloat32()); // Using the new SiLU method
            float result = silu * sum3.getFloat32();
            hb.set(rowId, result);
        }
    }

    /**
     * Gaussian Error Linear Unit (GELU) activation function. Approximation formula: GELU(x) ≈ 0.5 *
     * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     *
     * @param x Input value
     * @return Activated value
     */
    public static float geluActivation(float x) {
        float x3 = x * x * x;
        return 0.5f * x * (1.0f + TornadoMath.tanh((0.797885f * (x + 0.044715f * x3))));
    }

    /**
     * Sigmoid-weighted Linear Unit (SiLU) activation function. Also known as Swish activation.
     *
     * <p>Formula: SiLU(x) = x * σ(x) = x / (1 + e^(-x))
     *
     * @param x Input value
     * @return Activated value
     */
    public static float siluActivation(float x) {
        return x * (1.0f / (1.0f + TornadoMath.exp(-x)));
    }

    /**
     * Optimized row-major matrix-vector multiplication for a single row. Uses parallel reduction
     * within a work group to compute one dot product.
     *
     * <p>Algorithm: 1. Each thread computes partial dot product 2. Partial results stored in local
     * memory 3. Tree-based reduction combines partial results 4. Returns final dot product for the
     * row
     *
     * @param context Kernel execution context
     * @param localSize Work group size
     * @param x Input vector
     * @param w Weight matrix row
     * @param n Input dimension
     * @return Dot product result for this row
     */
    public static float matrixVectorRowMajorOptimized(
            KernelContext context, int localSize, FloatArray x, FloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            partialSum += w.get(matrixIdx) * x.get(j);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static float matrixVectorRowMajorOptimized(
            KernelContext context, int localSize, FloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread accumulates over consecutive FP16 pairs read with a single packed
        // 32-bit load; consecutive threads read consecutive pairs, so a warp still issues
        // fully coalesced 128-byte transactions with half the memory instructions.
        // Rows start at rowId * n with n even for all supported models, so pair indices
        // stay even (4-byte aligned) as required by getHalf2.
        float partialSum = 0.0f;
        int nEven = n & ~1;
        for (int j = localId * 2; j < nEven; j += localSize * 2) {
            int matrixIdx = rowOffset + j;
            Half2 pair = w.getHalf2(matrixIdx);
            partialSum += Half2.lowFloat(pair) * x.get(j);
            partialSum += Half2.highFloat(pair) * x.get(j + 1);
        }
        if (nEven != n && localId == 0) {
            partialSum += w.get(rowOffset + nEven).getFloat32() * x.get(nEven);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static HalfFloat matrixVectorRowMajorOptimizedFHF(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        HalfFloat[] localSum = context.allocateHalfFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        //        HalfFloat partialSum = new HalfFloat(0f);
        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            //            HalfFloat mul = HalfFloat.mult(w.get(matrixIdx), x.get(j));
            partialSum += w.get(matrixIdx).getFloat32() * x.get(j).getFloat32();
            //            partialSum = HalfFloat.add(partialSum, mul);
        }

        // Store partial sum in local memory
        localSum[localId] = new HalfFloat(partialSum);
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] = HalfFloat.add(localSum[localId], localSum[localId + stride]);
                //                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static float matrixVectorRowMajorOptimizedF(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        //        HalfFloat partialSum = new HalfFloat(0f);
        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            //            HalfFloat mul = HalfFloat.mult(w.get(matrixIdx), x.get(j));
            partialSum += w.get(matrixIdx).getFloat32() * x.get(j).getFloat32();
            //            partialSum = HalfFloat.add(partialSum, mul);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static float matrixVectorRowMajorOptimizedXX(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        int stride = localSize;
        int stride2 = localSize << 1;
        int stride3 = localSize * 3;
        int stride4 = localSize << 2;

        // Already coalesced: thread 0 reads idx 0, thread 1 reads idx 1, etc.
        int j = localId;
        int limit = n - stride3;

        for (; j < limit; j += stride4) {
            int base = rowOffset + j;
            // Hoist x.get() calls - they're reused across all rows
            float x0 = x.get(j).getFloat32();
            float x1 = x.get(j + stride).getFloat32();
            float x2 = x.get(j + stride2).getFloat32();
            float x3 = x.get(j + stride3).getFloat32();

            sum0 += w.get(base).getFloat32() * x0;
            sum1 += w.get(base + stride).getFloat32() * x1;
            sum2 += w.get(base + stride2).getFloat32() * x2;
            sum3 += w.get(base + stride3).getFloat32() * x3;
        }

        for (; j < n; j += stride) {
            sum0 += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
        }

        localSum[localId] = (sum0 + sum1) + (sum2 + sum3);
        context.localBarrier();

        // Reduction with minimal barriers
        for (int s = localSize >> 1; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static float matrixVectorRowMajorOptimizedSingle(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Both operands are FP16: read them as packed pairs (single 32-bit loads), multiply
        // with the packed __hmul2 intrinsic and accumulate in FP32. n is even.
        float partialSum = 0.0f;
        for (int j = localId * 2; j < n; j += localSize * 2) {
            int matrixIdx = rowOffset + j;
            Half2 product = Half2.mult(w.getHalf2(matrixIdx), x.getHalf2(j));
            partialSum += Half2.lowFloat(product);
            partialSum += Half2.highFloat(product);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static float matrixVectorRowMajorOptimized(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Accumulate in HalfFloat to avoid conversions in inner loop
        HalfFloat sum0 = new HalfFloat(0f);
        HalfFloat sum1 = new HalfFloat(0f);
        HalfFloat sum2 = new HalfFloat(0f);
        HalfFloat sum3 = new HalfFloat(0f);

        int stride = localSize;
        int stride2 = localSize << 1;
        int stride3 = localSize * 3;
        int stride4 = localSize << 2;

        int j = localId;
        int limit = n - stride3;

        for (; j < limit; j += stride4) {
            int base = rowOffset + j;

            // Stay in HalfFloat - no getFloat32() calls
            HalfFloat x0 = x.get(j);
            HalfFloat x1 = x.get(j + stride);
            HalfFloat x2 = x.get(j + stride2);
            HalfFloat x3 = x.get(j + stride3);

            sum0 = HalfFloat.add(sum0, HalfFloat.mult(w.get(base), x0));
            sum1 = HalfFloat.add(sum1, HalfFloat.mult(w.get(base + stride), x1));
            sum2 = HalfFloat.add(sum2, HalfFloat.mult(w.get(base + stride2), x2));
            sum3 = HalfFloat.add(sum3, HalfFloat.mult(w.get(base + stride3), x3));
        }

        // Cleanup loop
        for (; j < n; j += stride) {
            sum0 = HalfFloat.add(sum0, HalfFloat.mult(w.get(rowOffset + j), x.get(j)));
        }

        // Convert to float32 only at the end for reduction
        localSum[localId] =
                sum0.getFloat32() + sum1.getFloat32() + sum2.getFloat32() + sum3.getFloat32();
        context.localBarrier();

        for (int s = localSize >> 1; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    public static void fusedQKVMatmul(
            KernelContext context,
            HalfFloatArray x, // input (read once!)
            FloatArray q,
            FloatArray k,
            FloatArray v, // outputs
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Determine which output this workgroup computes
        int totalRows = dim + 2 * kvDim; // Q rows + K rows + V rows

        if (rowId < dim) {
            // Q projection
            float sum = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wq, dim);
            if (localId == 0) {
                q.set(rowId, sum);
            }
        } else if (rowId < dim + kvDim) {
            // K projection
            int kRow = rowId - dim;
            float sum = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wk, dim);
            if (localId == 0) {
                k.set(kRow, sum);
            }
        } else {
            // V projection
            int vRow = rowId - dim - kvDim;
            float sum = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wv, dim);
            if (localId == 0) {
                v.set(vRow, sum);
            }
        }
    }

    public static float matrixVectorRowMajorOptimizedx(
            KernelContext context, int localSize, HalfFloatArray x, HalfFloatArray w, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread calculates partial dot product - UNROLLED BY 4
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        int j = localId;
        int stride = localSize;
        int stride4 = localSize << 2; // localSize * 4
        int limit = n - (stride * 3); // Safe limit for 4 elements

        // Main loop unrolled by 4 with separate accumulators
        for (; j < limit; j += stride4) {
            int base = rowOffset + j;
            int j1 = j + stride;
            int j2 = j + (stride << 1);
            int j3 = j + stride * 3;

            sum0 += w.get(base).getFloat32() * x.get(j).getFloat32();
            sum1 += w.get(base + stride).getFloat32() * x.get(j1).getFloat32();
            sum2 += w.get(base + (stride << 1)).getFloat32() * x.get(j2).getFloat32();
            sum3 += w.get(base + stride * 3).getFloat32() * x.get(j3).getFloat32();
        }

        // Handle remainder
        for (; j < n; j += stride) {
            sum0 += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32();
        }

        // Combine accumulators (tree reduction for better precision)
        float partialSum = (sum0 + sum1) + (sum2 + sum3);

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int s = localSize >> 1; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }

        return localSum[0];
    }

    /**
     * Race-free RMS-norm reduction in ONE workgroup: threads stride over the input, reduce in local
     * memory, and lane 0 writes the final scale factor to {@code output[0]}.
     *
     * <p>The multi-workgroup variant ({@code reductionOneBlockWithLayer}) has workgroup 0 combining
     * the other workgroups' partial sums with <em>no inter-workgroup synchronization</em> — a data
     * race. On the CUDA backend the race outcome is schedule/compilation dependent: e.g. Qwen3-1.7B
     * reads stale partials on every layer (wrong normalization scale → garbage output) while
     * Llama-1B and Qwen3-4B happen to win the race. The NON_NVIDIA scheduler avoids the race with a
     * separate {@code reductionFinalNormalization} task; this kernel is the NVIDIA-path equivalent.
     * Launch with exactly one workgroup: {@code global == local == localMemSize}.
     */
    public static void reductionOneBlockWithLayerSingleGroup(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int size,
            float ermsNorm,
            int localMemSize) {
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        float partial = 0.0f;
        for (int j = lid; j < size; j += groupSize) {
            float v = x.get(j);
            partial += v * v;
        }
        localX[lid] = partial;

        for (int stride = groupSize / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        if (lid == 0) {
            float ss = localX[0] / size + ermsNorm;
            output.set(0, 1.0f / TornadoMath.sqrt(ss));
        }
    }

    // Second kernel - Combines partial sums and computes final normalization
    public static void reductionFinalNormalization(
            KernelContext context, FloatArray output, int size, float ermsNorm) {
        int gid = context.globalIdx;

        // Only one thread needs to perform this calculation
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i < output.getSize(); i++) { // Fixed bounds to avoid out of bounds
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss); // Store the final scale factor
        }
    }

    public static void splitGateUpAndSiLU(
            FloatArray hb, FloatArray hbG, FloatArray hbU, int hiddenDim) {
        // Copy and apply SiLU to gate in one pass
        for (@Parallel int i = 0; i < hiddenDim; i++) {
            float gateVal = hb.get(i);
            float upVal = hb.get(hiddenDim + i);

            // Apply SiLU to gate
            float siluGate = gateVal / (1.0f + TornadoMath.exp(-gateVal));

            // Store activated gate and multiply with up
            hbG.set(i, siluGate);
            hbU.set(i, siluGate * upVal);
        }
    }

    public static void addInPlace(FloatArray arrayA, FloatArray arrayB, int size) {
        // Element-wise addition: arrayA[i] = arrayA[i] + arrayB[i]
        for (@Parallel int i = 0; i < size; i++) {
            float result = arrayA.get(i) + arrayB.get(i);
            arrayA.set(i, result);
        }
    }

    /**
     * Matrix-vector multiplication for Q8_0 quantized weights.
     *
     * @param context Kernel context
     * @param x Input activations (FloatArray)
     * @param output Output array (FloatArray)
     * @param weightsQ Quantized weights (Int8Array) - from Q8_0QuantizedTensor.getQuants()
     * @param weightScales Scale factors (HalfFloatArray) - from Q8_0QuantizedTensor.getScales()
     * @param dim1 Input dimension (n - number of columns)
     * @param dim0 Output dimension (d - number of rows)
     * @param localWorkGroupSize Local workgroup size
     */
    public static void matrixVectorGeneric(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            Int8Array weightsQ,
            HalfFloatArray weightScales,
            int dim1,
            int dim0,
            int localWorkGroupSize) {

        // One row per workgroup
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Early exit if this workgroup is beyond output dimension
        if (rowId >= dim0) {
            return;
        }

        float sum =
                matrixVectorRowMajorOptimizedQ8_0(
                        context, localWorkGroupSize, x, weightsQ, weightScales, dim1);

        // Thread 0 writes the result
        if (localId == 0) {
            output.set(rowId, sum);
        }
    }

    /**
     * Helper method to compute dot product for a single row with Q8_0 quantized weights. Uses 4-way
     * unrolling for better performance.
     */
    public static float matrixVectorRowMajorOptimizedQ8_0(
            KernelContext context,
            int localSize,
            FloatArray x,
            Int8Array weightsQ,
            HalfFloatArray weightScales,
            int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int blockSize = 32;

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;
        int scalesRowOffset = rowId * (n / blockSize);

        // 4-way unrolling
        float partialSum1 = 0.0f;
        float partialSum2 = 0.0f;
        float partialSum3 = 0.0f;
        float partialSum4 = 0.0f;

        // Main loop - process 4 elements at a time
        for (int j = localId * 4; j < n - 3; j += localSize * 4) {
            int blockIdx = j / blockSize;
            float scale = weightScales.get(scalesRowOffset + blockIdx).getFloat32();

            // Dequantize and multiply
            partialSum1 += ((float) weightsQ.get(rowOffset + j) * scale) * x.get(j);
            partialSum2 += ((float) weightsQ.get(rowOffset + j + 1) * scale) * x.get(j + 1);
            partialSum3 += ((float) weightsQ.get(rowOffset + j + 2) * scale) * x.get(j + 2);
            partialSum4 += ((float) weightsQ.get(rowOffset + j + 3) * scale) * x.get(j + 3);
        }

        float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

        // Handle remaining elements
        for (int j = ((n / 4) * 4) + localId; j < n; j += localSize) {
            int blockIdx = j / blockSize;
            float scale = weightScales.get(scalesRowOffset + blockIdx).getFloat32();
            partialSum += ((float) weightsQ.get(rowOffset + j) * scale) * x.get(j);
        }

        // Store partial sum
        localSums[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        return localSums[0];
    }

    public static void fusedRmsNormQKVMatmulQ8(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            ByteArray wqkv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        int totalRows = dim + 2 * kvDim;
        if (rowId >= totalRows) {
            return;
        }

        float rmsScaleFactor = rmsScale.get(0);

        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        int blocksPerRow = (dim + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum1 = 0.0f;
        float partialSum2 = 0.0f;
        float partialSum3 = 0.0f;
        float partialSum4 = 0.0f;

        // Main loop - 4-way unrolled, only when we have complete groups of 4
        int mainLoopEnd = (dim / 4) * 4; // Largest multiple of 4 <= dim
        for (int j = localId * 4; j < mainLoopEnd; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wqkv.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte quant1 = wqkv.get(quantsOffset);
            byte quant2 = wqkv.get(quantsOffset + 1);
            byte quant3 = wqkv.get(quantsOffset + 2);
            byte quant4 = wqkv.get(quantsOffset + 3);

            float norm1 = rmsWeights.get(j) * rmsScaleFactor * x.get(j);
            float norm2 = rmsWeights.get(j + 1) * rmsScaleFactor * x.get(j + 1);
            float norm3 = rmsWeights.get(j + 2) * rmsScaleFactor * x.get(j + 2);
            float norm4 = rmsWeights.get(j + 3) * rmsScaleFactor * x.get(j + 3);

            partialSum1 += ((float) quant1 * qScaleFloat) * norm1;
            partialSum2 += ((float) quant2 * qScaleFloat) * norm2;
            partialSum3 += ((float) quant3 * qScaleFloat) * norm3;
            partialSum4 += ((float) quant4 * qScaleFloat) * norm4;
        }

        // Tail loop - handle remaining 0-3 elements (one element per thread)
        int tailIdx = mainLoopEnd + localId;
        if (tailIdx < dim) {
            int blockIdx = tailIdx / blockSize;
            int withinBlockIdx = tailIdx % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wqkv.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte quant = wqkv.get(quantsOffset);

            float normalized = rmsWeights.get(tailIdx) * rmsScaleFactor * x.get(tailIdx);
            partialSum1 += ((float) quant * qScaleFloat) * normalized;
        }

        localSums[localId] = partialSum1 + partialSum2 + partialSum3 + partialSum4;
        context.localBarrier();

        // Parallel reduction
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        // Thread 0 writes to appropriate output
        if (localId == 0) {
            float result = localSums[0];

            if (rowId < dim) {
                q.set(rowId, result);
            } else if (rowId < dim + kvDim) {
                k.set(rowId - dim, result);
            } else {
                v.set(rowId - dim - kvDim, result);
            }
        }
    }

    public static void matrixVectorGenericQ8Byte(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray q,
            int dim1,
            int dim0,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= dim0) {
            return;
        }

        float sum = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, q, dim1);

        // Thread 0 writes the result
        if (localId == 0) {
            output.set(rowId, sum);
        }
    }

    public static float matrixVectorRowMajorOptimizedQ8_0Byte(
            KernelContext context, int localSize, FloatArray x, ByteArray q, int n) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow; // Starting block index for this row

        // 4-way unrolling
        float partialSum1 = 0.0f;
        float partialSum2 = 0.0f;
        float partialSum3 = 0.0f;
        float partialSum4 = 0.0f;

        // Main loop - process 4 elements at a time
        for (int j = localId * 4; j < n - 3; j += localSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            // Calculate byte offset for this Q8_0 block
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            // Load scale (first 2 bytes of block as HalfFloat)
            HalfFloat scale = q.getHalfFloat(blockByteOffset);
            float scaleFloat = scale.getFloat32();

            // Load 4 consecutive quantized values
            int quantsOffset = blockByteOffset + 2 + withinBlockIdx; // Skip 2-byte scale
            byte quant1 = q.get(quantsOffset);
            byte quant2 = q.get(quantsOffset + 1);
            byte quant3 = q.get(quantsOffset + 2);
            byte quant4 = q.get(quantsOffset + 3);

            // Dequantize and multiply
            partialSum1 += ((float) quant1 * scaleFloat) * x.get(j);
            partialSum2 += ((float) quant2 * scaleFloat) * x.get(j + 1);
            partialSum3 += ((float) quant3 * scaleFloat) * x.get(j + 2);
            partialSum4 += ((float) quant4 * scaleFloat) * x.get(j + 3);
        }

        float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

        // Handle remaining elements
        for (int j = ((n / 4) * 4) + localId; j < n; j += localSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            // Calculate byte offset for this Q8_0 block
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            // Load scale
            HalfFloat scale = q.getHalfFloat(blockByteOffset);
            float scaleFloat = scale.getFloat32();

            // Load quantized value
            byte quant = q.get(blockByteOffset + 2 + withinBlockIdx);

            partialSum += ((float) quant * scaleFloat) * x.get(j);
        }

        localSums[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        return localSums[0];
    }

    public static void matrixVectorGenericWithResidual(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            Int8Array w_quants,
            HalfFloatArray w_scales,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimizedQ8_0(context, localSize, x, w_quants, w_scales, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    public static void matrixVectorGenericWithResidualQ8_0Byte(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimizedQ8_0Byte(context, localSize, x, w, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    /**
     * Warp-shuffle variant of {@link #matrixVectorGenericWithResidualQ8_0Byte}: one 32-lane warp
     * per output row, reduced via {@code simdShuffleDown} instead of a shared-memory tree. Q8_0
     * byte layout (34-byte blocks: 2-byte half scale + 32 int8 quants).
     */
    public static void matrixVectorGenericWithResidualQ8_0ByteSimd32(
            KernelContext context, FloatArray x, FloatArray hb, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2-byte scale + 32 int8 quants
        int blocksPerRow = (n + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j - blockIdx * blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            float scaleFloat = w.getHalfFloat(blockByteOffset).getFloat32();
            byte quant = w.get(blockByteOffset + 2 + withinBlockIdx);
            partialSum += ((float) quant * scaleFloat) * x.get(j);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);

        if (localId == 0) {
            hb.set(rowId, hb.get(rowId) + partialSum);
        }
    }

    public static void fusedFeedForwardWithSiLUAndGLUActivation(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            Int8Array w1_quants,
            HalfFloatArray w1_scales,
            Int8Array w3_quants,
            HalfFloatArray w3_scales,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 =
                matrixVectorRowMajorOptimizedQ8_0(
                        context, localWorkGroupSize, x, w1_quants, w1_scales, n);
        float sum3 =
                matrixVectorRowMajorOptimizedQ8_0(
                        context, localWorkGroupSize, x, w3_quants, w3_scales, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1); // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    public static void fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, w1, n);
        float sum3 = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, w3, n);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1); // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    /**
     * Fused Q/K/V matrix-vector multiplication for Q8_0 quantized weights. Reduces kernel launch
     * overhead and improves input vector cache utilization.
     *
     * <p>Workgroup assignment: - rowId [0, dim): Q projection - rowId [dim, dim+kvDim): K
     * projection - rowId [dim+kvDim, dim+2*kvDim): V projection
     */
    public static void fusedQKVMatmulQ8(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < dim) {
            // ========== Q projection ==========
            int rowBlockOffset = rowId * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < dim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wq.get(quantsOffset);
                byte quant2 = wq.get(quantsOffset + 1);
                byte quant3 = wq.get(quantsOffset + 2);
                byte quant4 = wq.get(quantsOffset + 3);

                partialSum1 += ((float) quant1 * scaleFloat) * x.get(j);
                partialSum2 += ((float) quant2 * scaleFloat) * x.get(j + 1);
                partialSum3 += ((float) quant3 * scaleFloat) * x.get(j + 2);
                partialSum4 += ((float) quant4 * scaleFloat) * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((dim / 4) * 4) + localId; j < dim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                byte quant = wq.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * scaleFloat) * x.get(j);
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
                q.set(rowId, localSums[0]);
            }

        } else if (rowId < dim + kvDim) {
            // ========== K projection ==========
            int kRow = rowId - dim;
            int rowBlockOffset = kRow * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < dim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wk.get(quantsOffset);
                byte quant2 = wk.get(quantsOffset + 1);
                byte quant3 = wk.get(quantsOffset + 2);
                byte quant4 = wk.get(quantsOffset + 3);

                partialSum1 += ((float) quant1 * scaleFloat) * x.get(j);
                partialSum2 += ((float) quant2 * scaleFloat) * x.get(j + 1);
                partialSum3 += ((float) quant3 * scaleFloat) * x.get(j + 2);
                partialSum4 += ((float) quant4 * scaleFloat) * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((dim / 4) * 4) + localId; j < dim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                byte quant = wk.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * scaleFloat) * x.get(j);
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

        } else if (rowId < dim + 2 * kvDim) {
            // ========== V projection ==========
            int vRow = rowId - dim - kvDim;
            int rowBlockOffset = vRow * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < dim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wv.get(quantsOffset);
                byte quant2 = wv.get(quantsOffset + 1);
                byte quant3 = wv.get(quantsOffset + 2);
                byte quant4 = wv.get(quantsOffset + 3);

                partialSum1 += ((float) quant1 * scaleFloat) * x.get(j);
                partialSum2 += ((float) quant2 * scaleFloat) * x.get(j + 1);
                partialSum3 += ((float) quant3 * scaleFloat) * x.get(j + 2);
                partialSum4 += ((float) quant4 * scaleFloat) * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((dim / 4) * 4) + localId; j < dim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat scale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = scale.getFloat32();

                byte quant = wv.get(blockByteOffset + 2 + withinBlockIdx);
                partialSum += ((float) quant * scaleFloat) * x.get(j);
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
     * Fused QKV matmul for models where Q output dim != input dim (e.g., Devstral 2). Separates
     * inputDim (embedding size) from qDim (num_heads * head_dim).
     */
    public static void fusedQKVMatmulQ8NonSquare(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int inputDim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (inputDim + blockSize - 1) / blockSize;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection ==========
            int rowBlockOffset = rowId * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                float scaleFloat = wq.getHalfFloat(blockByteOffset).getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                partialSum1 += ((float) wq.get(quantsOffset)) * scaleFloat * x.get(j);
                partialSum2 += ((float) wq.get(quantsOffset + 1)) * scaleFloat * x.get(j + 1);
                partialSum3 += ((float) wq.get(quantsOffset + 2)) * scaleFloat * x.get(j + 2);
                partialSum4 += ((float) wq.get(quantsOffset + 3)) * scaleFloat * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float scaleFloat = wq.getHalfFloat(blockByteOffset).getFloat32();
                partialSum +=
                        ((float) wq.get(blockByteOffset + 2 + withinBlockIdx))
                                * scaleFloat
                                * x.get(j);
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
                q.set(rowId, localSums[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection ==========
            int kRow = rowId - qDim;
            int rowBlockOffset = kRow * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float scaleFloat = wk.getHalfFloat(blockByteOffset).getFloat32();
                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                partialSum1 += ((float) wk.get(quantsOffset)) * scaleFloat * x.get(j);
                partialSum2 += ((float) wk.get(quantsOffset + 1)) * scaleFloat * x.get(j + 1);
                partialSum3 += ((float) wk.get(quantsOffset + 2)) * scaleFloat * x.get(j + 2);
                partialSum4 += ((float) wk.get(quantsOffset + 3)) * scaleFloat * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float scaleFloat = wk.getHalfFloat(blockByteOffset).getFloat32();
                partialSum +=
                        ((float) wk.get(blockByteOffset + 2 + withinBlockIdx))
                                * scaleFloat
                                * x.get(j);
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
            // ========== V projection ==========
            int vRow = rowId - qDim - kvDim;
            int rowBlockOffset = vRow * blocksPerRow;

            float partialSum1 = 0.0f;
            float partialSum2 = 0.0f;
            float partialSum3 = 0.0f;
            float partialSum4 = 0.0f;

            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float scaleFloat = wv.getHalfFloat(blockByteOffset).getFloat32();
                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                partialSum1 += ((float) wv.get(quantsOffset)) * scaleFloat * x.get(j);
                partialSum2 += ((float) wv.get(quantsOffset + 1)) * scaleFloat * x.get(j + 1);
                partialSum3 += ((float) wv.get(quantsOffset + 2)) * scaleFloat * x.get(j + 2);
                partialSum4 += ((float) wv.get(quantsOffset + 3)) * scaleFloat * x.get(j + 3);
            }

            float partialSum = partialSum1 + partialSum2 + partialSum3 + partialSum4;

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;
                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
                float scaleFloat = wv.getHalfFloat(blockByteOffset).getFloat32();
                partialSum +=
                        ((float) wv.get(blockByteOffset + 2 + withinBlockIdx))
                                * scaleFloat
                                * x.get(j);
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
     * Fully fused RMS normalization + FFN W1/W3 matmul with SiLU/GLU for Q8_0 weights. Each
     * workgroup redundantly computes RMS scale to avoid cross-workgroup sync.
     */
    public static void fullyFusedRmsNormFFNGateUpQ8(
            KernelContext context,
            FloatArray x, // raw input (FP32)
            FloatArray hb, // output
            FloatArray rmsWeights, // RMS norm weights
            ByteArray w1, // Q8_0 quantized
            ByteArray w3, // Q8_0 quantized
            int dim, // input dimension
            int hiddenDim, // output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        // ========== RMS Norm: Compute scale (each workgroup does this redundantly) ==========
        float sumSquares = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float val = x.get(j);
            sumSquares += val * val;
        }

        localSum[localId] = sumSquares;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        float scale = 1.0f / TornadoMath.sqrt(localSum[0] / dim + 1e-5f);

        // ========== W1 matmul with inline RMS normalization ==========
        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum1_a = 0.0f;
        float partialSum1_b = 0.0f;
        float partialSum1_c = 0.0f;
        float partialSum1_d = 0.0f;

        for (int j = localId * 4; j < dim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat w1Scale = w1.getHalfFloat(blockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte q1 = w1.get(quantsOffset);
            byte q2 = w1.get(quantsOffset + 1);
            byte q3 = w1.get(quantsOffset + 2);
            byte q4 = w1.get(quantsOffset + 3);

            float norm0 = rmsWeights.get(j) * scale * x.get(j);
            float norm1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            float norm2 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
            float norm3 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

            partialSum1_a += ((float) q1 * w1ScaleFloat) * norm0;
            partialSum1_b += ((float) q2 * w1ScaleFloat) * norm1;
            partialSum1_c += ((float) q3 * w1ScaleFloat) * norm2;
            partialSum1_d += ((float) q4 * w1ScaleFloat) * norm3;
        }

        float partialSum1 = partialSum1_a + partialSum1_b + partialSum1_c + partialSum1_d;

        for (int j = ((dim / 4) * 4) + localId; j < dim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat w1Scale = w1.getHalfFloat(blockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            byte quant = w1.get(blockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            partialSum1 += ((float) quant * w1ScaleFloat) * normalized;
        }

        localSum[localId] = partialSum1;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];

        // ========== W3 matmul with inline RMS normalization ==========
        float partialSum3_a = 0.0f;
        float partialSum3_b = 0.0f;
        float partialSum3_c = 0.0f;
        float partialSum3_d = 0.0f;

        for (int j = localId * 4; j < dim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat w3Scale = w3.getHalfFloat(blockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte q1 = w3.get(quantsOffset);
            byte q2 = w3.get(quantsOffset + 1);
            byte q3 = w3.get(quantsOffset + 2);
            byte q4 = w3.get(quantsOffset + 3);

            float norm0 = rmsWeights.get(j) * scale * x.get(j);
            float norm1 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            float norm2 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
            float norm3 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

            partialSum3_a += ((float) q1 * w3ScaleFloat) * norm0;
            partialSum3_b += ((float) q2 * w3ScaleFloat) * norm1;
            partialSum3_c += ((float) q3 * w3ScaleFloat) * norm2;
            partialSum3_d += ((float) q4 * w3ScaleFloat) * norm3;
        }

        float partialSum3 = partialSum3_a + partialSum3_b + partialSum3_c + partialSum3_d;

        for (int j = ((dim / 4) * 4) + localId; j < dim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat w3Scale = w3.getHalfFloat(blockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            byte quant = w3.get(blockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            partialSum3 += ((float) quant * w3ScaleFloat) * normalized;
        }

        localSum[localId] = partialSum3;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        // ========== SiLU + GLU ==========
        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            hb.set(rowId, silu * result3);
        }
    }

    /**
     * Fused RMSNorm apply + Gate/Up Q8 projection + SiLU + GLU in one kernel.
     *
     * <p>Each workgroup computes one output element by:
     *
     * <ul>
     *   <li>gate[i] = dot(wUp[i], RMSNorm(x))
     *   <li>up[i] = dot(wUp[hiddenDim + i], RMSNorm(x))
     *   <li>output[i] = SiLU(gate[i]) × up[i]
     * </ul>
     *
     * @param context Kernel execution context
     * @param x Input hidden state (FP32) [dim]
     * @param output Output buffer (FP32) [hiddenDim] - ready for wDown
     * @param rmsWeights RMS normalization weights (FP32) [dim]
     * @param rmsScale Precomputed RMS scale factor [1]
     * @param wUp Combined gate+up weight matrix (Q8) [2×hiddenDim × dim]
     * @param dim Input dimension
     * @param hiddenDim Hidden dimension (output size)
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormFFNGateUpSiLUQ8(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            FloatArray rmsWeights,
            FloatArray rmsScale,
            ByteArray wUp, // Q8 quantized [2×hiddenDim × dim]
            int dim,
            int hiddenDim,
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);

        // Q8_0 format constants
        int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        int blocksPerRow = (dim + blockSize - 1) / blockSize;

        // ═══════════════════════════════════════════════════════════════════════
        //                         GATE PROJECTION (row i)
        // ═══════════════════════════════════════════════════════════════════════
        int gateRowBlockOffset = rowId * blocksPerRow;

        float gateSum1 = 0.0f, gateSum2 = 0.0f, gateSum3 = 0.0f, gateSum4 = 0.0f;

        int mainLoopEnd = (dim / 4) * 4;
        for (int j = localId * 4; j < mainLoopEnd; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (gateRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wUp.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte q1 = wUp.get(quantsOffset);
            byte q2 = wUp.get(quantsOffset + 1);
            byte q3 = wUp.get(quantsOffset + 2);
            byte q4 = wUp.get(quantsOffset + 3);

            // Inline RMS normalization
            float norm1 = rmsWeights.get(j) * scale * x.get(j);
            float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
            float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

            gateSum1 += ((float) q1 * qScaleFloat) * norm1;
            gateSum2 += ((float) q2 * qScaleFloat) * norm2;
            gateSum3 += ((float) q3 * qScaleFloat) * norm3;
            gateSum4 += ((float) q4 * qScaleFloat) * norm4;
        }

        // Tail for gate
        int tailIdx = mainLoopEnd + localId;
        if (tailIdx < dim) {
            int blockIdx = tailIdx / blockSize;
            int withinBlockIdx = tailIdx % blockSize;
            int blockByteOffset = (gateRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wUp.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();
            byte quant = wUp.get(blockByteOffset + 2 + withinBlockIdx);

            float normalized = rmsWeights.get(tailIdx) * scale * x.get(tailIdx);
            gateSum1 += ((float) quant * qScaleFloat) * normalized;
        }

        localSum[localId] = gateSum1 + gateSum2 + gateSum3 + gateSum4;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        float gateResult = localSum[0];

        // ═══════════════════════════════════════════════════════════════════════
        //                      UP PROJECTION (row hiddenDim + i)
        // ═══════════════════════════════════════════════════════════════════════
        int upRowBlockOffset = (hiddenDim + rowId) * blocksPerRow;

        float upSum1 = 0.0f, upSum2 = 0.0f, upSum3 = 0.0f, upSum4 = 0.0f;

        for (int j = localId * 4; j < mainLoopEnd; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;
            int blockByteOffset = (upRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wUp.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();

            int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
            byte q1 = wUp.get(quantsOffset);
            byte q2 = wUp.get(quantsOffset + 1);
            byte q3 = wUp.get(quantsOffset + 2);
            byte q4 = wUp.get(quantsOffset + 3);

            // Inline RMS normalization (same values as gate)
            float norm1 = rmsWeights.get(j) * scale * x.get(j);
            float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
            float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
            float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

            upSum1 += ((float) q1 * qScaleFloat) * norm1;
            upSum2 += ((float) q2 * qScaleFloat) * norm2;
            upSum3 += ((float) q3 * qScaleFloat) * norm3;
            upSum4 += ((float) q4 * qScaleFloat) * norm4;
        }

        // Tail for up
        if (tailIdx < dim) {
            int blockIdx = tailIdx / blockSize;
            int withinBlockIdx = tailIdx % blockSize;
            int blockByteOffset = (upRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

            HalfFloat qScale = wUp.getHalfFloat(blockByteOffset);
            float qScaleFloat = qScale.getFloat32();
            byte quant = wUp.get(blockByteOffset + 2 + withinBlockIdx);

            float normalized = rmsWeights.get(tailIdx) * scale * x.get(tailIdx);
            upSum1 += ((float) quant * qScaleFloat) * normalized;
        }

        localSum[localId] = upSum1 + upSum2 + upSum3 + upSum4;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        float upResult = localSum[0];

        // ═══════════════════════════════════════════════════════════════════════
        //                         SiLU(gate) × up
        // ═══════════════════════════════════════════════════════════════════════
        if (localId == 0) {
            float silu = gateResult / (1.0f + TornadoMath.exp(-gateResult));
            output.set(rowId, silu * upResult);
        }
    }
}
