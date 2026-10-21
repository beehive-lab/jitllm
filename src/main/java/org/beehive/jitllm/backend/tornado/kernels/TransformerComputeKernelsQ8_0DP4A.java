package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

// @formatter:off
/**
 * Decode projections over Q8_0 weights as packed integer dot products ({@code dp4a}).
 *
 * <p>The activation is the one {@link TransformerComputeKernelsQ4_0#quantizeActivationQ8Blocks}
 * prepares for the Q4_0 projections: 32-element blocks of signed bytes, four to an int, with an
 * FP32 scale per block. A Q8_0 weight block has the same shape — an FP16 scale and 32 signed bytes
 * — so a block's dot product is eight {@code dp4a} instructions on the raw bytes, times the two
 * scales. Unlike Q4_0 nothing is recentred, so the activation's block sums are not read.
 *
 * <p>One workgroup per output row, as in the Q4_0 kernels: each lane walks blocks, each warp
 * reduces with shuffles, and lane 0 combines the warps. {@code localWorkGroupSize} must be a
 * multiple of 32. The weight bytes are read two at a time, which keeps every read aligned: a block
 * starts at an even offset and its quants two bytes later.
 */
// @formatter:on
public final class TransformerComputeKernelsQ8_0DP4A {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;
    private static final int QS_OFFSET = 2;

    private TransformerComputeKernelsQ8_0DP4A() {}

    /** The integer dot product of one weight block with the activation block it multiplies. */
    private static int blockDot(ByteArray w, int blockByteOffset, IntArray xQuants, int quantBase) {
        int dot = 0;
        for (int g = 0; g < QK / 4; g++) {
            int offset = blockByteOffset + QS_OFFSET + g * 4;
            int packed =
                    (w.getHalfFloat(offset).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(offset + 2).getHalfFloatValue() & 0xFFFF) << 16);
            dot = QuantizationUtils.dp4a_packed(packed, xQuants.get(quantBase + g), dot);
        }
        return dot;
    }

    /** This lane's share of row {@code rowId}'s dot product. */
    private static float rowPartial(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        int blocksPerRow = n / QK;
        int rowBlockOffset = rowId * blocksPerRow;
        float partial = 0.0f;
        for (int block = context.localIdx; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float weightScale = w.getHalfFloat(blockByteOffset).getFloat32();
            partial +=
                    weightScale
                            * xScales.get(block)
                            * blockDot(w, blockByteOffset, xQuants, block * (QK / 4));
        }
        return partial;
    }

    /** {@code output[row] = w[row]·x}, the activation quantized. */
    public static void matrixVectorGenericQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            float total =
                    rowDot(context, xQuants, xScales, w, n, rowId, localWorkGroupSize, warpSums);
            if (context.localIdx == 0) {
                output.set(rowId, total);
            }
        }
    }

    /** Row {@code rowId} of {@code w}·x reduced across the group; the total is valid in lane 0. */
    private static float rowDot(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize,
            float[] warpSums) {
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        float partial = rowPartial(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
        partial += context.simdShuffleDown(partial, 16);
        partial += context.simdShuffleDown(partial, 8);
        partial += context.simdShuffleDown(partial, 4);
        partial += context.simdShuffleDown(partial, 2);
        partial += context.simdShuffleDown(partial, 1);
        if ((localId & 31) == 0) {
            warpSums[localId >> 5] = partial;
        }
        context.localBarrier();
        float total = 0.0f;
        for (int warp = 0; warp < warpCount; warp++) {
            total += warpSums[warp];
        }
        return total;
    }

    /**
     * The Q, K and V projections of one quantized activation in one launch: groups {@code [0,
     * qDim)} produce Q from {@code wq}, the next {@code kvDim} K from {@code wk}, the last {@code
     * kvDim} V from {@code wv}. Each branch is a whole row computation, so no weight array is
     * selected at run time.
     *
     * <p>Worker: {@code (qDim + 2 kvDim) * localWorkGroupSize} lanes, local {@code
     * localWorkGroupSize}, a multiple of 32.
     */
    public static void fusedQKVMatmulQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int n,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        // Each condition is uniform across the group, so the barrier inside is reached by all.
        if (rowId < qDim) {
            float total =
                    rowDot(context, xQuants, xScales, wq, n, rowId, localWorkGroupSize, warpSums);
            if (localId == 0) {
                q.set(rowId, total);
            }
        } else if (rowId < qDim + kvDim) {
            int row = rowId - qDim;
            float total =
                    rowDot(context, xQuants, xScales, wk, n, row, localWorkGroupSize, warpSums);
            if (localId == 0) {
                k.set(row, total);
            }
        } else if (rowId < qDim + 2 * kvDim) {
            int row = rowId - qDim - kvDim;
            float total =
                    rowDot(context, xQuants, xScales, wv, n, row, localWorkGroupSize, warpSums);
            if (localId == 0) {
                v.set(row, total);
            }
        }
    }

    /**
     * The Q, K and V projections of one quantized activation from a single combined weight, rows
     * {@code [0, qDim)} Q, the next {@code kvDim} K and the last {@code kvDim} V (Phi-3's {@code
     * wqkv}).
     *
     * <p>Worker: {@code (qDim + 2 kvDim) * localWorkGroupSize} lanes, local {@code
     * localWorkGroupSize}, a multiple of 32.
     */
    public static void combinedQKVMatmulQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wqkv,
            int n,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < qDim + 2 * kvDim) {
            float total =
                    rowDot(context, xQuants, xScales, wqkv, n, rowId, localWorkGroupSize, warpSums);
            if (context.localIdx == 0) {
                if (rowId < qDim) {
                    q.set(rowId, total);
                } else if (rowId < qDim + kvDim) {
                    k.set(rowId - qDim, total);
                } else {
                    v.set(rowId - qDim - kvDim, total);
                }
            }
        }
    }

    /**
     * {@code hb[row] = silu(wUp[row]·x) * (wUp[d + row]·x)} from a single combined gate/up weight,
     * gate rows first (Phi-3's {@code w_up}). The two reductions use separate scratch, so neither
     * overwrites the other's partials while they are read.
     */
    public static void combinedFFNGateUpSiLUQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray wUp,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] gateSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        float[] upSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            float gate =
                    rowDot(context, xQuants, xScales, wUp, n, rowId, localWorkGroupSize, gateSums);
            float up =
                    rowDot(
                            context,
                            xQuants,
                            xScales,
                            wUp,
                            n,
                            d + rowId,
                            localWorkGroupSize,
                            upSums);
            if (context.localIdx == 0) {
                float silu = gate / (1.0f + TornadoMath.exp(-gate));
                hb.set(rowId, silu * up);
            }
        }
    }

    /** {@code hb[row] += w[row]·x}, the activation quantized. */
    public static void matrixVectorGenericWithResidualQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            float total =
                    rowDot(context, xQuants, xScales, w, n, rowId, localWorkGroupSize, warpSums);
            if (context.localIdx == 0) {
                hb.set(rowId, hb.get(rowId) + total);
            }
        }
    }

    /**
     * {@code hb[row] += residualScale * (w[row]·x)}, the activation quantized: the residual
     * connection of a family that scales the branch before adding it (Granite's residual
     * multiplier).
     */
    public static void matrixVectorGenericWithScaledResidualQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize,
            float residualScale) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            float total =
                    rowDot(context, xQuants, xScales, w, n, rowId, localWorkGroupSize, warpSums);
            if (context.localIdx == 0) {
                hb.set(rowId, hb.get(rowId) + residualScale * total);
            }
        }
    }

    /**
     * {@code hb[row] = silu(w1[row]·x) * (w3[row]·x)}, both projections against the same quantized
     * activation. SiLU is applied to the gate reduced across every warp, never per warp.
     */
    public static void fusedFFNGateUpSiLUQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        // Gate in the first warpCount entries, up in the second.
        float[] warpSums = context.allocateFloatLocalArray(warpCount * 2);

        float gate = rowPartial(context, xQuants, xScales, w1, n, rowId, localWorkGroupSize);
        float up = rowPartial(context, xQuants, xScales, w3, n, rowId, localWorkGroupSize);
        gate += context.simdShuffleDown(gate, 16);
        gate += context.simdShuffleDown(gate, 8);
        gate += context.simdShuffleDown(gate, 4);
        gate += context.simdShuffleDown(gate, 2);
        gate += context.simdShuffleDown(gate, 1);
        up += context.simdShuffleDown(up, 16);
        up += context.simdShuffleDown(up, 8);
        up += context.simdShuffleDown(up, 4);
        up += context.simdShuffleDown(up, 2);
        up += context.simdShuffleDown(up, 1);
        if ((localId & 31) == 0) {
            warpSums[localId >> 5] = gate;
            warpSums[warpCount + (localId >> 5)] = up;
        }
        context.localBarrier();

        if (localId == 0) {
            float gateSum = 0.0f;
            float upSum = 0.0f;
            for (int warp = 0; warp < warpCount; warp++) {
                gateSum += warpSums[warp];
                upSum += warpSums[warpCount + warp];
            }
            float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
            hb.set(rowId, silu * upSum);
        }
    }

    /**
     * {@code hb[row] = gelu(w1[row]·x) * (w3[row]·x)}, both projections against the same quantized
     * activation. GELU is applied to the gate reduced across every warp, never per warp.
     */
    public static void fusedFFNGateUpGeGLUQ8_0DP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        // Gate in the first warpCount entries, up in the second.
        float[] warpSums = context.allocateFloatLocalArray(warpCount * 2);

        float gate = rowPartial(context, xQuants, xScales, w1, n, rowId, localWorkGroupSize);
        float up = rowPartial(context, xQuants, xScales, w3, n, rowId, localWorkGroupSize);
        gate += context.simdShuffleDown(gate, 16);
        gate += context.simdShuffleDown(gate, 8);
        gate += context.simdShuffleDown(gate, 4);
        gate += context.simdShuffleDown(gate, 2);
        gate += context.simdShuffleDown(gate, 1);
        up += context.simdShuffleDown(up, 16);
        up += context.simdShuffleDown(up, 8);
        up += context.simdShuffleDown(up, 4);
        up += context.simdShuffleDown(up, 2);
        up += context.simdShuffleDown(up, 1);
        if ((localId & 31) == 0) {
            warpSums[localId >> 5] = gate;
            warpSums[warpCount + (localId >> 5)] = up;
        }
        context.localBarrier();

        if (localId == 0) {
            float gateSum = 0.0f;
            float upSum = 0.0f;
            for (int warp = 0; warp < warpCount; warp++) {
                gateSum += warpSums[warp];
                upSum += warpSums[warpCount + warp];
            }
            hb.set(rowId, TransformerComputeKernelsLayered.geluActivation(gateSum) * upSum);
        }
    }
}
