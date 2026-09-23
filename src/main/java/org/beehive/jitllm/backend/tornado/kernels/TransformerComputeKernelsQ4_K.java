package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q4_K} weights <b>in the file's own representation</b>.
 *
 * <p>They are deliberately shaped like their Q8_0 siblings in {@link
 * TransformerComputeKernelsLayered} — same signatures, same workgroup-per-row structure, same
 * reduction — so the layer classes differ only in which method reference they name. What changes is
 * the decode in the inner loop.
 *
 * <h2>The super-block</h2>
 *
 * <p>256 weights in 144 bytes: {@code d} (fp16) at 0, {@code dmin} (fp16) at 2, twelve bytes of
 * packed 6-bit scale/min pairs at 4, and 128 bytes of 4-bit weights at 16. A weight is {@code d *
 * scale(sub) * q - dmin * min(sub)}, where the 256 weights are covered by eight sub-blocks of 32.
 * Within a 64-weight pair the first 32 take the low nibble and the next 32 the high nibble of the
 * same byte.
 *
 * <p>The unpacking below is the same arithmetic as the host's {@code Q4_KFloatTensor}, which is the
 * reference this was written against — the two must agree, and {@code Q4_KDeviceDecodeAccelTest}
 * asserts exactly that on real weights rather than trusting the restatement.
 */
public final class TransformerComputeKernelsQ4_K {

    /** Weights per super-block. */
    private static final int QK_K = 256;

    /** Bytes per super-block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs). */
    private static final int BLOCK_BYTES = 144;

    /** Byte offset of the packed 6-bit scale/min pairs within a super-block. */
    private static final int SCALES_OFFSET = 4;

    /** Byte offset of the 4-bit weights within a super-block. */
    private static final int QS_OFFSET = 16;

    private TransformerComputeKernelsQ4_K() {}

    /**
     * One weight, decoded from its super-block.
     *
     * <p>Kept as a static helper so every kernel below decodes identically; TornadoVM inlines it.
     * Package-private rather than private so {@code Q4_KDecodeTest} can hold it against the host's
     * {@code Q4_KFloatTensor} directly, on the same bytes.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's super-block
     * @param withinBlock the element's index inside the super-block, 0.255
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock - pairIndex * 64; // 0..63
        int highNibble = posInPair / 32; // 0 for the first 32, 1 for the next
        int subBlock = pairIndex * 2 + highNibble;

        int qByte =
                w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + (posInPair - highNibble * 32))
                        & 0xFF;
        int q = (highNibble == 0) ? (qByte & 0xF) : ((qByte >> 4) & 0xF);

        int scalesBase = blockByteOffset + SCALES_OFFSET;
        int sc;
        int m;
        if (subBlock < 4) {
            sc = w.get(scalesBase + subBlock) & 63;
            m = w.get(scalesBase + subBlock + 4) & 63;
        } else {
            int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
            int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
            sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
            m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        }
        return d * sc * q - dmin * m;
    }

    /**
     * One row's dot product against {@code x}, reduced across a 32-lane subgroup.
     *
     * <p>The Q4_K counterpart of {@code matrixVectorRowMajorOptimizedQ8_0Byte}'s reduction, in the
     * shuffle form — used where {@code DeviceCapability.SUBGROUP_SHUFFLE_32} holds.
     */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /**
     * One row's dot product against {@code x}, reduced through shared memory.
     *
     * <p>The portable form, for devices without a verified 32-lane shuffle — the same choice {@code
     * matrixVectorRowMajorOptimizedQ8_0Byte} makes.
     */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        localSums[localId] = partialSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        return localSums[0];
    }

    /** {@code output[row] = w[row]·x}. Q4_K counterpart of {@code matrixVectorGenericQ8Byte}. */
    public static void matrixVectorGenericQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_K}. */
    public static void matrixVectorGenericQ4_KSimd32(
            KernelContext context, FloatArray x, FloatArray output, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotSimd32(context, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /**
     * {@code hb[row] += w[row]·x}. Q4_K counterpart of {@code
     * matrixVectorGenericWithResidualQ8_0Byte}.
     */
    public static void matrixVectorGenericWithResidualQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, x, w, n, rowId);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + sum);
        }
    }

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ4_K}. */
    public static void matrixVectorGenericWithResidualQ4_KSimd32(
            KernelContext context, FloatArray x, FloatArray hb, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotSimd32(context, x, w, n, rowId);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + sum);
        }
    }

    /**
     * Fused Q/K/V projection for a model whose head dimension is independent of {@code dim /
     * numberOfHeads} — Devstral's shape. Q4_K counterpart of {@code fusedQKVMatmulQ8NonSquare}: one
     * workgroup per output row, with the row's identity selecting which of the three projections it
     * belongs to.
     */
    public static void fusedQKVMatmulQ4_KNonSquare(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int totalRows = qDim + 2 * kvDim;
        if (rowId >= totalRows) {
            return;
        }

        if (rowId < qDim) {
            float sum = rowDotShared(context, localWorkGroupSize, x, wq, dim, rowId);
            if (context.localIdx == 0) {
                q.set(rowId, sum);
            }
        } else if (rowId < qDim + kvDim) {
            int row = rowId - qDim;
            float sum = rowDotShared(context, localWorkGroupSize, x, wk, dim, row);
            if (context.localIdx == 0) {
                k.set(row, sum);
            }
        } else {
            int row = rowId - qDim - kvDim;
            float sum = rowDotShared(context, localWorkGroupSize, x, wv, dim, row);
            if (context.localIdx == 0) {
                v.set(row, sum);
            }
        }
    }

    /**
     * Fused FFN gate/up projection with SwiGLU. Q4_K counterpart of the gate/up half of {@code
     * fullyFusedRmsNormFFNGateUpQ8}, taking the already-normalized activation rather than folding
     * the RMS norm in: the norm is dtype-independent and its existing task is reused.
     */
    public static void fusedFFNGateUpSiLUQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }

        // Both projections in one pass over one local array: two calls to a helper that allocates
        // local memory and barriers would allocate twice and reduce twice, and the two reductions
        // would have to interleave their barriers correctly to be safe. Gate occupies the first
        // half, up the second, and one tree reduces both.
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * 2);
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            float activation = x.get(j);
            gate += decode(w1, blockByteOffset, withinBlock) * activation;
            up += decode(w3, blockByteOffset, withinBlock) * activation;
        }
        localSums[localId] = gate;
        localSums[localWorkGroupSize + localId] = up;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
                localSums[localWorkGroupSize + localId] +=
                        localSums[localWorkGroupSize + localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0) {
            float gateSum = localSums[0];
            float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
            hb.set(rowId, silu * localSums[localWorkGroupSize]);
        }
    }
}
