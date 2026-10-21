package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

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
     * Decodes a Q4_K weight into FP16 at {@code destOffset} of {@code out}, for a GEMM that reads
     * FP16 operands. One thread per element of the {@code n}; the row width a multiple of 256, so
     * element {@code i} lies in super-block {@code i / 256}.
     */
    public static void dequantizeQ4_KToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset, int n) {
        int gid = context.globalIdx;
        if (gid < n) {
            // decode(...) copied in with every read unconditional: called through the helper, the
            // CUDA backend cannot lower the FP16 store's HalfFloat allocation (NewInstance).
            int block = gid / QK_K;
            int blockByteOffset = block * BLOCK_BYTES;
            int withinBlock = gid - block * QK_K;
            float d = w.getHalfFloat(blockByteOffset).getFloat32();
            float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();
            int pairIndex = withinBlock / 64;
            int posInPair = withinBlock - pairIndex * 64;
            int highNibble = posInPair / 32;
            int subBlock = pairIndex * 2 + highNibble;
            int qByte =
                    w.get(
                                    blockByteOffset
                                            + QS_OFFSET
                                            + pairIndex * 32
                                            + (posInPair - highNibble * 32))
                            & 0xFF;
            int q = (qByte >> (highNibble * 4)) & 0xF;
            // Sub-blocks 0-3 read bytes [j] and [j+4]; 4-7 read [j+4], [j-4] and [j]. Reading all
            // three stays inside the block: [j-4] for j < 4 lands on d/dmin and is discarded.
            int scalesBase = blockByteOffset + SCALES_OFFSET;
            int a = w.get(scalesBase + subBlock) & 0xFF;
            int b = w.get(scalesBase + subBlock + 4) & 0xFF;
            int c = w.get(scalesBase + subBlock - 4) & 0xFF;
            int low = subBlock < 4 ? 1 : 0;
            int sc = low * (a & 63) + (1 - low) * ((b & 0xF) | ((c >> 6) << 4));
            int m = low * (b & 63) + (1 - low) * (((b >> 4) & 0xF) | ((a >> 6) << 4));
            out.set(destOffset + gid, new HalfFloat(d * sc * q - dmin * m));
        }
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

    // @formatter:off
    /**
     * {@code output[row] = w[row]·x} reading Q4_K super-blocks where they lie, the dot products
     * done in packed integers against activations quantized by {@code
     * TransformerComputeKernelsQ4_0.quantizeActivationQ8Blocks}.
     *
     * <p>A lane takes sixteen weights per step: eight bytes of one 32-byte nibble run, whose low
     * nibbles are eight weights of sub-block {@code 2p} and whose high nibbles are the same eight
     * of sub-block {@code 2p + 1}. Each sub-block's six-bit scale multiplies its dot product; its
     * six-bit minimum multiplies the activation block's sum, which covers all 32 weights, so it is
     * applied once, by the lane holding the first eight. The scales are read by arithmetic select
     * rather than a branch on the sub-block. Worker: one workgroup of {@code localWorkGroupSize}
     * lanes (a multiple of 32) per row; {@code n} a multiple of 256.
     */
    // @formatter:on
    public static void matrixVectorGenericQ4_KDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            int localId = context.localIdx;
            float partial =
                    warpPartialDP4A(
                            context, xQuants, xScales, xSums, w, n, rowId, localWorkGroupSize);
            if ((localId & 31) == 0) {
                warpSums[localId >> 5] = partial;
            }
            context.localBarrier();
            if (localId == 0) {
                float total = 0.0f;
                for (int warp = 0; warp < localWorkGroupSize / 32; warp++) {
                    total += warpSums[warp];
                }
                output.set(rowId, total);
            }
        }
    }

    /**
     * {@code hb[row] += w[row]·x}, the packed-integer form of {@link
     * #matrixVectorGenericWithResidualQ4_K}: the residual projections of a decode layer (attention
     * output, FFN down) over the activation quantized by {@code quantizeActivationQ8Blocks}. Same
     * lane mapping and reduction as {@link #matrixVectorGenericQ4_KDP4A}.
     */
    public static void matrixVectorGenericWithResidualQ4_KDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        float[] warpSums = context.allocateFloatLocalArray(localWorkGroupSize / 32);
        if (rowId < d) {
            int localId = context.localIdx;
            float partial =
                    warpPartialDP4A(
                            context, xQuants, xScales, xSums, w, n, rowId, localWorkGroupSize);
            if ((localId & 31) == 0) {
                warpSums[localId >> 5] = partial;
            }
            context.localBarrier();
            if (localId == 0) {
                float total = 0.0f;
                for (int warp = 0; warp < localWorkGroupSize / 32; warp++) {
                    total += warpSums[warp];
                }
                hb.set(rowId, hb.get(rowId) + total);
            }
        }
    }

    /**
     * {@code hb[row] = silu(w1[row]·x) * (w3[row]·x)}, both Q4_K projections against the same
     * quantized activation in one launch. SiLU is applied to the gate reduced across every warp,
     * never per warp. Worker: one workgroup of {@code localWorkGroupSize} lanes (a multiple of 32)
     * per row; {@code n} a multiple of 256.
     */
    public static void fusedFFNGateUpSiLUQ4_KDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int warpCount = localWorkGroupSize / 32;
        // Gate in the first warpCount entries, up in the second.
        float[] warpSums = context.allocateFloatLocalArray(warpCount * 2);
        if (rowId < d) {
            int localId = context.localIdx;
            float gate =
                    rowPartialDP4A(
                            context, xQuants, xScales, xSums, w1, n, rowId, localWorkGroupSize);
            float up =
                    rowPartialDP4A(
                            context, xQuants, xScales, xSums, w3, n, rowId, localWorkGroupSize);
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
    }

    /**
     * This lane's share of row {@code rowId}'s dot product, folded across its warp with shuffles:
     * valid in the warp's lane 0. The cross-warp combine stays in the kernel, which owns the local
     * array it goes through.
     */
    private static float warpPartialDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        float partialSum =
                rowPartialDP4A(context, xQuants, xScales, xSums, w, n, rowId, localWorkGroupSize);
        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /**
     * This lane's share of row {@code rowId}'s dot product, sixteen weights per step as {@link
     * #matrixVectorGenericQ4_KDP4A} describes.
     */
    private static float rowPartialDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        int localId = context.localIdx;
        int superBlocksPerRow = n / QK_K;
        int rowByteOffset = rowId * superBlocksPerRow * BLOCK_BYTES;

        float partialSum = 0.0f;
        int unitsPerRow = n / 16;
        for (int unit = localId; unit < unitsPerRow; unit += localWorkGroupSize) {
            int superBlock = unit >> 4;
            int inBlock = unit & 15;
            int pair = inBlock >> 2;
            int quarter = inBlock & 3;
            int base = rowByteOffset + superBlock * BLOCK_BYTES;
            float dAll = w.getHalfFloat(base).getFloat32();
            float dMin = w.getHalfFloat(base + 2).getFloat32();

            // Scales and minimums of sub-blocks 2p and 2p+1. Sub-blocks 0-3 keep six bits in
            // bytes j and j+4; 4-7 take four bits from byte j+4 and their top two from the top
            // bits of bytes j-4 and j.
            int high = pair >> 1;
            int j0 = (pair << 1) & 3;
            int sc = base + SCALES_OFFSET;
            int lo0 = w.get(sc + j0) & 0xFF;
            int mid0 = w.get(sc + j0 + 4) & 0xFF;
            int hi0 = w.get(sc + j0 + 8) & 0xFF;
            int lo1 = w.get(sc + j0 + 1) & 0xFF;
            int mid1 = w.get(sc + j0 + 5) & 0xFF;
            int hi1 = w.get(sc + j0 + 9) & 0xFF;
            int scale0 = (lo0 & 63) + high * (((hi0 & 15) | ((lo0 >> 6) << 4)) - (lo0 & 63));
            int min0 = (mid0 & 63) + high * (((hi0 >> 4) | ((mid0 >> 6) << 4)) - (mid0 & 63));
            int scale1 = (lo1 & 63) + high * (((hi1 & 15) | ((lo1 >> 6) << 4)) - (lo1 & 63));
            int min1 = (mid1 & 63) + high * (((hi1 >> 4) | ((mid1 >> 6) << 4)) - (mid1 & 63));

            int qs = base + QS_OFFSET + pair * 32 + quarter * 8;
            int word0 =
                    (w.getHalfFloat(qs).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(qs + 2).getHalfFloatValue() & 0xFFFF) << 16);
            int word1 =
                    (w.getHalfFloat(qs + 4).getHalfFloatValue() & 0xFFFF)
                            | ((w.getHalfFloat(qs + 6).getHalfFloatValue() & 0xFFFF) << 16);
            int block0 = superBlock * 8 + pair * 2;
            int block1 = block0 + 1;
            int a0 = block0 * 8 + quarter * 2;
            int a1 = block1 * 8 + quarter * 2;
            int dot0 = QuantizationUtils.dp4a_packed(word0 & 0x0F0F0F0F, xQuants.get(a0), 0);
            dot0 = QuantizationUtils.dp4a_packed(word1 & 0x0F0F0F0F, xQuants.get(a0 + 1), dot0);
            int dot1 =
                    QuantizationUtils.dp4a_packed((word0 >>> 4) & 0x0F0F0F0F, xQuants.get(a1), 0);
            dot1 =
                    QuantizationUtils.dp4a_packed(
                            (word1 >>> 4) & 0x0F0F0F0F, xQuants.get(a1 + 1), dot1);
            float xs0 = xScales.get(block0);
            float xs1 = xScales.get(block1);
            float first = 1 - Math.min(quarter, 1);
            partialSum +=
                    dAll * (scale0 * xs0 * dot0 + scale1 * xs1 * dot1)
                            - first
                                    * dMin
                                    * (min0 * xs0 * xSums.get(block0)
                                            + min1 * xs1 * xSums.get(block1));
        }

        return partialSum;
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
