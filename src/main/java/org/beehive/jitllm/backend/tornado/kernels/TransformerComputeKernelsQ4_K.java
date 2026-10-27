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

    /**
     * Prompt rows a tiled batch workgroup covers: eight, as for Q5_K, whose decode is as costly.
     */
    private static final int ROW_TILE = 8;

    /** Output rows a tiled batch workgroup covers, as for Q5_K. */
    private static final int COL_TILE = 2;

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
     * {@link #dequantizeQ4_KToFP16} into a weight's resident FP16 copy, skipped once {@code
     * done[flag]} is set, on {@code TransformerBatchPrefillKernels.dequantizeQ8_0ToFP16Once}'s
     * fixed striding grid.
     */
    public static void dequantizeQ4_KToFP16Once(
            KernelContext context,
            ByteArray w,
            HalfFloatArray out,
            int destOffset,
            int n,
            IntArray done,
            int flag) {
        int gid = context.globalIdx;
        if (gid < TransformerBatchPrefillKernels.DECODE_ONCE_LANES && done.get(flag) == 0) {
            for (int i = gid; i < n; i += TransformerBatchPrefillKernels.DECODE_ONCE_LANES) {
                // decode(...) copied in with every read unconditional: called through the helper,
                // the
                // CUDA backend cannot lower the FP16 store's HalfFloat allocation (NewInstance).
                int block = i / QK_K;
                int blockByteOffset = block * BLOCK_BYTES;
                int withinBlock = i - block * QK_K;
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
                // Sub-blocks 0-3 read bytes [j] and [j+4]; 4-7 read [j+4], [j-4] and [j]. Reading
                // all
                // three stays inside the block: [j-4] for j < 4 lands on d/dmin and is discarded.
                int scalesBase = blockByteOffset + SCALES_OFFSET;
                int a = w.get(scalesBase + subBlock) & 0xFF;
                int b = w.get(scalesBase + subBlock + 4) & 0xFF;
                int c = w.get(scalesBase + subBlock - 4) & 0xFF;
                int low = subBlock < 4 ? 1 : 0;
                int sc = low * (a & 63) + (1 - low) * ((b & 0xF) | ((c >> 6) << 4));
                int m = low * (b & 63) + (1 - low) * (((b >> 4) & 0xF) | ((a >> 6) << 4));
                out.set(destOffset + i, new HalfFloat(d * sc * q - dmin * m));
            }
        }
    }

    /**
     * The dot product of the eight weights in four consecutive bytes of one 32-byte nibble run with
     * {@code x}. The bytes start at {@code posInRun}, a multiple of four below 32, of run {@code
     * pairIndex}: their low nibbles are four weights of sub-block {@code 2 * pairIndex} and their
     * high nibbles the same four of sub-block {@code 2 * pairIndex + 1}, 32 weights later. The
     * bytes, {@code d}, {@code dmin} and both sub-blocks' scales and minimums are read once rather
     * than once per weight. Each weight is still {@code d * sc * q - dmin * m} as {@link #decode}
     * computes it.
     */
    private static float dot8(
            ByteArray w, int blockByteOffset, int pairIndex, int posInRun, FloatArray x, int xAt) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();

        int scalesBase = blockByteOffset + SCALES_OFFSET;
        int lowPacked = scaleAndMin(w, scalesBase, pairIndex * 2);
        int highPacked = scaleAndMin(w, scalesBase, pairIndex * 2 + 1);
        float dScLow = d * (lowPacked >> 8);
        float dminMLow = dmin * (lowPacked & 0xFF);
        float dScHigh = d * (highPacked >> 8);
        float dminMHigh = dmin * (highPacked & 0xFF);

        int qAt = blockByteOffset + QS_OFFSET + pairIndex * 32 + posInRun;
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            int qByte = w.get(qAt + k) & 0xFF;
            sum += (dScLow * (qByte & 0xF) - dminMLow) * x.get(xAt + k);
            sum += (dScHigh * (qByte >> 4) - dminMHigh) * x.get(xAt + 32 + k);
        }
        return sum;
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
     * matrixVectorRowMajorOptimizedQ8_0Byte} makes. A thread takes eight weights per step, the two
     * nibbles of four consecutive bytes, through {@link #dot8}. {@code n} is a multiple of 256.
     */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int rowByteOffset = rowId * (n / QK_K) * BLOCK_BYTES;

        float partialSum = 0.0f;
        for (int unit = localId; unit < n / 8; unit += localSize) {
            int blockIdx = unit / 32;
            int unitInBlock = unit - blockIdx * 32;
            int pairIndex = unitInBlock / 8;
            int posInRun = (unitInBlock - pairIndex * 8) * 4;
            partialSum +=
                    dot8(
                            w,
                            rowByteOffset + blockIdx * BLOCK_BYTES,
                            pairIndex,
                            posInRun,
                            x,
                            blockIdx * QK_K + pairIndex * 64 + posInRun);
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
        int rowByteOffset = rowId * (n / QK_K) * BLOCK_BYTES;

        // Eight weights per step, as rowDotShared takes them.
        float gate = 0.0f;
        float up = 0.0f;
        for (int unit = localId; unit < n / 8; unit += localWorkGroupSize) {
            int blockIdx = unit / 32;
            int unitInBlock = unit - blockIdx * 32;
            int pairIndex = unitInBlock / 8;
            int posInRun = (unitInBlock - pairIndex * 8) * 4;
            int blockByteOffset = rowByteOffset + blockIdx * BLOCK_BYTES;
            int xAt = blockIdx * QK_K + pairIndex * 64 + posInRun;
            gate += dot8(w1, blockByteOffset, pairIndex, posInRun, x, xAt);
            up += dot8(w3, blockByteOffset, pairIndex, posInRun, x, xAt);
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

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    /** Output rows a tiled batch workgroup covers. */
    public static int colTile() {
        return COL_TILE;
    }

    /**
     * The six-bit scale and minimum of {@code subBlock}, packed as {@code (scale << 8) | min}: the
     * same unpacking as {@code TransformerComputeKernelsQ5_K.scaleAndMin}, kept in its own method
     * for the same reason, the tiled kernels' shared-memory reduction.
     */
    static int scaleAndMin(ByteArray w, int scalesBase, int subBlock) {
        if (subBlock < 4) {
            return ((w.get(scalesBase + subBlock) & 63) << 8)
                    | (w.get(scalesBase + subBlock + 4) & 63);
        }
        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
        int sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
        int m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a tile of up to {@link #ROW_TILE} prompt rows, one
     * workgroup per (row tile, output row tile), shaped like {@code
     * TransformerComputeKernelsQ5_K.matrixVectorTiledBatchQ5_K}: a warp takes one 32-weight
     * sub-block, a lane one weight of it, and each weight is decoded once for every row of the
     * tile.
     *
     * <p>Rows past {@code activeRows} contribute nothing and are not written, which is what makes a
     * partially filled chunk produce the same numbers as a full one.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ4_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int colGroups = (d + COL_TILE - 1) / COL_TILE;
        int groupId = context.groupIdx;
        int tile = groupId / colGroups;
        int colGroup = groupId - tile * colGroups;
        int firstRow = tile * ROW_TILE;
        int firstCol = colGroup * COL_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        float[] localSums =
                context.allocateFloatLocalArray(localWorkGroupSize * ROW_TILE * COL_TILE);
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int subBlocks = n / 32;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        int lane = localId & 31;
        int warp = localId >> 5;
        int warps = localWorkGroupSize >> 5;

        for (int sb = warp; sb < subBlocks; sb += warps) {
            int block = sb / 8;
            int subInBlock = sb - block * 8;
            int pairIndex = subInBlock >> 1;
            int highNibble = subInBlock & 1;
            int j = sb * 32 + lane;

            for (int r = 0; r < ROW_TILE; r++) {
                int row = firstRow + r;
                float value = 0.0f;
                if (row < activeRows) {
                    value = xBatch.get(row * n + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset = (outRow * blocksPerRow + block) * BLOCK_BYTES;
                    float dScale = w.getHalfFloat(blockByteOffset).getFloat32();
                    float dMin = w.getHalfFloat(blockByteOffset + 2).getFloat32();
                    int packed = scaleAndMin(w, blockByteOffset + SCALES_OFFSET, subInBlock);
                    float scale = dScale * (packed >> 8);
                    float minimum = dMin * (packed & 0xFF);
                    int qsByte = w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + lane) & 0xFF;
                    int q = qsByte & 0xF;
                    if (highNibble == 1) {
                        q = (qsByte >> 4) & 0xF;
                    }
                    float weight = scale * q - minimum;
                    for (int r = 0; r < ROW_TILE; r++) {
                        acc[c * ROW_TILE + r] += weight * xs[r];
                    }
                }
            }
        }

        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            localSums[t * localWorkGroupSize + localId] = acc[t];
        }
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
                    localSums[t * localWorkGroupSize + localId] +=
                            localSums[t * localWorkGroupSize + localId + stride];
                }
            }
            context.localBarrier();
        }

        if (localId == 0) {
            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                for (int r = 0; r < ROW_TILE; r++) {
                    int row = firstRow + r;
                    int slot = c * ROW_TILE + r;
                    if (outRow < d && row < activeRows) {
                        outBatch.set(row * d + outRow, localSums[slot * localWorkGroupSize]);
                    }
                }
            }
        }
    }

    /**
     * {@code out[b][row] += w[row]·x[b]} for a tile of rows. See {@link
     * #matrixVectorTiledBatchQ4_K}.
     */
    public static void matrixVectorTiledBatchWithResidualQ4_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int colGroups = (d + COL_TILE - 1) / COL_TILE;
        int groupId = context.groupIdx;
        int tile = groupId / colGroups;
        int colGroup = groupId - tile * colGroups;
        int firstRow = tile * ROW_TILE;
        int firstCol = colGroup * COL_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        float[] localSums =
                context.allocateFloatLocalArray(localWorkGroupSize * ROW_TILE * COL_TILE);
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int subBlocks = n / 32;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        int lane = localId & 31;
        int warp = localId >> 5;
        int warps = localWorkGroupSize >> 5;

        for (int sb = warp; sb < subBlocks; sb += warps) {
            int block = sb / 8;
            int subInBlock = sb - block * 8;
            int pairIndex = subInBlock >> 1;
            int highNibble = subInBlock & 1;
            int j = sb * 32 + lane;

            for (int r = 0; r < ROW_TILE; r++) {
                int row = firstRow + r;
                float value = 0.0f;
                if (row < activeRows) {
                    value = xBatch.get(row * n + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset = (outRow * blocksPerRow + block) * BLOCK_BYTES;
                    float dScale = w.getHalfFloat(blockByteOffset).getFloat32();
                    float dMin = w.getHalfFloat(blockByteOffset + 2).getFloat32();
                    int packed = scaleAndMin(w, blockByteOffset + SCALES_OFFSET, subInBlock);
                    float scale = dScale * (packed >> 8);
                    float minimum = dMin * (packed & 0xFF);
                    int qsByte = w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + lane) & 0xFF;
                    int q = qsByte & 0xF;
                    if (highNibble == 1) {
                        q = (qsByte >> 4) & 0xF;
                    }
                    float weight = scale * q - minimum;
                    for (int r = 0; r < ROW_TILE; r++) {
                        acc[c * ROW_TILE + r] += weight * xs[r];
                    }
                }
            }
        }

        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            localSums[t * localWorkGroupSize + localId] = acc[t];
        }
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
                    localSums[t * localWorkGroupSize + localId] +=
                            localSums[t * localWorkGroupSize + localId + stride];
                }
            }
            context.localBarrier();
        }

        if (localId == 0) {
            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                for (int r = 0; r < ROW_TILE; r++) {
                    int row = firstRow + r;
                    int slot = c * ROW_TILE + r;
                    if (outRow < d && row < activeRows) {
                        int index = row * d + outRow;
                        outBatch.set(
                                index, outBatch.get(index) + localSums[slot * localWorkGroupSize]);
                    }
                }
            }
        }
    }

    // @formatter:off
    /**
     * The fused gate/up feed-forward with SwiGLU over a tile of up to {@link #ROW_TILE} prompt
     * rows, {@code hb[b][row] = SiLU(w1[row]·x[b]) * (w3[row]·x[b])}, decoding each weight once for
     * the tile: the {@link #matrixVectorTiledBatchQ4_K} tiling with a gate and an up accumulator
     * per slot, as {@code TransformerComputeKernelsQ4_0.fusedFFNGateUpSiLUTiledBatchQ4_0} does.
     */
    // @formatter:on
    public static void fusedFFNGateUpSiLUTiledBatchQ4_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray hbBatch,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int colGroups = (d + COL_TILE - 1) / COL_TILE;
        int groupId = context.groupIdx;
        int tile = groupId / colGroups;
        int colGroup = groupId - tile * colGroups;
        int firstRow = tile * ROW_TILE;
        int firstCol = colGroup * COL_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        int slots = ROW_TILE * COL_TILE * 2;
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * slots);
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int subBlocks = n / 32;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack.
        float[] acc = new float[ROW_TILE * COL_TILE * 2];
        for (int t = 0; t < ROW_TILE * COL_TILE * 2; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        int lane = localId & 31;
        int warp = localId >> 5;
        int warps = localWorkGroupSize >> 5;

        for (int sb = warp; sb < subBlocks; sb += warps) {
            int block = sb / 8;
            int subInBlock = sb - block * 8;
            int pairIndex = subInBlock >> 1;
            int highNibble = subInBlock & 1;
            int j = sb * 32 + lane;

            for (int r = 0; r < ROW_TILE; r++) {
                int row = firstRow + r;
                float value = 0.0f;
                if (row < activeRows) {
                    value = xBatch.get(row * n + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset = (outRow * blocksPerRow + block) * BLOCK_BYTES;
                    int qsOffset = blockByteOffset + QS_OFFSET + pairIndex * 32 + lane;
                    int packedGate = scaleAndMin(w1, blockByteOffset + SCALES_OFFSET, subInBlock);
                    int gateByte = w1.get(qsOffset) & 0xFF;
                    int gateQ = gateByte & 0xF;
                    if (highNibble == 1) {
                        gateQ = (gateByte >> 4) & 0xF;
                    }
                    float gateWeight =
                            w1.getHalfFloat(blockByteOffset).getFloat32()
                                            * (packedGate >> 8)
                                            * gateQ
                                    - w1.getHalfFloat(blockByteOffset + 2).getFloat32()
                                            * (packedGate & 0xFF);
                    int packedUp = scaleAndMin(w3, blockByteOffset + SCALES_OFFSET, subInBlock);
                    int upByte = w3.get(qsOffset) & 0xFF;
                    int upQ = upByte & 0xF;
                    if (highNibble == 1) {
                        upQ = (upByte >> 4) & 0xF;
                    }
                    float upWeight =
                            w3.getHalfFloat(blockByteOffset).getFloat32() * (packedUp >> 8) * upQ
                                    - w3.getHalfFloat(blockByteOffset + 2).getFloat32()
                                            * (packedUp & 0xFF);
                    for (int r = 0; r < ROW_TILE; r++) {
                        int slot = (c * ROW_TILE + r) * 2;
                        acc[slot] += gateWeight * xs[r];
                        acc[slot + 1] += upWeight * xs[r];
                    }
                }
            }
        }

        for (int t = 0; t < slots; t++) {
            localSums[t * localWorkGroupSize + localId] = acc[t];
        }
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < slots; t++) {
                    localSums[t * localWorkGroupSize + localId] +=
                            localSums[t * localWorkGroupSize + localId + stride];
                }
            }
            context.localBarrier();
        }

        if (localId == 0) {
            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                for (int r = 0; r < ROW_TILE; r++) {
                    int row = firstRow + r;
                    if (outRow < d && row < activeRows) {
                        int slot = (c * ROW_TILE + r) * 2;
                        float gateSum = localSums[slot * localWorkGroupSize];
                        float upSum = localSums[(slot + 1) * localWorkGroupSize];
                        float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
                        hbBatch.set(row * d + outRow, silu * upSum);
                    }
                }
            }
        }
    }
}
