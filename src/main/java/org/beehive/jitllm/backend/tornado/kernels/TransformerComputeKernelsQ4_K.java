package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
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
     * The activation's scale-downs that let a 4-bit quant be masked where it lies in its word, not
     * shifted down: 256, 65536 and 2^24 for its second, third and fourth byte (exact).
     */
    private static final float S1 = 1.0f / 256.0f;

    private static final float S2 = 1.0f / 65536.0f;
    private static final float S3 = 1.0f / 16777216.0f;

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
     * FP16 operands. Lane {@code g} decodes elements {@code 4g} to {@code 4g + 3}: one sub-block's,
     * so {@code d}, {@code dmin} and the sub-block's scale and minimum are decoded once for the
     * four and their nibbles come from four consecutive bytes. The row width is a multiple of 256,
     * so element {@code i} lies in super-block {@code i / 256}. Any grid of at least {@code n / 4}
     * lanes; the rest do nothing.
     */
    public static void dequantizeQ4_KToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset, int n) {
        int gid = context.globalIdx;
        if (gid < n >>> 2) {
            // decode(...) copied in with every read unconditional: called through the helper, the
            // CUDA backend cannot lower the FP16 store's HalfFloat allocation (NewInstance).
            int first = gid << 2;
            int block = first / QK_K;
            int blockByteOffset = block * BLOCK_BYTES;
            int withinBlock = first - block * QK_K;
            float d = w.getHalfFloat(blockByteOffset).getFloat32();
            float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();
            int pairIndex = withinBlock / 64;
            int posInPair = withinBlock - pairIndex * 64;
            int highNibble = posInPair / 32;
            int subBlock = pairIndex * 2 + highNibble;
            // Sub-blocks 0-3 read bytes [j] and [j+4]; 4-7 read [j+4], [j-4] and [j]. Reading all
            // three stays inside the block: [j-4] for j < 4 lands on d/dmin and is discarded.
            int scalesBase = blockByteOffset + SCALES_OFFSET;
            int a = w.get(scalesBase + subBlock) & 0xFF;
            int b = w.get(scalesBase + subBlock + 4) & 0xFF;
            int c = w.get(scalesBase + subBlock - 4) & 0xFF;
            int low = subBlock < 4 ? 1 : 0;
            int sc = low * (a & 63) + (1 - low) * ((b & 0xF) | ((c >> 6) << 4));
            int m = low * (b & 63) + (1 - low) * (((b >> 4) & 0xF) | ((a >> 6) << 4));
            int qBase =
                    blockByteOffset + QS_OFFSET + pairIndex * 32 + (posInPair - highNibble * 32);
            int shift = highNibble * 4;
            for (int j = 0; j < 4; j++) {
                int q = ((w.get(qBase + j) & 0xFF) >> shift) & 0xF;
                out.set(destOffset + first + j, new HalfFloat(d * sc * q - dmin * m));
            }
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

    // @formatter:off
    /**
     * One lane's share of one row's super-block at {@code base}: 32 weights, in four runs of eight
     * (sub-blocks {@code 2iq}, {@code 2iq+1}, {@code 2iq+4} and {@code 2iq+5}). {@code qOffset} is
     * where the lane's quants start in {@code qs}; {@code shift} picks its two scale bytes of each
     * word. {@code ya1}, {@code ya2}, {@code ya3} (and so on) arrive divided by 256, 65536 and 2^24
     * so that the quants are masked where they lie; {@code sa..sd} are each run's plain sum, for
     * the mins.
     */
    // @formatter:on
    private static float blockDot(
            ByteArray w,
            int base,
            int shift,
            int qOffset,
            float ya0,
            float ya1,
            float ya2,
            float ya3,
            float ya4,
            float ya5,
            float ya6,
            float ya7,
            float yb0,
            float yb1,
            float yb2,
            float yb3,
            float yb4,
            float yb5,
            float yb6,
            float yb7,
            float yc0,
            float yc1,
            float yc2,
            float yc3,
            float yc4,
            float yc5,
            float yc6,
            float yc7,
            float yd0,
            float yd1,
            float yd2,
            float yd3,
            float yd4,
            float yd5,
            float yd6,
            float yd7,
            float sa,
            float sb,
            float sc,
            float sd) {
        float dd = w.getHalfFloat(base).getFloat32();
        float dmin = w.getHalfFloat(base + 2).getFloat32();
        int a = (w.getInt(base + SCALES_OFFSET) >>> shift) & 0xFFFF;
        int b = (w.getInt(base + SCALES_OFFSET + 4) >>> shift) & 0xFFFF;
        int c = (w.getInt(base + SCALES_OFFSET + 8) >>> shift) & 0xFFFF;
        int qa = base + QS_OFFSET + qOffset;
        int q0 = w.getInt(qa);
        int q1 = w.getInt(qa + 4);
        int q2 = w.getInt(qa + 64);
        int q3 = w.getInt(qa + 68);
        int h0 = q0 >>> 4;
        int h1 = q1 >>> 4;
        int h2 = q2 >>> 4;
        int h3 = q3 >>> 4;
        float da =
                ya0 * (q0 & 0xF)
                        + ya1 * (q0 & 0xF00)
                        + ya2 * (q0 & 0xF0000)
                        + ya3 * (q0 & 0xF000000)
                        + ya4 * (q1 & 0xF)
                        + ya5 * (q1 & 0xF00)
                        + ya6 * (q1 & 0xF0000)
                        + ya7 * (q1 & 0xF000000);
        float db =
                yb0 * (h0 & 0xF)
                        + yb1 * (h0 & 0xF00)
                        + yb2 * (h0 & 0xF0000)
                        + yb3 * (h0 & 0xF000000)
                        + yb4 * (h1 & 0xF)
                        + yb5 * (h1 & 0xF00)
                        + yb6 * (h1 & 0xF0000)
                        + yb7 * (h1 & 0xF000000);
        float dc =
                yc0 * (q2 & 0xF)
                        + yc1 * (q2 & 0xF00)
                        + yc2 * (q2 & 0xF0000)
                        + yc3 * (q2 & 0xF000000)
                        + yc4 * (q3 & 0xF)
                        + yc5 * (q3 & 0xF00)
                        + yc6 * (q3 & 0xF0000)
                        + yc7 * (q3 & 0xF000000);
        float de =
                yd0 * (h2 & 0xF)
                        + yd1 * (h2 & 0xF00)
                        + yd2 * (h2 & 0xF0000)
                        + yd3 * (h2 & 0xF000000)
                        + yd4 * (h3 & 0xF)
                        + yd5 * (h3 & 0xF00)
                        + yd6 * (h3 & 0xF0000)
                        + yd7 * (h3 & 0xF000000);
        // Sub-blocks 2iq and 2iq+1 take the plain 6-bit pairs; 2iq+4 and 2iq+5 take four bits
        // from c and two from the top of a (scale) and b (min).
        int sc0 = a & 63;
        int sc1 = (a >>> 8) & 63;
        int m0 = b & 63;
        int m1 = (b >>> 8) & 63;
        int sc4 = (c & 0xF) | (((a >>> 6) & 3) << 4);
        int sc5 = ((c >>> 8) & 0xF) | (((a >>> 14) & 3) << 4);
        int m4 = ((c >>> 4) & 0xF) | (((b >>> 6) & 3) << 4);
        int m5 = ((c >>> 12) & 0xF) | (((b >>> 14) & 3) << 4);
        return dd * (sc0 * da + sc1 * db + sc4 * dc + sc5 * de)
                - dmin * (m0 * sa + m1 * sb + m4 * sc + m5 * sd);
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
        return rowDotShared(context, localSize, x, 0, w, n, rowId);
    }

    /**
     * The same reduction against the activation that starts at {@code xOffset}: one row of a batch,
     * or one expert slot's hidden activation.
     */
    static float rowDotShared(
            KernelContext context,
            int localSize,
            FloatArray x,
            int xOffset,
            ByteArray w,
            int n,
            int rowId) {
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
                            xOffset + blockIdx * QK_K + pairIndex * 64 + posInRun);
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

    /**
     * {@code hb[row] = SiLU(w1[row]·x) * (w3[row]·x)} for four rows per SIMD group (Metal): the
     * activation a lane loads serves the four gate rows and the four up rows. Launch {@link
     * #SIMDGROUP_GROUP_ROWS} rows per workgroup of {@code 32 * SIMDGROUPS} threads, as {@link
     * #matrixVectorQ4_KSimdgroup}.
     */
    public static void fusedFFNGateUpSiLUQ4_KSimdgroup(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d) {
        int lane = context.localIdx & 31;
        int row0 = (context.groupIdx * SIMDGROUPS + (context.localIdx >> 5)) * SIMDGROUP_ROWS;
        int blocks = n / QK_K;
        int rowBytes = blocks * BLOCK_BYTES;
        int ix = lane >> 3;
        int iq = (lane & 7) >> 2;
        int ir = lane & 3;
        int shift = 16 * iq;
        float gate0 = 0.0f;
        float gate1 = 0.0f;
        float gate2 = 0.0f;
        float gate3 = 0.0f;
        float up0 = 0.0f;
        float up1 = 0.0f;
        float up2 = 0.0f;
        float up3 = 0.0f;
        for (int ib = ix; ib < blocks; ib += 4) {
            int xb = ib * QK_K + 64 * iq + 8 * ir;
            Float4 xa0 = x.getFloat4(xb + 0);
            Float4 xa1 = x.getFloat4(xb + 4);
            Float4 xb0 = x.getFloat4(xb + 32);
            Float4 xb1 = x.getFloat4(xb + 36);
            Float4 xc0 = x.getFloat4(xb + 128);
            Float4 xc1 = x.getFloat4(xb + 132);
            Float4 xd0 = x.getFloat4(xb + 160);
            Float4 xd1 = x.getFloat4(xb + 164);
            float sa =
                    xa0.getX()
                            + xa0.getY()
                            + xa0.getZ()
                            + xa0.getW()
                            + xa1.getX()
                            + xa1.getY()
                            + xa1.getZ()
                            + xa1.getW();
            float sb =
                    xb0.getX()
                            + xb0.getY()
                            + xb0.getZ()
                            + xb0.getW()
                            + xb1.getX()
                            + xb1.getY()
                            + xb1.getZ()
                            + xb1.getW();
            float sc =
                    xc0.getX()
                            + xc0.getY()
                            + xc0.getZ()
                            + xc0.getW()
                            + xc1.getX()
                            + xc1.getY()
                            + xc1.getZ()
                            + xc1.getW();
            float sd =
                    xd0.getX()
                            + xd0.getY()
                            + xd0.getZ()
                            + xd0.getW()
                            + xd1.getX()
                            + xd1.getY()
                            + xd1.getZ()
                            + xd1.getW();
            float ya0 = xa0.getX();
            float ya1 = xa0.getY() * S1;
            float ya2 = xa0.getZ() * S2;
            float ya3 = xa0.getW() * S3;
            float ya4 = xa1.getX();
            float ya5 = xa1.getY() * S1;
            float ya6 = xa1.getZ() * S2;
            float ya7 = xa1.getW() * S3;
            float yb0 = xb0.getX();
            float yb1 = xb0.getY() * S1;
            float yb2 = xb0.getZ() * S2;
            float yb3 = xb0.getW() * S3;
            float yb4 = xb1.getX();
            float yb5 = xb1.getY() * S1;
            float yb6 = xb1.getZ() * S2;
            float yb7 = xb1.getW() * S3;
            float yc0 = xc0.getX();
            float yc1 = xc0.getY() * S1;
            float yc2 = xc0.getZ() * S2;
            float yc3 = xc0.getW() * S3;
            float yc4 = xc1.getX();
            float yc5 = xc1.getY() * S1;
            float yc6 = xc1.getZ() * S2;
            float yc7 = xc1.getW() * S3;
            float yd0 = xd0.getX();
            float yd1 = xd0.getY() * S1;
            float yd2 = xd0.getZ() * S2;
            float yd3 = xd0.getW() * S3;
            float yd4 = xd1.getX();
            float yd5 = xd1.getY() * S1;
            float yd6 = xd1.getZ() * S2;
            float yd7 = xd1.getW() * S3;
            for (int r = 0; r < SIMDGROUP_ROWS; r++) {
                int row = row0 + r;
                if (row < d) {
                    int base = row * rowBytes + ib * BLOCK_BYTES;
                    float g =
                            blockDot(
                                    w1,
                                    base,
                                    shift,
                                    32 * iq + 8 * ir,
                                    ya0,
                                    ya1,
                                    ya2,
                                    ya3,
                                    ya4,
                                    ya5,
                                    ya6,
                                    ya7,
                                    yb0,
                                    yb1,
                                    yb2,
                                    yb3,
                                    yb4,
                                    yb5,
                                    yb6,
                                    yb7,
                                    yc0,
                                    yc1,
                                    yc2,
                                    yc3,
                                    yc4,
                                    yc5,
                                    yc6,
                                    yc7,
                                    yd0,
                                    yd1,
                                    yd2,
                                    yd3,
                                    yd4,
                                    yd5,
                                    yd6,
                                    yd7,
                                    sa,
                                    sb,
                                    sc,
                                    sd);
                    ;
                    float u =
                            blockDot(
                                    w3,
                                    base,
                                    shift,
                                    32 * iq + 8 * ir,
                                    ya0,
                                    ya1,
                                    ya2,
                                    ya3,
                                    ya4,
                                    ya5,
                                    ya6,
                                    ya7,
                                    yb0,
                                    yb1,
                                    yb2,
                                    yb3,
                                    yb4,
                                    yb5,
                                    yb6,
                                    yb7,
                                    yc0,
                                    yc1,
                                    yc2,
                                    yc3,
                                    yc4,
                                    yc5,
                                    yc6,
                                    yc7,
                                    yd0,
                                    yd1,
                                    yd2,
                                    yd3,
                                    yd4,
                                    yd5,
                                    yd6,
                                    yd7,
                                    sa,
                                    sb,
                                    sc,
                                    sd);
                    ;
                    if (r == 0) {
                        gate0 += g;
                        up0 += u;
                    } else if (r == 1) {
                        gate1 += g;
                        up1 += u;
                    } else if (r == 2) {
                        gate2 += g;
                        up2 += u;
                    } else {
                        gate3 += g;
                        up3 += u;
                    }
                }
            }
        }
        gate0 = simdSum(context, gate0);
        gate1 = simdSum(context, gate1);
        gate2 = simdSum(context, gate2);
        gate3 = simdSum(context, gate3);
        up0 = simdSum(context, up0);
        up1 = simdSum(context, up1);
        up2 = simdSum(context, up2);
        up3 = simdSum(context, up3);
        if (lane == 0) {
            storeSwiGlu(hb, row0, d, gate0, up0);
            storeSwiGlu(hb, row0 + 1, d, gate1, up1);
            storeSwiGlu(hb, row0 + 2, d, gate2, up2);
            storeSwiGlu(hb, row0 + 3, d, gate3, up3);
        }
    }

    /** {@code hb[row] = SiLU(gate) * up} where the row exists. */
    private static void storeSwiGlu(FloatArray hb, int row, int d, float gate, float up) {
        if (row < d) {
            hb.set(row, gate / (1.0f + TornadoMath.exp(-gate)) * up);
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
     * Sub-tiles of {@link #ROW_TILE} rows in one expert tile of {@link
     * #groupedExpertTiledBatchQ4_K}.
     */
    public static final int GROUPED_SUB_TILES = 64 / ROW_TILE;

    // @formatter:off
    /**
     * {@link #matrixVectorTiledBatchQ4_K} over one tile of expert rows, for a batched
     * mixture-of-experts prefill off Metal: {@code tiles} as {@code
     * Qwen35MoeBatchKernels.groupByExpert} writes it, {@code (expert, first sorted row, rows)},
     * each cut into {@link #GROUPED_SUB_TILES} sub-tiles of {@link #ROW_TILE} rows. The expert's
     * {@code d} weight rows are read in place from the stacked tensor; row {@code i} reads input
     * row {@code sortedToken[first + i]} when {@code gather} is 1, {@code first + i} otherwise, and
     * writes output row {@code first + i}. Grid: {@code maxTiles * GROUPED_SUB_TILES * ceil(d /
     * COL_TILE)} workgroups of {@code localWorkGroupSize} lanes.
     */
    // @formatter:on
    public static void groupedExpertTiledBatchQ4_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            IntArray tiles,
            IntArray sortedToken,
            int gather,
            int n,
            int d,
            int localWorkGroupSize) {
        int colGroups = (d + COL_TILE - 1) / COL_TILE;
        int groupId = context.groupIdx;
        int job = groupId / colGroups;
        int tile = job / GROUPED_SUB_TILES;
        if (tile >= tiles.get(0)) {
            return;
        }
        int expert = tiles.get(1 + 3 * tile);
        int sub = job - tile * GROUPED_SUB_TILES;
        int first = tiles.get(2 + 3 * tile) + sub * ROW_TILE;
        int activeRows = tiles.get(3 + 3 * tile) - sub * ROW_TILE;
        int colGroup = groupId - job * colGroups;
        int firstRow = 0;
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
                    value =
                            xBatch.get(
                                    (gather != 0 ? sortedToken.get(first + row) : first + row) * n
                                            + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset =
                            ((expert * d + outRow) * blocksPerRow + block) * BLOCK_BYTES;
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
                        outBatch.set(
                                (first + row) * d + outRow, localSums[slot * localWorkGroupSize]);
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

    /**
     * Output rows per SIMD group of {@link #matrixVectorQ4_KSimdgroup}: the activation a lane loads
     * serves all four.
     */
    public static final int SIMDGROUP_ROWS = 4;

    /** SIMD groups per workgroup of {@link #matrixVectorQ4_KSimdgroup}. */
    public static final int SIMDGROUPS = 2;

    /** Output rows per workgroup of {@link #matrixVectorQ4_KSimdgroup}. */
    public static final int SIMDGROUP_GROUP_ROWS = SIMDGROUP_ROWS * SIMDGROUPS;

    /**
     * {@code output[row] = w[row]·x}, one SIMD group per four rows (Metal).
     *
     * <p>Eight lanes cover a super-block, so a SIMD group walks four at a time. Each lane reads 32
     * weights of a super-block as four 32-bit words and the scales as three, instead of a byte per
     * weight. Example: lane 13 takes super-blocks 1, 5, 9, ... and, in each, weights 8..15 of
     * sub-blocks 2, 3, 6 and 7. A nibble is masked in place, not shifted down: the activation it
     * multiplies was divided by 256, 65536 or 2^24 instead (exact, powers of two). Launch {@link
     * #SIMDGROUP_GROUP_ROWS} rows per workgroup of {@code 32 * SIMDGROUPS} threads.
     */
    public static void matrixVectorQ4_KSimdgroup(
            KernelContext context, FloatArray x, FloatArray out, ByteArray w, int n, int d) {
        // The body is here, not in a shared helper: TornadoVM 7.3 refuses to inline a callee of
        // more than 600 nodes, and the kernel root has no such limit.
        int lane = context.localIdx & 31;
        int row0 = (context.groupIdx * SIMDGROUPS + (context.localIdx >> 5)) * SIMDGROUP_ROWS;
        int blocks = n / QK_K;
        int rowBytes = blocks * BLOCK_BYTES;
        int ix = lane >> 3;
        int iq = (lane & 7) >> 2;
        int ir = lane & 3;
        int shift = 16 * iq;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int ib = ix; ib < blocks; ib += 4) {
            int xb = ib * QK_K + 64 * iq + 8 * ir;
            Float4 xa0 = x.getFloat4(xb + 0);
            Float4 xa1 = x.getFloat4(xb + 4);
            Float4 xb0 = x.getFloat4(xb + 32);
            Float4 xb1 = x.getFloat4(xb + 36);
            Float4 xc0 = x.getFloat4(xb + 128);
            Float4 xc1 = x.getFloat4(xb + 132);
            Float4 xd0 = x.getFloat4(xb + 160);
            Float4 xd1 = x.getFloat4(xb + 164);
            float sa =
                    xa0.getX()
                            + xa0.getY()
                            + xa0.getZ()
                            + xa0.getW()
                            + xa1.getX()
                            + xa1.getY()
                            + xa1.getZ()
                            + xa1.getW();
            float sb =
                    xb0.getX()
                            + xb0.getY()
                            + xb0.getZ()
                            + xb0.getW()
                            + xb1.getX()
                            + xb1.getY()
                            + xb1.getZ()
                            + xb1.getW();
            float sc =
                    xc0.getX()
                            + xc0.getY()
                            + xc0.getZ()
                            + xc0.getW()
                            + xc1.getX()
                            + xc1.getY()
                            + xc1.getZ()
                            + xc1.getW();
            float sd =
                    xd0.getX()
                            + xd0.getY()
                            + xd0.getZ()
                            + xd0.getW()
                            + xd1.getX()
                            + xd1.getY()
                            + xd1.getZ()
                            + xd1.getW();
            float ya0 = xa0.getX();
            float ya1 = xa0.getY() * S1;
            float ya2 = xa0.getZ() * S2;
            float ya3 = xa0.getW() * S3;
            float ya4 = xa1.getX();
            float ya5 = xa1.getY() * S1;
            float ya6 = xa1.getZ() * S2;
            float ya7 = xa1.getW() * S3;
            float yb0 = xb0.getX();
            float yb1 = xb0.getY() * S1;
            float yb2 = xb0.getZ() * S2;
            float yb3 = xb0.getW() * S3;
            float yb4 = xb1.getX();
            float yb5 = xb1.getY() * S1;
            float yb6 = xb1.getZ() * S2;
            float yb7 = xb1.getW() * S3;
            float yc0 = xc0.getX();
            float yc1 = xc0.getY() * S1;
            float yc2 = xc0.getZ() * S2;
            float yc3 = xc0.getW() * S3;
            float yc4 = xc1.getX();
            float yc5 = xc1.getY() * S1;
            float yc6 = xc1.getZ() * S2;
            float yc7 = xc1.getW() * S3;
            float yd0 = xd0.getX();
            float yd1 = xd0.getY() * S1;
            float yd2 = xd0.getZ() * S2;
            float yd3 = xd0.getW() * S3;
            float yd4 = xd1.getX();
            float yd5 = xd1.getY() * S1;
            float yd6 = xd1.getZ() * S2;
            float yd7 = xd1.getW() * S3;
            for (int r = 0; r < SIMDGROUP_ROWS; r++) {
                int row = row0 + r;
                if (row < d) {
                    float part =
                            blockDot(
                                    w,
                                    row * rowBytes + ib * BLOCK_BYTES,
                                    shift,
                                    32 * iq + 8 * ir,
                                    ya0,
                                    ya1,
                                    ya2,
                                    ya3,
                                    ya4,
                                    ya5,
                                    ya6,
                                    ya7,
                                    yb0,
                                    yb1,
                                    yb2,
                                    yb3,
                                    yb4,
                                    yb5,
                                    yb6,
                                    yb7,
                                    yc0,
                                    yc1,
                                    yc2,
                                    yc3,
                                    yc4,
                                    yc5,
                                    yc6,
                                    yc7,
                                    yd0,
                                    yd1,
                                    yd2,
                                    yd3,
                                    yd4,
                                    yd5,
                                    yd6,
                                    yd7,
                                    sa,
                                    sb,
                                    sc,
                                    sd);
                    if (r == 0) {
                        sum0 += part;
                    } else if (r == 1) {
                        sum1 += part;
                    } else if (r == 2) {
                        sum2 += part;
                    } else {
                        sum3 += part;
                    }
                }
            }
        }
        sum0 = simdSum(context, sum0);
        sum1 = simdSum(context, sum1);
        sum2 = simdSum(context, sum2);
        sum3 = simdSum(context, sum3);
        if (lane == 0) {
            store(out, row0, d, sum0, false);
            store(out, row0 + 1, d, sum1, false);
            store(out, row0 + 2, d, sum2, false);
            store(out, row0 + 3, d, sum3, false);
        }
    }

    /** {@code hb[row] += w[row]·x}; see {@link #matrixVectorQ4_KSimdgroup}. */
    public static void matrixVectorWithResidualQ4_KSimdgroup(
            KernelContext context, FloatArray x, FloatArray hb, ByteArray w, int n, int d) {
        int lane = context.localIdx & 31;
        int row0 = (context.groupIdx * SIMDGROUPS + (context.localIdx >> 5)) * SIMDGROUP_ROWS;
        int blocks = n / QK_K;
        int rowBytes = blocks * BLOCK_BYTES;
        int ix = lane >> 3;
        int iq = (lane & 7) >> 2;
        int ir = lane & 3;
        int shift = 16 * iq;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int ib = ix; ib < blocks; ib += 4) {
            int xb = ib * QK_K + 64 * iq + 8 * ir;
            Float4 xa0 = x.getFloat4(xb + 0);
            Float4 xa1 = x.getFloat4(xb + 4);
            Float4 xb0 = x.getFloat4(xb + 32);
            Float4 xb1 = x.getFloat4(xb + 36);
            Float4 xc0 = x.getFloat4(xb + 128);
            Float4 xc1 = x.getFloat4(xb + 132);
            Float4 xd0 = x.getFloat4(xb + 160);
            Float4 xd1 = x.getFloat4(xb + 164);
            float sa =
                    xa0.getX()
                            + xa0.getY()
                            + xa0.getZ()
                            + xa0.getW()
                            + xa1.getX()
                            + xa1.getY()
                            + xa1.getZ()
                            + xa1.getW();
            float sb =
                    xb0.getX()
                            + xb0.getY()
                            + xb0.getZ()
                            + xb0.getW()
                            + xb1.getX()
                            + xb1.getY()
                            + xb1.getZ()
                            + xb1.getW();
            float sc =
                    xc0.getX()
                            + xc0.getY()
                            + xc0.getZ()
                            + xc0.getW()
                            + xc1.getX()
                            + xc1.getY()
                            + xc1.getZ()
                            + xc1.getW();
            float sd =
                    xd0.getX()
                            + xd0.getY()
                            + xd0.getZ()
                            + xd0.getW()
                            + xd1.getX()
                            + xd1.getY()
                            + xd1.getZ()
                            + xd1.getW();
            float ya0 = xa0.getX();
            float ya1 = xa0.getY() * S1;
            float ya2 = xa0.getZ() * S2;
            float ya3 = xa0.getW() * S3;
            float ya4 = xa1.getX();
            float ya5 = xa1.getY() * S1;
            float ya6 = xa1.getZ() * S2;
            float ya7 = xa1.getW() * S3;
            float yb0 = xb0.getX();
            float yb1 = xb0.getY() * S1;
            float yb2 = xb0.getZ() * S2;
            float yb3 = xb0.getW() * S3;
            float yb4 = xb1.getX();
            float yb5 = xb1.getY() * S1;
            float yb6 = xb1.getZ() * S2;
            float yb7 = xb1.getW() * S3;
            float yc0 = xc0.getX();
            float yc1 = xc0.getY() * S1;
            float yc2 = xc0.getZ() * S2;
            float yc3 = xc0.getW() * S3;
            float yc4 = xc1.getX();
            float yc5 = xc1.getY() * S1;
            float yc6 = xc1.getZ() * S2;
            float yc7 = xc1.getW() * S3;
            float yd0 = xd0.getX();
            float yd1 = xd0.getY() * S1;
            float yd2 = xd0.getZ() * S2;
            float yd3 = xd0.getW() * S3;
            float yd4 = xd1.getX();
            float yd5 = xd1.getY() * S1;
            float yd6 = xd1.getZ() * S2;
            float yd7 = xd1.getW() * S3;
            for (int r = 0; r < SIMDGROUP_ROWS; r++) {
                int row = row0 + r;
                if (row < d) {
                    float part =
                            blockDot(
                                    w,
                                    row * rowBytes + ib * BLOCK_BYTES,
                                    shift,
                                    32 * iq + 8 * ir,
                                    ya0,
                                    ya1,
                                    ya2,
                                    ya3,
                                    ya4,
                                    ya5,
                                    ya6,
                                    ya7,
                                    yb0,
                                    yb1,
                                    yb2,
                                    yb3,
                                    yb4,
                                    yb5,
                                    yb6,
                                    yb7,
                                    yc0,
                                    yc1,
                                    yc2,
                                    yc3,
                                    yc4,
                                    yc5,
                                    yc6,
                                    yc7,
                                    yd0,
                                    yd1,
                                    yd2,
                                    yd3,
                                    yd4,
                                    yd5,
                                    yd6,
                                    yd7,
                                    sa,
                                    sb,
                                    sc,
                                    sd);
                    if (r == 0) {
                        sum0 += part;
                    } else if (r == 1) {
                        sum1 += part;
                    } else if (r == 2) {
                        sum2 += part;
                    } else {
                        sum3 += part;
                    }
                }
            }
        }
        sum0 = simdSum(context, sum0);
        sum1 = simdSum(context, sum1);
        sum2 = simdSum(context, sum2);
        sum3 = simdSum(context, sum3);
        if (lane == 0) {
            store(hb, row0, d, sum0, true);
            store(hb, row0 + 1, d, sum1, true);
            store(hb, row0 + 2, d, sum2, true);
            store(hb, row0 + 3, d, sum3, true);
        }
    }

    /**
     * {@code hb[s·H + r] = silu(gate_e[r]·x) · (up_e[r]·x)} for every selected slot {@code s} and
     * expert row {@code r}, {@code e = selected[s]}, over stacked {@code [experts·H, n]} gate and
     * up tensors: {@link #matrixVectorQ4_KSimdgroup}'s scheme, with each super-block's activation
     * loaded once for both tensors. {@code H} is a multiple of four, so a SIMD group's four rows
     * belong to one slot. Example: LFM2-8B-A1B, 4 slots × 1792 rows over {@code n = 2048}. Launch
     * {@link #SIMDGROUP_GROUP_ROWS} rows per workgroup of {@code 32 * SIMDGROUPS} threads.
     */
    public static void expertGateUpSiLUQ4_KSimdgroup(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray gate,
            ByteArray up,
            IntArray selected,
            int n,
            int expertHidden,
            int used) {
        int lane = context.localIdx & 31;
        int row0 = (context.groupIdx * SIMDGROUPS + (context.localIdx >> 5)) * SIMDGROUP_ROWS;
        int rows = used * expertHidden;
        if (row0 >= rows) {
            return;
        }
        int slot = row0 / expertHidden;
        int wrow0 = selected.get(slot) * expertHidden + (row0 - slot * expertHidden);
        int blocks = n / QK_K;
        int rowBytes = blocks * BLOCK_BYTES;
        int ix = lane >> 3;
        int iq = (lane & 7) >> 2;
        int ir = lane & 3;
        int shift = 16 * iq;
        float g0 = 0.0f;
        float g1 = 0.0f;
        float g2 = 0.0f;
        float g3 = 0.0f;
        float u0 = 0.0f;
        float u1 = 0.0f;
        float u2 = 0.0f;
        float u3 = 0.0f;
        for (int ib = ix; ib < blocks; ib += 4) {
            int xb = ib * QK_K + 64 * iq + 8 * ir;
            Float4 xa0 = x.getFloat4(xb + 0);
            Float4 xa1 = x.getFloat4(xb + 4);
            Float4 xb0 = x.getFloat4(xb + 32);
            Float4 xb1 = x.getFloat4(xb + 36);
            Float4 xc0 = x.getFloat4(xb + 128);
            Float4 xc1 = x.getFloat4(xb + 132);
            Float4 xd0 = x.getFloat4(xb + 160);
            Float4 xd1 = x.getFloat4(xb + 164);
            float sa =
                    xa0.getX()
                            + xa0.getY()
                            + xa0.getZ()
                            + xa0.getW()
                            + xa1.getX()
                            + xa1.getY()
                            + xa1.getZ()
                            + xa1.getW();
            float sb =
                    xb0.getX()
                            + xb0.getY()
                            + xb0.getZ()
                            + xb0.getW()
                            + xb1.getX()
                            + xb1.getY()
                            + xb1.getZ()
                            + xb1.getW();
            float sc =
                    xc0.getX()
                            + xc0.getY()
                            + xc0.getZ()
                            + xc0.getW()
                            + xc1.getX()
                            + xc1.getY()
                            + xc1.getZ()
                            + xc1.getW();
            float sd =
                    xd0.getX()
                            + xd0.getY()
                            + xd0.getZ()
                            + xd0.getW()
                            + xd1.getX()
                            + xd1.getY()
                            + xd1.getZ()
                            + xd1.getW();
            float ya0 = xa0.getX();
            float ya1 = xa0.getY() * S1;
            float ya2 = xa0.getZ() * S2;
            float ya3 = xa0.getW() * S3;
            float ya4 = xa1.getX();
            float ya5 = xa1.getY() * S1;
            float ya6 = xa1.getZ() * S2;
            float ya7 = xa1.getW() * S3;
            float yb0 = xb0.getX();
            float yb1 = xb0.getY() * S1;
            float yb2 = xb0.getZ() * S2;
            float yb3 = xb0.getW() * S3;
            float yb4 = xb1.getX();
            float yb5 = xb1.getY() * S1;
            float yb6 = xb1.getZ() * S2;
            float yb7 = xb1.getW() * S3;
            float yc0 = xc0.getX();
            float yc1 = xc0.getY() * S1;
            float yc2 = xc0.getZ() * S2;
            float yc3 = xc0.getW() * S3;
            float yc4 = xc1.getX();
            float yc5 = xc1.getY() * S1;
            float yc6 = xc1.getZ() * S2;
            float yc7 = xc1.getW() * S3;
            float yd0 = xd0.getX();
            float yd1 = xd0.getY() * S1;
            float yd2 = xd0.getZ() * S2;
            float yd3 = xd0.getW() * S3;
            float yd4 = xd1.getX();
            float yd5 = xd1.getY() * S1;
            float yd6 = xd1.getZ() * S2;
            float yd7 = xd1.getW() * S3;
            for (int r = 0; r < SIMDGROUP_ROWS; r++) {
                int at = (wrow0 + r) * rowBytes + ib * BLOCK_BYTES;
                float pg =
                        blockDot(
                                gate,
                                at,
                                shift,
                                32 * iq + 8 * ir,
                                ya0,
                                ya1,
                                ya2,
                                ya3,
                                ya4,
                                ya5,
                                ya6,
                                ya7,
                                yb0,
                                yb1,
                                yb2,
                                yb3,
                                yb4,
                                yb5,
                                yb6,
                                yb7,
                                yc0,
                                yc1,
                                yc2,
                                yc3,
                                yc4,
                                yc5,
                                yc6,
                                yc7,
                                yd0,
                                yd1,
                                yd2,
                                yd3,
                                yd4,
                                yd5,
                                yd6,
                                yd7,
                                sa,
                                sb,
                                sc,
                                sd);
                float pu =
                        blockDot(
                                up,
                                at,
                                shift,
                                32 * iq + 8 * ir,
                                ya0,
                                ya1,
                                ya2,
                                ya3,
                                ya4,
                                ya5,
                                ya6,
                                ya7,
                                yb0,
                                yb1,
                                yb2,
                                yb3,
                                yb4,
                                yb5,
                                yb6,
                                yb7,
                                yc0,
                                yc1,
                                yc2,
                                yc3,
                                yc4,
                                yc5,
                                yc6,
                                yc7,
                                yd0,
                                yd1,
                                yd2,
                                yd3,
                                yd4,
                                yd5,
                                yd6,
                                yd7,
                                sa,
                                sb,
                                sc,
                                sd);
                if (r == 0) {
                    g0 += pg;
                    u0 += pu;
                } else if (r == 1) {
                    g1 += pg;
                    u1 += pu;
                } else if (r == 2) {
                    g2 += pg;
                    u2 += pu;
                } else {
                    g3 += pg;
                    u3 += pu;
                }
            }
        }
        g0 = simdSum(context, g0);
        g1 = simdSum(context, g1);
        g2 = simdSum(context, g2);
        g3 = simdSum(context, g3);
        u0 = simdSum(context, u0);
        u1 = simdSum(context, u1);
        u2 = simdSum(context, u2);
        u3 = simdSum(context, u3);
        if (lane == 0) {
            store(hb, row0, rows, g0 / (1.0f + TornadoMath.exp(-g0)) * u0, false);
            store(hb, row0 + 1, rows, g1 / (1.0f + TornadoMath.exp(-g1)) * u1, false);
            store(hb, row0 + 2, rows, g2 / (1.0f + TornadoMath.exp(-g2)) * u2, false);
            store(hb, row0 + 3, rows, g3 / (1.0f + TornadoMath.exp(-g3)) * u3, false);
        }
    }

    /**
     * {@code out[r] += Σ_s weights[s] · (down_e[r] · x[s·H ..])}, {@code e = selected[s]}, over a
     * stacked {@code [experts·d, H]} down tensor: {@link #matrixVectorQ4_KSimdgroup}'s scheme per
     * slot, the slots summed in selection order before the SIMD group reduces. Example:
     * LFM2-8B-A1B, 2048 output rows over {@code H = 1792}. Launch as {@link
     * #matrixVectorQ4_KSimdgroup} over {@code d} rows.
     */
    public static void expertDownResidualQ4_KSimdgroup(
            KernelContext context,
            FloatArray x,
            FloatArray out,
            ByteArray down,
            IntArray selected,
            FloatArray weights,
            int d,
            int expertHidden,
            int used) {
        int lane = context.localIdx & 31;
        int row0 = (context.groupIdx * SIMDGROUPS + (context.localIdx >> 5)) * SIMDGROUP_ROWS;
        if (row0 >= d) {
            return;
        }
        int blocks = expertHidden / QK_K;
        int rowBytes = blocks * BLOCK_BYTES;
        int ix = lane >> 3;
        int iq = (lane & 7) >> 2;
        int ir = lane & 3;
        int shift = 16 * iq;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int s = 0; s < used; s++) {
            int wrow0 = selected.get(s) * d + row0;
            int xo = s * expertHidden;
            float weight = weights.get(s);
            float p0 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float p3 = 0.0f;
            for (int ib = ix; ib < blocks; ib += 4) {
                int xb = xo + ib * QK_K + 64 * iq + 8 * ir;
                Float4 xa0 = x.getFloat4(xb + 0);
                Float4 xa1 = x.getFloat4(xb + 4);
                Float4 xb0 = x.getFloat4(xb + 32);
                Float4 xb1 = x.getFloat4(xb + 36);
                Float4 xc0 = x.getFloat4(xb + 128);
                Float4 xc1 = x.getFloat4(xb + 132);
                Float4 xd0 = x.getFloat4(xb + 160);
                Float4 xd1 = x.getFloat4(xb + 164);
                float sa =
                        xa0.getX()
                                + xa0.getY()
                                + xa0.getZ()
                                + xa0.getW()
                                + xa1.getX()
                                + xa1.getY()
                                + xa1.getZ()
                                + xa1.getW();
                float sb =
                        xb0.getX()
                                + xb0.getY()
                                + xb0.getZ()
                                + xb0.getW()
                                + xb1.getX()
                                + xb1.getY()
                                + xb1.getZ()
                                + xb1.getW();
                float sc =
                        xc0.getX()
                                + xc0.getY()
                                + xc0.getZ()
                                + xc0.getW()
                                + xc1.getX()
                                + xc1.getY()
                                + xc1.getZ()
                                + xc1.getW();
                float sd =
                        xd0.getX()
                                + xd0.getY()
                                + xd0.getZ()
                                + xd0.getW()
                                + xd1.getX()
                                + xd1.getY()
                                + xd1.getZ()
                                + xd1.getW();
                float ya0 = xa0.getX();
                float ya1 = xa0.getY() * S1;
                float ya2 = xa0.getZ() * S2;
                float ya3 = xa0.getW() * S3;
                float ya4 = xa1.getX();
                float ya5 = xa1.getY() * S1;
                float ya6 = xa1.getZ() * S2;
                float ya7 = xa1.getW() * S3;
                float yb0 = xb0.getX();
                float yb1 = xb0.getY() * S1;
                float yb2 = xb0.getZ() * S2;
                float yb3 = xb0.getW() * S3;
                float yb4 = xb1.getX();
                float yb5 = xb1.getY() * S1;
                float yb6 = xb1.getZ() * S2;
                float yb7 = xb1.getW() * S3;
                float yc0 = xc0.getX();
                float yc1 = xc0.getY() * S1;
                float yc2 = xc0.getZ() * S2;
                float yc3 = xc0.getW() * S3;
                float yc4 = xc1.getX();
                float yc5 = xc1.getY() * S1;
                float yc6 = xc1.getZ() * S2;
                float yc7 = xc1.getW() * S3;
                float yd0 = xd0.getX();
                float yd1 = xd0.getY() * S1;
                float yd2 = xd0.getZ() * S2;
                float yd3 = xd0.getW() * S3;
                float yd4 = xd1.getX();
                float yd5 = xd1.getY() * S1;
                float yd6 = xd1.getZ() * S2;
                float yd7 = xd1.getW() * S3;
                for (int r = 0; r < SIMDGROUP_ROWS; r++) {
                    if (row0 + r < d) {
                        float part =
                                blockDot(
                                        down,
                                        (wrow0 + r) * rowBytes + ib * BLOCK_BYTES,
                                        shift,
                                        32 * iq + 8 * ir,
                                        ya0,
                                        ya1,
                                        ya2,
                                        ya3,
                                        ya4,
                                        ya5,
                                        ya6,
                                        ya7,
                                        yb0,
                                        yb1,
                                        yb2,
                                        yb3,
                                        yb4,
                                        yb5,
                                        yb6,
                                        yb7,
                                        yc0,
                                        yc1,
                                        yc2,
                                        yc3,
                                        yc4,
                                        yc5,
                                        yc6,
                                        yc7,
                                        yd0,
                                        yd1,
                                        yd2,
                                        yd3,
                                        yd4,
                                        yd5,
                                        yd6,
                                        yd7,
                                        sa,
                                        sb,
                                        sc,
                                        sd);
                        if (r == 0) {
                            p0 += part;
                        } else if (r == 1) {
                            p1 += part;
                        } else if (r == 2) {
                            p2 += part;
                        } else {
                            p3 += part;
                        }
                    }
                }
            }
            sum0 += weight * p0;
            sum1 += weight * p1;
            sum2 += weight * p2;
            sum3 += weight * p3;
        }
        sum0 = simdSum(context, sum0);
        sum1 = simdSum(context, sum1);
        sum2 = simdSum(context, sum2);
        sum3 = simdSum(context, sum3);
        if (lane == 0) {
            store(out, row0, d, sum0, true);
            store(out, row0 + 1, d, sum1, true);
            store(out, row0 + 2, d, sum2, true);
            store(out, row0 + 3, d, sum3, true);
        }
    }

    /** The sum over a SIMD group's 32 lanes, in lane 0. */
    private static float simdSum(KernelContext context, float v) {
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    /** Writes, or adds to, {@code out[row]} where the row exists. */
    private static void store(FloatArray out, int row, int d, float v, boolean residual) {
        if (row < d) {
            out.set(row, residual ? out.get(row) + v : v);
        }
    }
}
