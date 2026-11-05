package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Device kernels that read {@code IQ4_XS} weights in the file's own representation, shaped like the
 * {@code Q5_0} kernels beside them: same signatures, same workgroups, same reductions.
 *
 * <h2>The block</h2>
 *
 * <p>256 weights in 136 bytes, in eight groups of 32: {@code d} (fp16) at 0, the high two bits of
 * each group's 6-bit scale at 2 (group {@code g} at bit {@code 2g}), the low four bits at 4 (group
 * {@code g} in byte {@code g / 2}), then 16 bytes of nibbles per group at 8 (weights 0..15 in the
 * low nibbles, 16..31 in the high ones). Weight {@code j} of group {@code g} is {@code d * (ls -
 * 32) * kvalue(q)}. {@code IQ4_XSDecodeTest} holds the decode against {@code IQ4_XSFloatTensor} on
 * the same bytes.
 */
public final class TransformerComputeKernelsIQ4_XS {

    /** Weights per block. */
    private static final int QK = 256;

    /** Weights per group, one scale each. */
    private static final int GROUP = 32;

    /** Bytes per block: 2 (d) + 2 (high scale bits) + 4 (low scale bits) + 128 (nibbles). */
    private static final int BLOCK_BYTES = 136;

    /** Byte offset of the nibbles within a block. */
    private static final int QS_OFFSET = 8;

    /** Prompt rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ5_0}. */
    private static final int ROW_TILE = 8;

    /** Output rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ5_0}. */
    private static final int COL_TILE = 2;

    private TransformerComputeKernelsIQ4_XS() {}

    /** Output rows a tiled batch workgroup covers. */
    public static int colTile() {
        return COL_TILE;
    }

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    /**
     * ggml's {@code kvalues_iq4nl[q]}: {@code -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25,
     * 38, 53, 69, 89, 113}, held as four little-endian words of four signed bytes, so that a kernel
     * needs no table.
     */
    public static int kvalue(int q) {
        int word = q < 8 ? (q < 4 ? 0xBFAD9881 : 0xF6EADDCF) : (q < 12 ? 0x26190D01 : 0x71594535);
        return (word << (24 - ((q & 3) << 3))) >> 24;
    }

    /** The four bytes' {@code kvalue}s of the nibbles at bits 0..3 of each byte of {@code q4}. */
    private static int kvalues4(int q4) {
        return (kvalue(q4 & 0xF) & 0xFF)
                | ((kvalue((q4 >>> 8) & 0xF) & 0xFF) << 8)
                | ((kvalue((q4 >>> 16) & 0xF) & 0xFF) << 16)
                | ((kvalue((q4 >>> 24) & 0xF) & 0xFF) << 24);
    }

    /** Group {@code g}'s scale {@code d * (ls - 32)} in the block at {@code blockByteOffset}. */
    static float groupScale(ByteArray w, int blockByteOffset, int g) {
        int scalesH = w.getHalfFloat(blockByteOffset + 2).getHalfFloatValue() & 0xFFFF;
        int low = w.get(blockByteOffset + 4 + (g >> 1)) & 0xFF;
        int ls = ((low >> ((g & 1) << 2)) & 0xF) | (((scalesH >> (g << 1)) & 3) << 4);
        return w.getHalfFloat(blockByteOffset).getFloat32() * (ls - 32);
    }

    /**
     * One weight, decoded from its block.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..255
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        int g = withinBlock >> 5;
        int within = withinBlock & 31;
        int half = within >> 4;
        int packed = w.get(blockByteOffset + QS_OFFSET + (g << 4) + (within & 15)) & 0xFF;
        return groupScale(w, blockByteOffset, g) * kvalue((packed >> (half << 2)) & 0xF);
    }

    /**
     * One row's dot product against {@code x}, reduced through shared memory. A thread takes eight
     * weights per step: four consecutive bytes of one group, both nibbles. {@code n} is a multiple
     * of 256.
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
        int rowByteOffset = rowId * (n / QK) * BLOCK_BYTES;
        float partialSum = 0.0f;
        for (int unit = localId; unit < n / 8; unit += localSize) {
            int group = unit >> 2; // 32-weight groups across the row
            int j = (unit & 3) << 2;
            int block = group >> 3;
            int g = group & 7;
            int blockByteOffset = rowByteOffset + block * BLOCK_BYTES;
            float dl = groupScale(w, blockByteOffset, g);
            int qAt = blockByteOffset + QS_OFFSET + (g << 4) + j;
            int xAt = xOffset + group * GROUP + j;
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                int qByte = w.get(qAt + k) & 0xFF;
                sum += kvalue(qByte & 0xF) * x.get(xAt + k);
                sum += kvalue(qByte >> 4) * x.get(xAt + 16 + k);
            }
            partialSum += dl * sum;
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

    /**
     * One row's packed-integer dot product, reduced across the workgroup, the activation quantized
     * by {@link TransformerComputeKernelsQ4_0#quantizeActivationQ8Blocks} in blocks of 32: a group
     * contributes {@code xScale * d * (ls - 32) * dot(kvalue(q), xq)}. The table values are signed
     * bytes, so no recentring term. Lanes walk groups, warps reduce by shuffle; {@code
     * localWorkGroupSize} is a multiple of 32.
     */
    private static float rowDotDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        float[] warpSums = context.allocateFloatLocalArray(warpCount);
        int groupsPerRow = n / GROUP;
        int rowByteOffset = rowId * (n / QK) * BLOCK_BYTES;
        float partialSum = 0.0f;
        for (int group = localId; group < groupsPerRow; group += localWorkGroupSize) {
            int blockByteOffset = rowByteOffset + (group >> 3) * BLOCK_BYTES;
            int g = group & 7;
            float scale = groupScale(w, blockByteOffset, g);
            int quantBase = group * (GROUP / 4);
            int dot = 0;
            for (int q = 0; q < 4; q++) {
                int quantOffset = blockByteOffset + QS_OFFSET + (g << 4) + q * 4;
                int packed =
                        (w.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                dot =
                        QuantizationUtils.dp4a_packed(
                                kvalues4(packed & 0x0F0F0F0F), xQuants.get(quantBase + q), dot);
                dot =
                        QuantizationUtils.dp4a_packed(
                                kvalues4((packed >>> 4) & 0x0F0F0F0F),
                                xQuants.get(quantBase + 4 + q),
                                dot);
            }
            partialSum += scale * xScales.get(group) * dot;
        }
        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        if ((localId & 31) == 0) {
            warpSums[localId >> 5] = partialSum;
        }
        context.localBarrier();
        float total = 0.0f;
        for (int warp = 0; warp < warpCount; warp++) {
            total += warpSums[warp];
        }
        return total;
    }

    /**
     * {@code output[row] = w[row]·x} as packed integer dot products ({@link #rowDotDP4A}). {@code
     * xSums} is unused: the signature is the other types' one.
     */
    public static void matrixVectorGenericIQ4_XSDP4A(
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
        if (rowId >= d) {
            return;
        }
        float total = rowDotDP4A(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
        if (context.localIdx == 0) {
            output.set(rowId, total);
        }
    }

    /** {@code hb[row] += w[row]·x} as packed integer dot products ({@link #rowDotDP4A}). */
    public static void matrixVectorGenericWithResidualIQ4_XSDP4A(
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
        if (rowId >= d) {
            return;
        }
        float total = rowDotDP4A(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + total);
        }
    }

    /** {@code output[row] = w[row]·x}. */
    public static void matrixVectorGenericIQ4_XS(
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
        float sum = rowDotShared(context, localWorkGroupSize, x, 0, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /** {@code hb[row] += w[row]·x}. */
    public static void matrixVectorGenericWithResidualIQ4_XS(
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
        float sum = rowDotShared(context, localWorkGroupSize, x, 0, w, n, rowId);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + sum);
        }
    }

    /**
     * {@code out[b][row] = w[row]·x[b]} for a tile of up to {@link #ROW_TILE} prompt rows, one
     * workgroup per (row tile, output row pair): {@code TransformerComputeKernelsQ5_0}'s tiled
     * kernel with IQ4_XS's decode.
     */
    public static void matrixVectorTiledBatchIQ4_XS(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        tiledBatch(context, xBatch, outBatch, w, n, d, activeRows, localWorkGroupSize, false);
    }

    /**
     * {@code out[b][row] += w[row]·x[b]} for a tile of rows. See {@link
     * #matrixVectorTiledBatchIQ4_XS}.
     */
    public static void matrixVectorTiledBatchWithResidualIQ4_XS(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        tiledBatch(context, xBatch, outBatch, w, n, d, activeRows, localWorkGroupSize, true);
    }

    private static void tiledBatch(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize,
            boolean residual) {
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
        int blocksPerRow = n / QK;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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
                    int blockByteOffset = (outRow * blocksPerRow + blockIdx) * BLOCK_BYTES;
                    float weight = decode(w, blockByteOffset, withinBlock);
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
                        float value = localSums[slot * localWorkGroupSize];
                        if (residual) {
                            value += outBatch.get(index);
                        }
                        outBatch.set(index, value);
                    }
                }
            }
        }
    }

    /**
     * {@code IQ4_XS} weights decoded into FP16 at {@code destOffset} of {@code out}, for a GEMM
     * that reads FP16 operands, skipped once {@code done[flag]} is set, on {@code
     * TransformerBatchPrefillKernels.dequantizeQ8_0ToFP16Once}'s fixed striding grid. A per-chunk
     * caller passes a flag that is never set.
     */
    public static void dequantizeIQ4_XSToFP16Once(
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
                int block = i / QK;
                float value = decode(w, block * BLOCK_BYTES, i - block * QK);
                out.set(destOffset + i, new HalfFloat(value));
            }
        }
    }
}
