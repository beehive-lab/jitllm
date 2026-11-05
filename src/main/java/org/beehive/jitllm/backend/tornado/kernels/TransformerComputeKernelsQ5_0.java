package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Device kernels that read {@code Q5_0} weights in the file's own representation, shaped like the
 * {@code Q4_1} kernels beside them: same signatures, same workgroups, same reductions.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 22 bytes: {@code d} (fp16) at 0, {@code qh} (32 bits) at 2, then sixteen bytes
 * of packed nibbles at 6. Weight {@code j} is {@code d * (q - 16)}, with {@code q} its nibble (low
 * nibbles for 0..15, high ones for 16..31) and bit {@code j} of {@code qh} as its fifth bit. {@code
 * Q5_0DecodeTest} holds the decode against {@code Q5_0FloatTensor} on the same bytes.
 */
public final class TransformerComputeKernelsQ5_0 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 4 (fifth bits) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 22;

    /** Byte offset of the fifth bits within a block. */
    private static final int QH_OFFSET = 2;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 6;

    /** Prompt rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ4_0}. */
    private static final int ROW_TILE = 8;

    /** Output rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ4_0}. */
    private static final int COL_TILE = 2;

    /** Output rows a tiled batch workgroup covers. */
    public static int colTile() {
        return COL_TILE;
    }

    private TransformerComputeKernelsQ5_0() {}

    /**
     * One weight, decoded from its block.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int qh = highBits(w, blockByteOffset);
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int packed = w.get(blockByteOffset + QS_OFFSET + withinBlock - half * 16) & 0xFF;
        int q = ((packed >> (half * 4)) & 0xF) | (((qh >>> withinBlock) & 1) << 4);
        return d * (q - 16);
    }

    /** The block's 32 fifth bits, read as two 16-bit halves: a block is only 2-byte aligned. */
    static int highBits(ByteArray w, int blockByteOffset) {
        return (w.getHalfFloat(blockByteOffset + QH_OFFSET).getHalfFloatValue() & 0xFFFF)
                | ((w.getHalfFloat(blockByteOffset + QH_OFFSET + 2).getHalfFloatValue() & 0xFFFF)
                        << 16);
    }

    /** The fifth bits of weights {@code j} to {@code j + 3}, each at bit 4 of its byte. */
    private static int fifthBits(int qh, int j) {
        return (((qh >>> j) & 1) << 4)
                | (((qh >>> (j + 1)) & 1) << 12)
                | (((qh >>> (j + 2)) & 1) << 20)
                | (((qh >>> (j + 3)) & 1) << 28);
    }

    /**
     * The dot product of the eight weights in four consecutive bytes of one block with {@code x}:
     * bytes {@code j} to {@code j + 3} hold weights {@code j} to {@code j + 3} in their low nibbles
     * and weights {@code j + 16} to {@code j + 19} in their high ones. {@code xAt} is where weight
     * {@code j}'s activation is.
     */
    private static float dot8(ByteArray w, int blockByteOffset, int j, FloatArray x, int xAt) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int qh = highBits(w, blockByteOffset);
        int qAt = blockByteOffset + QS_OFFSET + j;
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            int qByte = w.get(qAt + k) & 0xFF;
            int lo = (qByte & 0xF) | (((qh >>> (j + k)) & 1) << 4);
            int hi = ((qByte >> 4) & 0xF) | (((qh >>> (j + k + 16)) & 1) << 4);
            sum += (d * (lo - 16)) * x.get(xAt + k);
            sum += (d * (hi - 16)) * x.get(xAt + 16 + k);
        }
        return sum;
    }

    /**
     * One row's dot product against {@code x}, reduced through shared memory. A thread takes eight
     * weights per step ({@link #dot8}); {@code n} is a multiple of 32.
     */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        return rowDotShared(context, localSize, x, 0, w, n, rowId);
    }

    /**
     * The same reduction over a row of a <b>batch</b> of activations.
     *
     * <p>{@code xOffset} is where this row's activation starts. Everything else — the block
     * addressing, the decode, the reduction — is the single-token path's, so a batched projection
     * is the same arithmetic in the same order over a different input offset.
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

        // Eight weights per step through dot8: four consecutive bytes, both nibbles.
        float partialSum = 0.0f;
        for (int unit = localId; unit < n / 8; unit += localSize) {
            int blockIdx = unit / 4;
            int j = (unit - blockIdx * 4) * 4;
            partialSum +=
                    dot8(
                            w,
                            rowByteOffset + blockIdx * BLOCK_BYTES,
                            j,
                            x,
                            xOffset + blockIdx * QK + j);
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
     * by {@link TransformerComputeKernelsQ4_0#quantizeActivationQ8Blocks}: a block contributes
     * {@code xScale * d * (dot(q, xq) - 16 * sum(xq))} with {@code q} the unsigned 5-bit quants, as
     * the Q4_0 kernels recentre theirs by 8. Lanes walk blocks, warps reduce by shuffle; {@code
     * localWorkGroupSize} is a multiple of 32.
     */
    private static float rowDotDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            IntArray xSums,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        float[] warpSums = context.allocateFloatLocalArray(warpCount);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;
        float partialSum = 0.0f;
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float scale = w.getHalfFloat(blockByteOffset).getFloat32();
            int qh = highBits(w, blockByteOffset);
            int quantBase = block * (QK / 4);
            int dot = 0;
            for (int g = 0; g < 4; g++) {
                int quantOffset = blockByteOffset + QS_OFFSET + g * 4;
                int packed =
                        (w.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                dot =
                        QuantizationUtils.dp4a_packed(
                                (packed & 0x0F0F0F0F) | fifthBits(qh, g * 4),
                                xQuants.get(quantBase + g),
                                dot);
                dot =
                        QuantizationUtils.dp4a_packed(
                                ((packed >>> 4) & 0x0F0F0F0F) | fifthBits(qh, 16 + g * 4),
                                xQuants.get(quantBase + 4 + g),
                                dot);
            }
            partialSum += scale * xScales.get(block) * (dot - 16 * xSums.get(block));
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

    /** {@code output[row] = w[row]·x} as packed integer dot products ({@link #rowDotDP4A}). */
    public static void matrixVectorGenericQ5_0DP4A(
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
        float total = rowDotDP4A(context, xQuants, xScales, xSums, w, n, rowId, localWorkGroupSize);
        if (context.localIdx == 0) {
            output.set(rowId, total);
        }
    }

    /** {@code hb[row] += w[row]·x} as packed integer dot products ({@link #rowDotDP4A}). */
    public static void matrixVectorGenericWithResidualQ5_0DP4A(
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
        float total = rowDotDP4A(context, xQuants, xScales, xSums, w, n, rowId, localWorkGroupSize);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + total);
        }
    }

    /** {@code output[row] = w[row]·x}. */
    public static void matrixVectorGenericQ5_0(
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

    /** {@code hb[row] += w[row]·x}. */
    public static void matrixVectorGenericWithResidualQ5_0(
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

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a tile of up to {@link #ROW_TILE} prompt rows, one
     * workgroup per (row tile, output row).
     *
     * <p>Q4_0's tiled kernel with Q5_0's decode. The untiled batch kernel beside this one reads the
     * weight row once per prompt row, which is what running the rows separately reads; this one
     * reads and decodes each weight once for the whole tile.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ5_0(
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
        int blocksPerRow = (n + QK - 1) / QK;

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
                        outBatch.set(row * d + outRow, localSums[slot * localWorkGroupSize]);
                    }
                }
            }
        }
    }

    /**
     * {@code out[b][row] += w[row]·x[b]} for a tile of rows. See {@link
     * #matrixVectorTiledBatchQ5_0}.
     */
    public static void matrixVectorTiledBatchWithResidualQ5_0(
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
        int blocksPerRow = (n + QK - 1) / QK;

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
                        outBatch.set(
                                index, outBatch.get(index) + localSums[slot * localWorkGroupSize]);
                    }
                }
            }
        }
    }

    /**
     * {@code Q5_0} weights decoded into FP16 at {@code destOffset} of {@code out}, for a GEMM that
     * reads FP16 operands, skipped once {@code done[flag]} is set, on {@code
     * TransformerBatchPrefillKernels.dequantizeQ8_0ToFP16Once}'s fixed striding grid. A per-chunk
     * caller passes a flag that is never set.
     */
    public static void dequantizeQ5_0ToFP16Once(
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
                // decode(...) copied in, as in the Q4_1 decoder.
                int block = i / QK;
                int blockByteOffset = block * BLOCK_BYTES;
                int withinBlock = i - block * QK;
                float d = w.getHalfFloat(blockByteOffset).getFloat32();
                int qh =
                        (w.getHalfFloat(blockByteOffset + QH_OFFSET).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(blockByteOffset + QH_OFFSET + 2)
                                                        .getHalfFloatValue()
                                                & 0xFFFF)
                                        << 16);
                int high = withinBlock / 16;
                int packed = w.get(blockByteOffset + QS_OFFSET + withinBlock - high * 16) & 0xFF;
                int q = ((packed >> (high * 4)) & 0xF) | (((qh >>> withinBlock) & 1) << 4);
                out.set(destOffset + i, new HalfFloat(d * (q - 16)));
            }
        }
    }
}
