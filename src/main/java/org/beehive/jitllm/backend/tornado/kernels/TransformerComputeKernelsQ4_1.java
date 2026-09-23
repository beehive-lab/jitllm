package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Device kernels that read {@code Q4_1} weights in the file's own representation.
 *
 * <p>Shaped like the {@code Q4_0}, {@code Q4_K} and {@code Q6_K} kernels beside them — same
 * signatures, same workgroup-per-row structure, same reductions — so a layer differs only in which
 * method reference it names. What differs is the decode.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 20 bytes: {@code d} (fp16) at 0, {@code m} (fp16) at 2, then sixteen bytes of
 * packed nibbles. A weight is {@code d * q + m} with an <b>unsigned</b> nibble.
 *
 * <p>That is the whole difference from Q4_0, which is {@code d * (q - 8)} — one recentring and one
 * extra half of header. Applying Q4_0's arithmetic to a Q4_1 block, or Q4_1's offsets to a Q4_0
 * one, gives weights of entirely plausible magnitude, so neither mistake announces itself. {@code
 * Q4_1DecodeTest} holds this against {@code Q4_1FloatTensor} on the same bytes.
 */
public final class TransformerComputeKernelsQ4_1 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 2 (m) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 20;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 4;

    /** Prompt rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ4_0}. */
    private static final int ROW_TILE = 8;

    /** Output rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ4_0}. */
    private static final int COL_TILE = 2;

    /** Output rows a tiled batch workgroup covers. */
    public static int colTile() {
        return COL_TILE;
    }

    private TransformerComputeKernelsQ4_1() {}

    /**
     * One weight, decoded from its block.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float m = w.getHalfFloat(blockByteOffset + 2).getFloat32();
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * q + m;
    }

    /** One row's dot product against {@code x}, reduced across a 32-lane subgroup. */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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

    /** One row's dot product against {@code x}, reduced through shared memory. */
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
    private static float rowDotShared(
            KernelContext context,
            int localSize,
            FloatArray x,
            int xOffset,
            ByteArray w,
            int n,
            int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(xOffset + j);
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
     * {@code out[row] += w[row]·x} with the activation in packed integers and the weights left in
     * native {@code Q4_1}.
     *
     * <p><b>The algebra, which is where this differs from Q4_0.</b> A Q4_1 weight is {@code d * q +
     * m} with {@code q} in 0..15 and no recentring, so against an activation quantized as {@code
     * x_i = sx * xq_i} a block's contribution is
     *
     * <pre>
     *   Σ (d·q_i + m)(sx·xq_i) = sx · ( d · Σ q_i·xq_i  +  m · Σ xq_i )
     * </pre>
     *
     * The first sum is the {@code dp4a} accumulation; the second is the activation block's sum of
     * quants, which {@code quantizeActivationQ8Blocks} already records. So where {@link
     * TransformerComputeKernelsQ4_0#matrixVectorGenericWithResidualQ4_0DP4A} applies {@code dot - 8
     * * sum} for Q4_0's recentring by eight, this applies {@code d * dot + m * sum}. It is the same
     * decomposition llama.cpp's {@code vec_dot_q4_1_q8_1_impl} uses: {@code sumi*d4*d8 + m4*s8}.
     *
     * <p>A block whose activation was all zero has {@code sx == 0} and {@code Σ xq_i == 0}, so it
     * contributes exactly nothing — the minimum term does not leak into an empty block.
     *
     * <p>The nibble layout is Q4_0's, one header wider: low nibbles are elements 0..15 and high
     * nibbles 16..31, which is why the low half pairs with quant group {@code g} and the high half
     * with {@code g + 4}. The reduction is the warp-shuffle butterfly the packed kernels share, so
     * the summation order differs from the floating-point kernel's and the two agree to rounding.
     *
     * <p><b>This changes the numerics</b>: the activation is quantized to int8 where the
     * floating-point kernel reads it exactly. The weights are untouched and stay Q4_1.
     */
    // @formatter:on
    public static void matrixVectorGenericWithResidualQ4_1DP4A(
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
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        float[] warpSums = context.allocateFloatLocalArray(warpCount);

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float weightScale = w.getHalfFloat(blockByteOffset).getFloat32();
            float weightMin = w.getHalfFloat(blockByteOffset + 2).getFloat32();
            int quantBase = block * (QK / 4);

            int dot = 0;
            for (int g = 0; g < 4; g++) {
                int b0 = w.get(blockByteOffset + QS_OFFSET + g * 4) & 0xFF;
                int b1 = w.get(blockByteOffset + QS_OFFSET + g * 4 + 1) & 0xFF;
                int b2 = w.get(blockByteOffset + QS_OFFSET + g * 4 + 2) & 0xFF;
                int b3 = w.get(blockByteOffset + QS_OFFSET + g * 4 + 3) & 0xFF;
                dot =
                        QuantizationUtils.dp4a_packed(
                                (b0 & 0xF)
                                        | ((b1 & 0xF) << 8)
                                        | ((b2 & 0xF) << 16)
                                        | ((b3 & 0xF) << 24),
                                xQuants.get(quantBase + g),
                                dot);
                dot =
                        QuantizationUtils.dp4a_packed(
                                ((b0 >> 4) & 0xF)
                                        | (((b1 >> 4) & 0xF) << 8)
                                        | (((b2 >> 4) & 0xF) << 16)
                                        | (((b3 >> 4) & 0xF) << 24),
                                xQuants.get(quantBase + 4 + g),
                                dot);
            }
            partialSum += xScales.get(block) * (weightScale * dot + weightMin * xSums.get(block));
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

        if (localId == 0) {
            float total = 0.0f;
            for (int warp = 0; warp < warpCount; warp++) {
                total += warpSums[warp];
            }
            hb.set(rowId, hb.get(rowId) + total);
        }
    }

    /** {@code output[row] = w[row]·x}. */
    public static void matrixVectorGenericQ4_1(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_1}. */
    public static void matrixVectorGenericQ4_1Simd32(
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

    /** {@code hb[row] += w[row]·x}. */
    public static void matrixVectorGenericWithResidualQ4_1(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ4_1}. */
    public static void matrixVectorGenericWithResidualQ4_1Simd32(
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

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a tile of up to {@link #ROW_TILE} prompt rows, one
     * workgroup per (row tile, output row).
     *
     * <p>Q4_0's tiled kernel with Q4_1's decode. The untiled batch kernel beside this one reads the
     * weight row once per prompt row, which is what running the rows separately reads; this one
     * reads and decodes each weight once for the whole tile.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ4_1(
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
     * #matrixVectorTiledBatchQ4_1}.
     */
    public static void matrixVectorTiledBatchWithResidualQ4_1(
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
}
