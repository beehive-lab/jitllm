package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Device kernels that read {@code Q4_0} weights <b>in the file's own representation</b>.
 *
 * <p>The Q4_K family's siblings, and shaped identically to them on purpose — same signatures, same
 * workgroup-per-row structure, same reductions — so a layer differs only in which method reference
 * it names. What differs is the decode, and Q4_0's is the simplest of the quantizations that reach
 * the device.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 18 bytes: {@code d} (fp16) at 0, then sixteen bytes of packed nibbles. A weight
 * is {@code d * (q - 8)} — an <b>unsigned</b> nibble recentred by eight, with a single scale and no
 * minimum, which is what separates it from {@code Q4_1} and from the K-quants' per-sub-block
 * scales. Element {@code i} below 16 is the low nibble of byte {@code i}; element {@code i} at 16
 * or above is the high nibble of byte {@code i - 16}.
 *
 * <h2>Reading the packed nibbles</h2>
 *
 * <p>The packed kernels below read each four-byte group of nibbles as two sixteen-bit words via
 * {@code ByteArray.getHalfFloat(o).getHalfFloatValue()}. That accessor is used purely as a
 * bit-preserving load — these bytes are quants, never a number — and each half is masked to sixteen
 * bits before it is shifted, so the signed short cannot sign-extend into the packed word. A block
 * is 18 bytes with its quants at offset 2, so every such offset is even, which is all a pair read
 * needs; only every other quant run is four-byte aligned, which is why these are pairs and not
 * single words.
 *
 * <h2>Why it exists</h2>
 *
 * <p>Q4_0 used to be materialized as Q8_0 at load, which roughly doubles a model's device
 * footprint: 4.5 bits per weight become 8.5. That is the same problem retaining Q4_K solved for
 * Devstral, and it is what puts a Q4_0 file's own size — rather than twice it — against the
 * device's memory.
 *
 * <p>The unpacking below is the same arithmetic as the host's {@code Q4_0FloatTensor}, which is the
 * reference it was written against. {@code Q4_0DecodeTest} holds the two against each other on the
 * same bytes rather than trusting the restatement.
 */
public final class TransformerComputeKernelsQ4_0 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 18;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 2;

    private TransformerComputeKernelsQ4_0() {}

    /**
     * One weight, decoded from its block.
     *
     * <p>Package-private, like its Q4_K counterpart, so the decode test can hold it against the
     * host tensor directly on the same bytes. TornadoVM inlines it.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * (q - 8);
    }

    /**
     * One row's dot product against {@code x}, reduced across a 32-lane subgroup.
     *
     * <p>Used where {@code DeviceCapability.SUBGROUP_SHUFFLE_32} holds.
     */
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

    // ── the packed-integer path ──────────────────────────────────────────────

    // @formatter:off
    /**
     * Quantizes one token's activations into {@code Q8}-style blocks, for the packed-integer
     * matrix-vector below.
     *
     * <p>One workgroup of 32 lanes per block, one lane per activation: the block's maximum and the
     * sum of its quants are shared reductions, and eight lanes do the packing. It writes three
     * things, and the third is the one that is easy to miss: the <b>sum</b> of the block's
     * quantized activations. Q4_0 stores an unsigned nibble meaning {@code q - 8}, and
     *
     * <pre>  sum (q_w - 8) * q_x  =  sum q_w * q_x  -  8 * sum q_x</pre>
     *
     * so the recentring comes out of the inner loop entirely and the dot product runs on the raw
     * nibbles, which are 0..15 and therefore already valid signed bytes. Nothing in the hot loop
     * has to build a negative byte — which also keeps it clear of the unsigned-recentring defect in
     * the TornadoVM backend.
     *
     * <p>A block of exact zeros gets a zero scale and zero quants rather than a division by zero;
     * its contribution is then zero, which is what it should be.
     *
     * @param x the token's activations
     * @param quants four quantized activations per int, in element order
     * @param scales one per block
     * @param sums the sum of each block's quantized activations
     */
    // @formatter:on
    public static void quantizeActivationQ8Blocks(
            KernelContext context,
            FloatArray x,
            IntArray quants,
            FloatArray scales,
            IntArray sums) {
        int block = context.groupIdx;
        int lane = context.localIdx;
        int base = block * QK;

        float[] shared = context.allocateFloatLocalArray(QK);
        int[] sharedQuants = context.allocateIntLocalArray(QK);

        float value = x.get(base + lane);
        shared[lane] = TornadoMath.abs(value);
        context.localBarrier();
        for (int stride = QK / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = TornadoMath.max(shared[lane], shared[lane + stride]);
            }
            context.localBarrier();
        }
        float maxAbs = shared[0];
        context.localBarrier();

        float inverse = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
        float scaled = value * inverse;
        int q = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        q = TornadoMath.min(127, TornadoMath.max(-127, q));
        sharedQuants[lane] = q;
        shared[lane] = q;
        context.localBarrier();

        // The four quants of one group live in four neighbouring lanes, so packing is eight lanes
        // reading four entries each rather than one thread walking the block.
        if (lane < QK / 4) {
            int packed =
                    (sharedQuants[lane * 4] & 0xFF)
                            | ((sharedQuants[lane * 4 + 1] & 0xFF) << 8)
                            | ((sharedQuants[lane * 4 + 2] & 0xFF) << 16)
                            | ((sharedQuants[lane * 4 + 3] & 0xFF) << 24);
            quants.set(block * (QK / 4) + lane, packed);
        }

        for (int stride = QK / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] += shared[lane + stride];
            }
            context.localBarrier();
        }
        if (lane == 0) {
            scales.set(block, maxAbs / 127.0f);
            sums.set(block, (int) shared[0]);
        }
    }

    // @formatter:off
    /**
     * The RMS-norm apply and the block quantization above it, in one kernel.
     *
     * <p>{@code TransformerComputeKernelsLayered.reductionOneBlock2WithLayer} followed by {@link
     * #quantizeActivationQ8Blocks} over the same activation, which is what every decode layer
     * dispatches twice — once for the attention norm's output and once for the feed-forward's. The
     * apply is <b>elementwise</b> despite its name ({@code xb[i] = weight[i] * (ss * x[i])}, with
     * {@code ss} the scale the reduce already wrote), and the quantization's workgroup owns exactly
     * the 32 consecutive elements its own lanes would have normalized. So each lane computes its
     * element instead of reading it back, and <b>no synchronization crosses a workgroup</b> — the
     * reduce that does span the row stays the separate task it was.
     *
     * <p>{@code xb} is still written. It is not dead: the F32 {@code ssm_alpha} and {@code
     * ssm_beta} projections read it directly, as does every non-packed projection. What the fusion
     * removes is the second launch and the read-back of {@code xb}, not the store.
     *
     * <p>Same arithmetic, in the same order, so the result is bit-identical: {@code weight * (ss *
     * x)} is a product of products with no addition, so there is no fused multiply-add for the
     * compiler to contract differently, and the value a lane computes is the one the separate apply
     * would have stored and the quantization read back.
     *
     * @param xb the normalized activation, still written for its other readers
     * @param x the residual stream
     * @param rmsWeights the norm's per-element weights
     * @param temp the reduce's output; {@code temp[0]} is the scale
     * @param quants four quantized activations per int, in element order
     * @param scales one per block
     * @param sums the sum of each block's quantized activations
     */
    // @formatter:on
    public static void rmsApplyAndQuantizeActivationQ8Blocks(
            KernelContext context,
            FloatArray xb,
            FloatArray x,
            FloatArray rmsWeights,
            FloatArray temp,
            IntArray quants,
            FloatArray scales,
            IntArray sums) {
        int block = context.groupIdx;
        int lane = context.localIdx;
        int base = block * QK;
        int index = base + lane;

        float[] shared = context.allocateFloatLocalArray(QK);
        int[] sharedQuants = context.allocateIntLocalArray(QK);

        float ss = temp.get(0);
        float value = rmsWeights.get(index) * (ss * x.get(index));
        xb.set(index, value);

        shared[lane] = TornadoMath.abs(value);
        context.localBarrier();
        for (int stride = QK / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = TornadoMath.max(shared[lane], shared[lane + stride]);
            }
            context.localBarrier();
        }
        float maxAbs = shared[0];
        context.localBarrier();

        float inverse = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
        float scaled = value * inverse;
        int q = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        q = TornadoMath.min(127, TornadoMath.max(-127, q));
        sharedQuants[lane] = q;
        shared[lane] = q;
        context.localBarrier();

        if (lane < QK / 4) {
            int packed =
                    (sharedQuants[lane * 4] & 0xFF)
                            | ((sharedQuants[lane * 4 + 1] & 0xFF) << 8)
                            | ((sharedQuants[lane * 4 + 2] & 0xFF) << 16)
                            | ((sharedQuants[lane * 4 + 3] & 0xFF) << 24);
            quants.set(block * (QK / 4) + lane, packed);
        }

        for (int stride = QK / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] += shared[lane + stride];
            }
            context.localBarrier();
        }
        if (lane == 0) {
            scales.set(block, maxAbs / 127.0f);
            sums.set(block, (int) shared[0]);
        }
    }

    // @formatter:off
    /**
     * {@code output[row] = w[row] · x} with the weights read as {@code Q4_0} and the dot product
     * done in packed integers.
     *
     * <p>Same shape as {@link #matrixVectorGenericQ4_0} — one workgroup per output row, a shared
     * reduction — and a different inner loop. That one walks <b>elements</b>, decoding a weight and
     * re-reading its block scale for each; this one walks <b>blocks</b>, reading the scale once per
     * 32 weights and issuing eight {@code dp4a} instructions.
     *
     * <p>Four consecutive weights come from four consecutive bytes' <b>low</b> nibbles, and the
     * high nibbles of those same bytes are the four weights sixteen positions later. So one group
     * of four bytes feeds two dot products, against activation groups {@code 4g} and {@code 16+4g}
     * — which is the pairing the block layout forces and the reason the loop is written in groups
     * of four rather than in halves.
     *
     * <p>The activations must have been prepared by {@link #quantizeActivationQ8Blocks}; the {@code
     * -8 * sum} correction it precomputes is applied once per block here.
     *
     * <p>The reduction is a warp-shuffle butterfly, not a shared-memory tree: each 32-lane warp
     * reduces its own partial with five {@code simdShuffleDown} steps, one lane per warp writes to
     * shared memory, and a single barrier separates that from the combine — one barrier where the
     * tree needed seven at the 128-lane width this repository dispatches. All lanes reach the
     * shuffles; the only early return is on {@code rowId}, which is uniform across the workgroup,
     * and {@code localWorkGroupSize} must be a multiple of 32. Shuffles are correct on CUDA and
     * miscompile on OpenCL, which is why every packed kernel rides on {@code
     * DeviceCapability.PACKED_INTEGER_DOT}, granted on CUDA alone. The order of summation differs
     * from the tree's, so the floating-point total may round differently; the integer dot products
     * are exact either way.
     */
    // @formatter:on
    public static void matrixVectorGenericQ4_0DP4A(
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
        int localId = context.localIdx;
        int warpCount = localWorkGroupSize / 32;
        float[] warpSums = context.allocateFloatLocalArray(warpCount);

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float weightScale = w.getHalfFloat(blockByteOffset).getFloat32();
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
                                packed & 0x0F0F0F0F, xQuants.get(quantBase + g), dot);
                dot =
                        QuantizationUtils.dp4a_packed(
                                (packed >>> 4) & 0x0F0F0F0F, xQuants.get(quantBase + 4 + g), dot);
            }
            partialSum += weightScale * xScales.get(block) * (dot - 8 * xSums.get(block));
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
            output.set(rowId, total);
        }
    }

    /** {@code output[row] = w[row]·x}. Q4_0 counterpart of {@code matrixVectorGenericQ8Byte}. */
    public static void matrixVectorGenericQ4_0(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_0}. */
    public static void matrixVectorGenericQ4_0Simd32(
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
     * {@code hb[row] += w[row]·x}. Q4_0 counterpart of {@code
     * matrixVectorGenericWithResidualQ8_0Byte}.
     */
    public static void matrixVectorGenericWithResidualQ4_0(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ4_0}. */
    public static void matrixVectorGenericWithResidualQ4_0Simd32(
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
     * Fused query/key/value projection, one workgroup per output row.
     *
     * <p>Stated in terms of {@code qDim} and {@code kvDim} rather than assuming the query width is
     * {@code dim}, so it serves a family whose head dimension is independent of {@code dim / heads}
     * as well as one where they agree. The row's identity selects which projection it belongs to.
     */
    public static void fusedQKVMatmulQ4_0(
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
     * Fused feed-forward gate/up projection with SwiGLU, over an already-normalized activation.
     *
     * <p>The normalization is dtype-independent and its existing task is reused rather than folded
     * in here, which is the same trade the Q4_K path makes: one more task per layer, one fewer
     * kernel to keep correct.
     */
    public static void fusedFFNGateUpSiLUQ4_0(
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
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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

    /**
     * {@link #fusedFFNGateUpSiLUQ4_0} with GeGLU in place of SwiGLU.
     *
     * <p>A separate method rather than a flag: the activation is the only difference, and the two
     * families that need them do not otherwise share a feed-forward. Everything above the final
     * combine — the single local array holding both halves, the one tree that reduces both, the
     * block addressing — is that kernel's and is deliberately identical, so a change to the
     * addressing has one place to be made and two places to be checked.
     */
    public static void fusedFFNGateUpGeGLUQ4_0(
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

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * 2);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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
            hb.set(
                    rowId,
                    TransformerComputeKernelsLayered.geluActivation(localSums[0])
                            * localSums[localWorkGroupSize]);
        }
    }

    // @formatter:off
    /**
     * {@code hb[row] += w[row]·x} in packed integers — the residual form, not yet dispatched.
     *
     * <p>{@link #matrixVectorGenericQ4_0DP4A} with the accumulate-into-destination the residual
     * projections need. Whether it is worth dispatching is a question about preparation cost rather
     * than about this loop: a residual projection's activation feeds one projection, not two or
     * three, and {@code ffn_down}'s is the feed-forward's hidden width rather than the embedding
     * width, so the quantization it needs is larger and amortized over less.
     */
    // @formatter:on
    public static void matrixVectorGenericWithResidualQ4_0DP4A(
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
                                packed & 0x0F0F0F0F, xQuants.get(quantBase + g), dot);
                dot =
                        QuantizationUtils.dp4a_packed(
                                (packed >>> 4) & 0x0F0F0F0F, xQuants.get(quantBase + 4 + g), dot);
            }
            partialSum += weightScale * xScales.get(block) * (dot - 8 * xSums.get(block));
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

    // @formatter:off
    /**
     * {@code hb[row] = silu(w1[row]·x) * (w3[row]·x)} with both projections in packed integers.
     *
     * <p>The packed counterpart of {@link #fusedFFNGateUpSiLUQ4_0}, and the same arithmetic around
     * it: one pass, one local array with gate in the first half and up in the second, one tree
     * reducing both, and the SiLU applied to the reduced gate before it multiplies the reduced up.
     * What changes is the inner loop, which walks blocks rather than elements — one block scale per
     * 32 weights instead of one per weight, and eight {@code dp4a} instructions per block per
     * projection.
     *
     * <p>Both projections read the <b>same</b> quantized activation, so the {@code -8 * sum}
     * correction and the activation scale are shared between them; only the weight scale and the
     * integer dot differ. That is what makes the fused form worth keeping in the packed path: the
     * activation is quantized once and consumed twice, exactly as the two weight matrices are read
     * against one staged activation in the floating-point form.
     *
     * <p>The activation must have been prepared by {@link #quantizeActivationQ8Blocks} from the
     * activation this projection actually reads — the feed-forward norm's output, not the attention
     * branch's.
     */
    // @formatter:on
    public static void fusedFFNGateUpSiLUQ4_0DP4A(
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
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }

        int warpCount = localWorkGroupSize / 32;
        // Gate in the first warpCount entries, up in the second — the halves-of-one-array layout
        // the tree used, at a thirty-second of the size.
        float[] warpSums = context.allocateFloatLocalArray(warpCount * 2);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float gateScale = w1.getHalfFloat(blockByteOffset).getFloat32();
            float upScale = w3.getHalfFloat(blockByteOffset).getFloat32();
            int quantBase = block * (QK / 4);
            float activationScale = xScales.get(block);
            int correction = 8 * xSums.get(block);

            int gateDot = 0;
            int upDot = 0;
            for (int g = 0; g < 4; g++) {
                // Paired nibble read; see the class comment for why it is a pair and not a word.
                int quantOffset = blockByteOffset + QS_OFFSET + g * 4;
                int gateWord =
                        (w1.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w1.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int upWord =
                        (w3.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w3.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int low = xQuants.get(quantBase + g);
                int high = xQuants.get(quantBase + 4 + g);
                // The same four operands the byte-wise packing produced: the low nibble of each of
                // the four bytes, then the high nibble of each, in the same DP4A pairing and order.
                gateDot = QuantizationUtils.dp4a_packed(gateWord & 0x0F0F0F0F, low, gateDot);
                gateDot =
                        QuantizationUtils.dp4a_packed((gateWord >>> 4) & 0x0F0F0F0F, high, gateDot);
                upDot = QuantizationUtils.dp4a_packed(upWord & 0x0F0F0F0F, low, upDot);
                upDot = QuantizationUtils.dp4a_packed((upWord >>> 4) & 0x0F0F0F0F, high, upDot);
            }
            gate += gateScale * activationScale * (gateDot - correction);
            up += upScale * activationScale * (upDot - correction);
        }

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
            // SiLU is applied to the gate reduced across every warp, never per warp: it is not
            // linear, so the order is the function.
            float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
            hb.set(rowId, silu * upSum);
        }
    }

    /**
     * {@link #fusedFFNGateUpSiLUQ4_0DP4A} with GeGLU in place of SwiGLU — the packed-integer
     * counterpart of {@link #fusedFFNGateUpGeGLUQ4_0}. Everything above the final combine is that
     * kernel's and is deliberately identical.
     */
    public static void fusedFFNGateUpGeGLUQ4_0DP4A(
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
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }

        int warpCount = localWorkGroupSize / 32;
        // Gate in the first warpCount entries, up in the second — the halves-of-one-array layout
        // the tree used, at a thirty-second of the size.
        float[] warpSums = context.allocateFloatLocalArray(warpCount * 2);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
            float gateScale = w1.getHalfFloat(blockByteOffset).getFloat32();
            float upScale = w3.getHalfFloat(blockByteOffset).getFloat32();
            int quantBase = block * (QK / 4);
            float activationScale = xScales.get(block);
            int correction = 8 * xSums.get(block);

            int gateDot = 0;
            int upDot = 0;
            for (int g = 0; g < 4; g++) {
                // Paired nibble read; see the class comment for why it is a pair and not a word.
                int quantOffset = blockByteOffset + QS_OFFSET + g * 4;
                int gateWord =
                        (w1.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w1.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int upWord =
                        (w3.getHalfFloat(quantOffset).getHalfFloatValue() & 0xFFFF)
                                | ((w3.getHalfFloat(quantOffset + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int low = xQuants.get(quantBase + g);
                int high = xQuants.get(quantBase + 4 + g);
                // The same four operands the byte-wise packing produced: the low nibble of each of
                // the four bytes, then the high nibble of each, in the same DP4A pairing and order.
                gateDot = QuantizationUtils.dp4a_packed(gateWord & 0x0F0F0F0F, low, gateDot);
                gateDot =
                        QuantizationUtils.dp4a_packed((gateWord >>> 4) & 0x0F0F0F0F, high, gateDot);
                upDot = QuantizationUtils.dp4a_packed(upWord & 0x0F0F0F0F, low, upDot);
                upDot = QuantizationUtils.dp4a_packed((upWord >>> 4) & 0x0F0F0F0F, high, upDot);
            }
            gate += gateScale * activationScale * (gateDot - correction);
            up += upScale * activationScale * (upDot - correction);
        }

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
            // GELU is applied to the gate reduced across every warp, never per warp: it is not
            // linear, so the order is the function.
            hb.set(rowId, TransformerComputeKernelsLayered.geluActivation(gateSum) * upSum);
        }
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} over a chunk of activations, one workgroup per (row, output
     * row).
     *
     * <p>The kernel launches a fixed number of rows and is told how many are active: a padding row
     * returns before reading anything, so a chunk shorter than the batch width costs launches and
     * nothing else.
     */
    // @formatter:on
    public static void matrixVectorBatchQ4_0(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int batchIdx = groupId / d;
        int rowId = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, xBatch, batchIdx * n, w, n, rowId);
        if (context.localIdx == 0) {
            outBatch.set(batchIdx * d + rowId, sum);
        }
    }

    /** {@code out[b][row] += w[row]·x[b]}, the residual form. */
    public static void matrixVectorBatchWithResidualQ4_0(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int batchIdx = groupId / d;
        int rowId = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, xBatch, batchIdx * n, w, n, rowId);
        if (context.localIdx == 0) {
            int index = batchIdx * d + rowId;
            outBatch.set(index, outBatch.get(index) + sum);
        }
    }

    // @formatter:off
    /**
     * The fused gate/up feed-forward with SwiGLU over a chunk of activations.
     *
     * <p>One workgroup per (row, hidden row), and the same single pass over one local array the
     * single-token kernel makes: gate in the first half, up in the second, one reduction tree for
     * both.
     */
    // @formatter:on
    public static void fusedFFNGateUpSiLUBatchQ4_0(
            KernelContext context,
            FloatArray xBatch,
            FloatArray hbBatch,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / d;
        int rowId = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * 2);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;
        int inputOffset = batchIdx * n;

        float gate = 0.0f;
        float up = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            float activation = xBatch.get(inputOffset + j);
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
            hbBatch.set(batchIdx * d + rowId, silu * localSums[localWorkGroupSize]);
        }
    }

    /** Rows a tiled batched projection decodes each weight for. */
    private static final int ROW_TILE = 8;

    // @formatter:off
    /**
     * Output rows a tiled batched projection covers per workgroup.
     *
     * <p>The row tile answers "how many times is a weight decoded"; this one answers "how many
     * times is an activation read". They are not the same question, and for this family the second
     * is the larger number: a workgroup covering one output row of {@code ffn_down} reads 8,704
     * bytes of weights and 557 KB of activations. Covering four output rows quarters the activation
     * traffic per output row and leaves the weight traffic where it was.
     *
     * <p>Four rather than eight: eight measured no better, the extra accumulators and the wider
     * shared-memory reduction costing what the saved traffic bought.
     */
    // @formatter:on
    private static final int COL_TILE = 4;

    /** Output rows a tiled batched projection covers per workgroup. */
    public static int colTile() {
        return COL_TILE;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a <b>tile of rows at once</b>, decoding each weight
     * once for the whole tile.
     *
     * <p>The untiled batched kernel is one workgroup per (row, output row), so a chunk of B tokens
     * reads the weight matrix B times — exactly what B separate single-token invocations read, and
     * measurably no faster than them. A quantized projection is memory-bound, so what a chunk is
     * worth is weight reuse: this decodes a block once and applies it to eight activations.
     *
     * <p>Per row the arithmetic is unchanged — the same lane-strided order over the same values —
     * so the tiled and untiled kernels agree element for element, and the chunk width stays
     * unobservable in the result.
     *
     * <p>{@code groupIdx} is {@code tile * d + row}. A tile past the active rows returns; a tile
     * that is partly active accumulates only its real rows.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ4_0(
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

        // Zeroed explicitly: a private array in generated device code is uninitialized stack,
        // not Java's zero-filled allocation. Leaving it out produced NaN logits, not small errors.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;

            // The activations, once, for every output row this workgroup covers. Inactive rows
            // read zero rather than branching: they contribute nothing either way, and they are
            // not written at the end.
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
     * #matrixVectorTiledBatchQ4_0}.
     */
    public static void matrixVectorTiledBatchWithResidualQ4_0(
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

        // Zeroed explicitly: a private array in generated device code is uninitialized stack,
        // not Java's zero-filled allocation. Leaving it out produced NaN logits, not small errors.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;

            // The activations, once, for every output row this workgroup covers. Inactive rows
            // read zero rather than branching: they contribute nothing either way, and they are
            // not written at the end.
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

    /** How many rows a tiled batched projection covers per workgroup. */
    public static int rowTile() {
        return ROW_TILE;
    }

    /**
     * Rows a tiled fused gate/up covers per workgroup. Four, because it holds two accumulators
     * each.
     */
    private static final int FFN_ROW_TILE = 8;

    /** Output rows a tiled fused gate/up covers per workgroup. See {@link #COL_TILE}. */
    private static final int FFN_COL_TILE = 2;

    /** Output rows a tiled fused gate/up covers per workgroup. */
    public static int ffnColTile() {
        return FFN_COL_TILE;
    }

    // @formatter:off
    /**
     * The fused gate/up feed-forward with SwiGLU over a <b>tile of rows</b>, decoding each weight
     * once for the tile.
     *
     * <p>The largest kernel in a decode step and the largest in a prefill chunk, and the untiled
     * batched form reads both weight matrices once per row — the same traffic as running the rows
     * separately. Four rows rather than eight: each row carries a gate and an up accumulator, so
     * the register and local-memory cost is the same as the eight-row projection tile.
     */
    // @formatter:on
    public static void fusedFFNGateUpSiLUTiledBatchQ4_0(
            KernelContext context,
            FloatArray xBatch,
            FloatArray hbBatch,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int colGroups = (d + FFN_COL_TILE - 1) / FFN_COL_TILE;
        int groupId = context.groupIdx;
        int tile = groupId / colGroups;
        int colGroup = groupId - tile * colGroups;
        int firstRow = tile * FFN_ROW_TILE;
        int firstCol = colGroup * FFN_COL_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        int slots = FFN_ROW_TILE * FFN_COL_TILE * 2;
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * slots);
        int blocksPerRow = (n + QK - 1) / QK;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack.
        float[] acc = new float[FFN_ROW_TILE * FFN_COL_TILE * 2];
        for (int t = 0; t < FFN_ROW_TILE * FFN_COL_TILE * 2; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[FFN_ROW_TILE];

        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;

            for (int r = 0; r < FFN_ROW_TILE; r++) {
                int row = firstRow + r;
                float value = 0.0f;
                if (row < activeRows) {
                    value = xBatch.get(row * n + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < FFN_COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset = (outRow * blocksPerRow + blockIdx) * BLOCK_BYTES;
                    float gateWeight = decode(w1, blockByteOffset, withinBlock);
                    float upWeight = decode(w3, blockByteOffset, withinBlock);
                    for (int r = 0; r < FFN_ROW_TILE; r++) {
                        int slot = (c * FFN_ROW_TILE + r) * 2;
                        acc[slot] += gateWeight * xs[r];
                        acc[slot + 1] += upWeight * xs[r];
                    }
                }
            }
        }

        for (int t = 0; t < FFN_ROW_TILE * FFN_COL_TILE * 2; t++) {
            localSums[t * localWorkGroupSize + localId] = acc[t];
        }
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < FFN_ROW_TILE * FFN_COL_TILE * 2; t++) {
                    localSums[t * localWorkGroupSize + localId] +=
                            localSums[t * localWorkGroupSize + localId + stride];
                }
            }
            context.localBarrier();
        }

        if (localId == 0) {
            for (int c = 0; c < FFN_COL_TILE; c++) {
                int outRow = firstCol + c;
                for (int r = 0; r < FFN_ROW_TILE; r++) {
                    int row = firstRow + r;
                    if (outRow < d && row < activeRows) {
                        int slot = (c * FFN_ROW_TILE + r) * 2;
                        float gateSum = localSums[slot * localWorkGroupSize];
                        float upSum = localSums[(slot + 1) * localWorkGroupSize];
                        float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
                        hbBatch.set(row * d + outRow, silu * upSum);
                    }
                }
            }
        }
    }

    /** How many rows a tiled fused gate/up covers per workgroup. */
    public static int ffnRowTile() {
        return FFN_ROW_TILE;
    }
}
