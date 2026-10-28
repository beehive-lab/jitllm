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
 * Device kernels that read {@code Q6_K} weights in the file's own representation.
 *
 * <p>The companion to {@link TransformerComputeKernelsQ4_K}, and the reason it has one: a "Q4_K_M"
 * file is not uniformly Q4_K. It mixes per tensor and per layer — Devstral's holds {@code attn_v}
 * and {@code ffn_down} as Q6_K in half its layers and Q4_K in the other half — so retaining Q4_K
 * alone still left those tensors materialized as Q8_0 at roughly 1.9x their size. With both, the
 * whole file stays in its own representation.
 *
 * <h2>The super-block</h2>
 *
 * <p>256 weights in 210 bytes: {@code ql} (128 B, the low 4 bits) at 0, {@code qh} (64 B, the high
 * 2 bits) at 128, sixteen signed per-16 scales at 192, and {@code d} (fp16) at 208. A weight is
 * {@code d * scale(sub) * (q - 32)}, where {@code q} is 6 bits assembled from the two planes. The
 * block is read as two halves of 128, each of four groups of 32, and which nibble and which pair of
 * {@code qh} bits a weight uses depends on its group — the arithmetic below is the host's {@code
 * Q6_KFloatTensor}, which {@code Q4_KDecodeTest} holds it against element by element.
 */
public final class TransformerComputeKernelsQ6_K {

    /** Weights per super-block. */
    private static final int QK_K = 256;

    /** Bytes per super-block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d). */
    private static final int BLOCK_BYTES = 210;

    private static final int QH_OFFSET = 128;
    private static final int SCALES_OFFSET = 192;
    private static final int D_OFFSET = 208;

    /**
     * Prompt rows a tiled batch workgroup covers: eight, as for Q5_K, whose decode is as costly.
     */
    private static final int ROW_TILE = 8;

    /** Output rows a tiled batch workgroup covers, as for Q5_K. */
    private static final int COL_TILE = 2;

    private TransformerComputeKernelsQ6_K() {}

    /**
     * The fp16 at {@code index}, assembled from two plain byte loads.
     *
     * <p>Deliberately not {@code ByteArray.getHalfFloat}: that call is what TornadoVM 5.2.0's
     * sketcher chokes on in this decode ("Unable to build sketch for method: fillInStackTrace"),
     * while whole-byte loads compile. Reading the two bytes and widening the half here keeps the
     * kernel on constructs the sketcher handles.
     *
     * <p>Subnormals are handled explicitly; infinities and NaNs are not, because a quantized block
     * scale is neither.
     */
    private static float halfFromBytes(ByteArray w, int index) {
        int lo = w.get(index) & 0xFF;
        int hi = w.get(index + 1) & 0xFF;
        int h = (hi << 8) | lo;
        int mantissa = h & 0x3FF;
        int exponent = (h >>> 10) & 0x1F;
        float magnitude;
        if (exponent == 0) {
            magnitude = mantissa * 5.9604645E-8f; // 2^-24, the subnormal step
        } else {
            // 2^(exponent-15) without a bit-pattern reinterpret and without a data-dependent
            // loop. Float.intBitsToFloat reaches the Metal backend as a node its LIR builder does
            // not implement ("TornadoInternalError: unimplemented" in MetalNodeLIRBuilder.doBlock),
            // and a counted loop over the exponent compiled but took minutes and produced zero.
            // The exponent is 1.30, so |e| < 16 and four fixed tests cover every case.
            int e = exponent - 15;
            int magnitudeOfE = e;
            if (e < 0) {
                magnitudeOfE = -e;
            }
            float scale = 1.0f;
            if ((magnitudeOfE & 1) != 0) {
                scale *= 2.0f;
            }
            if ((magnitudeOfE & 2) != 0) {
                scale *= 4.0f;
            }
            if ((magnitudeOfE & 4) != 0) {
                scale *= 16.0f;
            }
            if ((magnitudeOfE & 8) != 0) {
                scale *= 256.0f;
            }
            if (e < 0) {
                scale = 1.0f / scale;
            }
            magnitude = (1.0f + mantissa * (1.0f / 1024.0f)) * scale;
        }
        if ((h & 0x8000) != 0) {
            return -magnitude;
        }
        return magnitude;
    }

    /**
     * Decodes a Q6_K weight into FP16 at {@code destOffset} of {@code out}, for a GEMM that reads
     * FP16 operands. One thread per element of the {@code n}; the row width a multiple of 256, so
     * element {@code i} lies in super-block {@code i / 256}.
     */
    public static void dequantizeQ6_KToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset, int n) {
        int gid = context.globalIdx;
        if (gid < n) {
            // decode(...) copied in with its branches as selects: called through the helper, the
            // CUDA backend fails the FrameState dominance check on the FP16 store.
            int block = gid / QK_K;
            int blockByteOffset = block * BLOCK_BYTES;
            int withinBlock = gid - block * QK_K;
            float d = w.getHalfFloat(blockByteOffset + D_OFFSET).getFloat32();
            int halfIndex = withinBlock / 128;
            int posInHalf = withinBlock - halfIndex * 128;
            int groupInHalf = posInHalf / 32;
            int posInGroup = posInHalf - groupInHalf * 32;
            int qlBase = blockByteOffset + halfIndex * 64;
            int qhBase = blockByteOffset + QH_OFFSET + halfIndex * 32;
            int scBase = blockByteOffset + SCALES_OFFSET + halfIndex * 8;
            int is = posInGroup / 16;
            int qh = w.get(qhBase + posInGroup) & 0xFF;
            int ql = w.get(qlBase + (groupInHalf & 1) * 32 + posInGroup) & 0xFF;
            int lowBits = (ql >> ((groupInHalf >> 1) * 4)) & 0xF;
            int highPair = (qh >> (groupInHalf * 2)) & 3;
            int qValue = (lowBits | (highPair << 4)) - 32;
            int scale = w.get(scBase + is + groupInHalf * 2);
            out.set(destOffset + gid, new HalfFloat(d * scale * qValue));
        }
    }

    /**
     * {@link #dequantizeQ6_KToFP16} into a weight's resident FP16 copy, skipped once {@code
     * done[flag]} is set, on {@code TransformerBatchPrefillKernels.dequantizeQ8_0ToFP16Once}'s
     * fixed striding grid.
     */
    public static void dequantizeQ6_KToFP16Once(
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
                // decode(...) copied in with its branches as selects: called through the helper,
                // the
                // CUDA backend fails the FrameState dominance check on the FP16 store.
                int block = i / QK_K;
                int blockByteOffset = block * BLOCK_BYTES;
                int withinBlock = i - block * QK_K;
                float d = w.getHalfFloat(blockByteOffset + D_OFFSET).getFloat32();
                int halfIndex = withinBlock / 128;
                int posInHalf = withinBlock - halfIndex * 128;
                int groupInHalf = posInHalf / 32;
                int posInGroup = posInHalf - groupInHalf * 32;
                int qlBase = blockByteOffset + halfIndex * 64;
                int qhBase = blockByteOffset + QH_OFFSET + halfIndex * 32;
                int scBase = blockByteOffset + SCALES_OFFSET + halfIndex * 8;
                int is = posInGroup / 16;
                int qh = w.get(qhBase + posInGroup) & 0xFF;
                int ql = w.get(qlBase + (groupInHalf & 1) * 32 + posInGroup) & 0xFF;
                int lowBits = (ql >> ((groupInHalf >> 1) * 4)) & 0xF;
                int highPair = (qh >> (groupInHalf * 2)) & 3;
                int qValue = (lowBits | (highPair << 4)) - 32;
                int scale = w.get(scBase + is + groupInHalf * 2);
                out.set(destOffset + i, new HalfFloat(d * scale * qValue));
            }
        }
    }

    /**
     * One weight, decoded from its super-block.
     *
     * <p>Package-private so {@code Q4_KDecodeTest} can hold it against the host's {@code
     * Q6_KFloatTensor} directly, on the same bytes.
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = halfFromBytes(w, blockByteOffset + D_OFFSET);

        int halfIndex = withinBlock / 128; // 0 or 1
        int posInHalf = withinBlock - halfIndex * 128; // 0..127
        int groupInHalf = posInHalf / 32; // 0..3
        int posInGroup = posInHalf - groupInHalf * 32; // 0..31

        int qlBase = blockByteOffset + halfIndex * 64;
        int qhBase = blockByteOffset + QH_OFFSET + halfIndex * 32;
        int scBase = blockByteOffset + SCALES_OFFSET + halfIndex * 8;

        int is = posInGroup / 16; // 0 or 1
        int qh = w.get(qhBase + posInGroup) & 0xFF;

        int secondPlane = groupInHalf & 1;
        int highNibble = groupInHalf >> 1;
        int ql = w.get(qlBase + secondPlane * 32 + posInGroup) & 0xFF;
        int lowBits = ql & 0xF;
        if (highNibble == 1) {
            lowBits = (ql >> 4) & 0xF;
        }
        int highPair = qh & 3;
        if (groupInHalf == 1) {
            highPair = (qh >> 2) & 3;
        } else if (groupInHalf == 2) {
            highPair = (qh >> 4) & 3;
        } else if (groupInHalf == 3) {
            highPair = (qh >> 6) & 3;
        }
        int qValue = (lowBits | (highPair << 4)) - 32;

        int scale = w.get(scBase + is + groupInHalf * 2) & 0xFF;
        if (scale > 127) {
            scale = scale - 256;
        }
        return d * scale * qValue;
    }

    /** The byte at {@code index} as a signed scale, read as {@link #decode} reads it. */
    private static int signedByte(ByteArray w, int index) {
        int value = w.get(index) & 0xFF;
        if (value > 127) {
            value = value - 256;
        }
        return value;
    }

    /**
     * One row's dot product against {@code x}, reduced through shared memory.
     *
     * <p>A thread takes thirty-two weights per step: eight consecutive positions of a half-block's
     * thirty-two, in each of its four groups, the layout of llama.cpp's Metal {@code
     * kernel_mul_mv_q6_K_f32}. The four groups at one position are built from the same {@code qh}
     * byte and two {@code ql} bytes. Those are read four at a time, as words of two two-byte loads
     * ({@link #matrixVectorGenericQ6_KDP4A} documents why every address is even), and the six-bit
     * values are assembled on the whole word, four weights per operation. Each weight is still
     * {@code d * scale * q} as {@link #decode} computes it; only the order of the summation
     * differs. {@code n} is a multiple of 256.
     *
     * <p>Measured on Qwen3-0.6B Q4_K_M on an M4 Pro, the vocabulary projection per call (profiler):
     * 11,433 µs one weight per step, 3,445 µs sixteen weights from byte loads, 2,699 µs as here.
     */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int rowByteOffset = rowId * (n / QK_K) * BLOCK_BYTES;

        float partialSum = 0.0f;
        for (int unit = localId; unit < n / 32; unit += localSize) {
            int half = unit / 4;
            int posInGroup = (unit - half * 4) * 8;
            int blockIdx = half / 2;
            int halfIndex = half - blockIdx * 2;
            int blockByteOffset = rowByteOffset + blockIdx * BLOCK_BYTES;
            float d = w.getHalfFloat(blockByteOffset + D_OFFSET).getFloat32();

            int scaleAt = blockByteOffset + SCALES_OFFSET + halfIndex * 8 + posInGroup / 16;
            float dScale0 = d * signedByte(w, scaleAt);
            float dScale1 = d * signedByte(w, scaleAt + 2);
            float dScale2 = d * signedByte(w, scaleAt + 4);
            float dScale3 = d * signedByte(w, scaleAt + 6);

            int qlAt = blockByteOffset + halfIndex * 64 + posInGroup;
            int qhAt = blockByteOffset + QH_OFFSET + halfIndex * 32 + posInGroup;
            int xAt = blockIdx * QK_K + halfIndex * 128 + posInGroup;
            for (int k = 0; k < 8; k += 4) {
                int ql1Word =
                        (w.getHalfFloat(qlAt + k).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qlAt + k + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int ql2Word =
                        (w.getHalfFloat(qlAt + 32 + k).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qlAt + 34 + k).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int qhWord =
                        (w.getHalfFloat(qhAt + k).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qhAt + k + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int raw0 = (ql1Word & 0x0F0F0F0F) | ((qhWord & 0x03030303) << 4);
                int raw1 = (ql2Word & 0x0F0F0F0F) | (((qhWord >>> 2) & 0x03030303) << 4);
                int raw2 = ((ql1Word >>> 4) & 0x0F0F0F0F) | (((qhWord >>> 4) & 0x03030303) << 4);
                int raw3 = ((ql2Word >>> 4) & 0x0F0F0F0F) | (((qhWord >>> 6) & 0x03030303) << 4);
                for (int b = 0; b < 4; b++) {
                    int shift = 8 * b;
                    partialSum += dScale0 * (((raw0 >>> shift) & 0xFF) - 32) * x.get(xAt + k + b);
                    partialSum +=
                            dScale1 * (((raw1 >>> shift) & 0xFF) - 32) * x.get(xAt + 32 + k + b);
                    partialSum +=
                            dScale2 * (((raw2 >>> shift) & 0xFF) - 32) * x.get(xAt + 64 + k + b);
                    partialSum +=
                            dScale3 * (((raw3 >>> shift) & 0xFF) - 32) * x.get(xAt + 96 + k + b);
                }
            }
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
     * {@code output[row] = w[row]·x} with {@code Q6_K} weights and a packed-integer dot product.
     *
     * <p>For the vocabulary projection, which is one call per token against a quarter of a million
     * output rows and is the only {@code Q6_K} site this is dispatched to.
     *
     * <p><b>The recentring is done on the weight side.</b> A Q6_K quantum is {@code raw - 32} with
     * {@code raw} in 0..63, so the recentred value is in -32..31 and fits a signed byte directly.
     * Building it that way is what {@code vec_dot_q6_K_q8_1_impl_mmvq} does, and it is why this
     * kernel needs no per-run sum of the activation's quants: there is no correction term. The
     * subtraction is masked to a byte as it is packed, which is what makes it safe against the
     * unsigned-stamp wrap in the TornadoVM backend -- the wrapped value and the correct one agree
     * in their low eight bits, and the test walks all 64 values.
     *
     * <p><b>Scale structure.</b> A super-block is 256 weights with one fp16 {@code d}, sixteen
     * signed byte scales, and a quantum split across {@code ql} and {@code qh}. A scale covers
     * sixteen <b>contiguous</b> elements, and the activation's quantization block is thirty-two, so
     * each scale run sits inside one activation block and the two scales multiply cleanly. The
     * element-to-byte mapping is the host tensor's, restated: two halves of 128, four groups of 32
     * within a half, the low nibble for the first two groups and the high nibble for the last two,
     * with the {@code qh} pair of bits shifted by the group.
     *
     * <p>The activation must have been prepared by {@code quantizeActivationQ8Blocks} from the
     * activation this projection reads.
     *
     * <p><b>The reduction is a warp-shuffle butterfly</b>, the same tail {@code
     * matrixVectorGenericQ4_0DP4A} uses: each 32-lane warp folds its own partial with five {@code
     * simdShuffleDown} steps, one lane per warp writes to shared memory, and a single barrier
     * separates that from the combine — one barrier where the tree needed eight at the 256-lane
     * width the vocabulary projection dispatches. All lanes reach the shuffles; the only early
     * return is on {@code rowId}, which is uniform across the workgroup, and {@code
     * localWorkGroupSize} must be a multiple of 32. Shuffles are correct on CUDA and miscompile on
     * OpenCL, which is already why this kernel rides on {@code
     * DeviceCapability.PACKED_INTEGER_DOT}. The order of summation differs from the tree's, so the
     * floating-point total may round differently; the integer dot products are exact either way,
     * and nothing about the mapping, the packing or the scale algebra changes.
     */
    // @formatter:on
    public static void matrixVectorGenericQ6_KDP4A(
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
                    warpPartialDP4A(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
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
     * #matrixVectorGenericWithResidualQ6_K}: the residual projections a Q4_K_M file stores as Q6_K
     * ({@code ffn_down} in half of Devstral's layers). Same mapping and reduction as {@link
     * #matrixVectorGenericQ6_KDP4A}; {@code xSums} is not read, since Q6_K recentres on the weight
     * side, and is taken so every packed K-quant projection has one signature.
     */
    public static void matrixVectorGenericWithResidualQ6_KDP4A(
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
                    warpPartialDP4A(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
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
     * This lane's share of row {@code rowId}'s dot product, folded across its warp with shuffles:
     * valid in the warp's lane 0. The cross-warp combine stays in the kernel, which owns the local
     * array it goes through.
     */
    private static float warpPartialDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        float partialSum =
                rowPartialDP4A(context, xQuants, xScales, w, n, rowId, localWorkGroupSize);
        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /** This lane's share of row {@code rowId}'s dot product, one scale run of sixteen per step. */
    private static float rowPartialDP4A(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            ByteArray w,
            int n,
            int rowId,
            int localWorkGroupSize) {
        int localId = context.localIdx;
        int superBlocksPerRow = n / QK_K;
        int rowByteOffset = rowId * superBlocksPerRow * BLOCK_BYTES;

        float partialSum = 0.0f;
        // One scale run of sixteen weights per step: the unit over which d, the byte scale and the
        // activation's own scale are all constant.
        int runsPerRow = n / 16;
        for (int run = localId; run < runsPerRow; run += localWorkGroupSize) {
            int superBlock = run / 16;
            int runInBlock = run - superBlock * 16;
            int base = rowByteOffset + superBlock * BLOCK_BYTES;
            float d6 = w.getHalfFloat(base + D_OFFSET).getFloat32();

            int half = runInBlock / 8;
            int groupInHalf = (runInBlock - half * 8) / 2;
            int is = runInBlock & 1;
            int scale = w.get(base + SCALES_OFFSET + half * 8 + is + 2 * groupInHalf);

            int qlBase = base + half * 64 + (groupInHalf & 1) * 32 + is * 16;
            int qhBase = base + QH_OFFSET + half * 32 + is * 16;
            int qhShift = 2 * groupInHalf;
            boolean highNibble = groupInHalf >= 2;

            int quantBase = run * 4;
            int dot = 0;
            for (int g = 0; g < 4; g++) {
                // Both planes two bytes at a time, as TransformerComputeKernelsQ4_0 documents,
                // with the whole reconstruction on the packed word rather than lane by lane. A
                // super-block is 210 bytes, so its start is even but only every other one is
                // four-byte aligned; qlBase and qhBase add multiples of sixteen and this adds a
                // multiple of four, so every address here is even, which is what the pair read
                // needs. The last pair of a run reads qlBase+14..15 and qhBase+14..15, inside
                // ql's 0..127 and qh's 128..191.
                int at = g * 4;
                int qlWord =
                        (w.getHalfFloat(qlBase + at).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qlBase + at + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                int qhWord =
                        (w.getHalfFloat(qhBase + at).getHalfFloatValue() & 0xFFFF)
                                | ((w.getHalfFloat(qhBase + at + 2).getHalfFloatValue() & 0xFFFF)
                                        << 16);
                // The nibble the run takes, across the word. Uniform per run, not per lane.
                int low = qlWord & 0x0F0F0F0F;
                if (highNibble) {
                    low = (qlWord >>> 4) & 0x0F0F0F0F;
                }
                // Q6_K's high plane carries TWO bits per weight, not one: qhShift is 0, 2, 4 or
                // 6 and the field is two wide, so it never leaves its byte and a single shift of
                // the whole word with a 0x03030303 mask extracts all four at once.
                int raw = low | (((qhWord >>> qhShift) & 0x03030303) << 4);
                // The weight-side recentring, raw - 32, on all four bytes. A word subtract of
                // 0x20202020 would be wrong: a byte below 32 borrows into its neighbour. Each
                // byte is a six-bit value, so flipping bit five gives (raw - 32) modulo 64 and
                // what remains is to propagate that bit into bits six and seven, which is a
                // sign extension from six bits to eight and stays inside the byte.
                int flipped = raw ^ 0x20202020;
                int signBit = flipped & 0x20202020;
                int packed = flipped | (signBit << 1) | (signBit << 2);
                dot = QuantizationUtils.dp4a_packed(packed, xQuants.get(quantBase + g), dot);
            }
            partialSum += d6 * scale * xScales.get(run / 2) * dot;
        }

        return partialSum;
    }

    /** {@code output[row] = w[row]·x}. */
    public static void matrixVectorGenericQ6_K(
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
    public static void matrixVectorGenericWithResidualQ6_K(
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

    /**
     * {@code hb[i] = silu(hb[i]) * hb2[i]} — the SwiGLU half of the FFN, as its own task.
     *
     * <p>Separate from the projections because the gate and up weights of a mixed K-quant file can
     * be different representations, so they are computed by two independently-selected matvecs
     * rather than one fused kernel. Elementwise and dtype-independent; it lives here because this
     * is the path that needs it.
     */
    public static void siluAndMultiply(
            KernelContext context, FloatArray hb, FloatArray hb2, int d) {
        int i = context.globalIdx;
        if (i >= d) {
            return;
        }
        float gate = hb.get(i);
        hb.set(i, (gate / (1.0f + TornadoMath.exp(-gate))) * hb2.get(i));
    }

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    /** Output rows a tiled batch workgroup covers. */
    public static int colTile() {
        return COL_TILE;
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
    public static void matrixVectorTiledBatchQ6_K(
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
            int halfIndex = subInBlock >> 2;
            int groupInHalf = subInBlock & 3;
            int lowShift = (groupInHalf >> 1) * 4;
            int highShift = groupInHalf * 2;
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
                    float dScale = halfFromBytes(w, blockByteOffset + D_OFFSET);
                    int ql =
                            w.get(blockByteOffset + halfIndex * 64 + (groupInHalf & 1) * 32 + lane)
                                    & 0xFF;
                    int qh = w.get(blockByteOffset + QH_OFFSET + halfIndex * 32 + lane) & 0xFF;
                    int q = (((ql >> lowShift) & 0xF) | (((qh >> highShift) & 3) << 4)) - 32;
                    int scale =
                            w.get(
                                            blockByteOffset
                                                    + SCALES_OFFSET
                                                    + halfIndex * 8
                                                    + (lane >> 4)
                                                    + groupInHalf * 2)
                                    & 0xFF;
                    if (scale > 127) {
                        scale = scale - 256;
                    }
                    float weight = dScale * scale * q;
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
     * #matrixVectorTiledBatchQ6_K}.
     */
    public static void matrixVectorTiledBatchWithResidualQ6_K(
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
            int halfIndex = subInBlock >> 2;
            int groupInHalf = subInBlock & 3;
            int lowShift = (groupInHalf >> 1) * 4;
            int highShift = groupInHalf * 2;
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
                    float dScale = halfFromBytes(w, blockByteOffset + D_OFFSET);
                    int ql =
                            w.get(blockByteOffset + halfIndex * 64 + (groupInHalf & 1) * 32 + lane)
                                    & 0xFF;
                    int qh = w.get(blockByteOffset + QH_OFFSET + halfIndex * 32 + lane) & 0xFF;
                    int q = (((ql >> lowShift) & 0xF) | (((qh >> highShift) & 3) << 4)) - 32;
                    int scale =
                            w.get(
                                            blockByteOffset
                                                    + SCALES_OFFSET
                                                    + halfIndex * 8
                                                    + (lane >> 4)
                                                    + groupInHalf * 2)
                                    & 0xFF;
                    if (scale > 127) {
                        scale = scale - 256;
                    }
                    float weight = dScale * scale * q;
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
