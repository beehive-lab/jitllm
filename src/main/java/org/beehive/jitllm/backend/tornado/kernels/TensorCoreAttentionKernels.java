package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The layout work around a prefill attention computed as tensor-core GEMMs: queries in FP16, and
 * the paged FP16 key/value cache gathered into contiguous per-key/value-head tiles, the values
 * transposed. The scores, the causal softmax and the weighted values are {@link
 * BatchMmaKernels#gemmMMAStrided} and {@link BatchMmaKernels#causalSoftmax}.
 *
 * <p>{@code batchInfo} is the chunk's {@code [start position, active rows, slot]}.
 */
// @formatter:on
public final class TensorCoreAttentionKernels {

    private TensorCoreAttentionKernels() {}

    /**
     * {@code out[t][c] = qkv[t * qkvStride + c]} in FP16 for the {@code dim} query columns of every
     * padded row. Worker: {@code rows * dim} lanes.
     */
    public static void queriesToHalf(
            KernelContext context,
            FloatArray qkv,
            HalfFloatArray out,
            int dim,
            int qkvStride,
            int total) {
        int i = context.globalIdx;
        if (i < total) {
            int t = i / dim;
            int c = i - t * dim;
            out.set(i, new HalfFloat(qkv.get(t * qkvStride + c)));
        }
    }

    // @formatter:off
    /**
     * The layer's paged FP16 cache up to the chunk's last position, per key/value head {@code g}:
     * {@code keys[g][t][d]} and {@code valuesT[g][d][t]}, {@code paddedPositions} positions each.
     * Positions past the chunk's last one, up to the next multiple of 128, are written as zero;
     * nothing past that is read. Worker: {@code paddedPositions * kvDim} lanes.
     */
    // @formatter:on
    public static void gatherPagedKeyValues(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keys,
            HalfFloatArray valuesT,
            int layer,
            int kvDim,
            int headSize,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int kIndex = (g * paddedPositions + t) * headSize + d;
            int vIndex = (g * headSize + d) * paddedPositions + t;
            if (t < valid) {
                int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
                int off =
                        KvBlockAddress.offset(
                                        blockTable,
                                        batchInfo.get(2),
                                        t,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + c;
                keys.set(kIndex, keyCache.get(off));
                valuesT.set(vIndex, valueCache.get(off));
            }
            if (t >= valid) {
                keys.set(kIndex, new HalfFloat(0.0f));
                valuesT.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }

    // @formatter:off
    /**
     * {@link #gatherPagedKeyValues} for a contiguous FP16 cache: the layer's rows start at {@code
     * cacheOffset}, one {@code kvDim}-wide row per position.
     */
    // @formatter:on
    public static void gatherKeyValues(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keys,
            HalfFloatArray valuesT,
            int cacheOffset,
            int kvDim,
            int headSize,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int kIndex = (g * paddedPositions + t) * headSize + d;
            int vIndex = (g * headSize + d) * paddedPositions + t;
            if (t < valid) {
                keys.set(kIndex, keyCache.get(cacheOffset + idx));
                valuesT.set(vIndex, valueCache.get(cacheOffset + idx));
            }
            if (t >= valid) {
                keys.set(kIndex, new HalfFloat(0.0f));
                valuesT.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }

    /** Lanes of a softmax workgroup: one workgroup per (token, head) row. */
    public static final int SOFTMAX_LANES = 256;

    // @formatter:off
    /**
     * The causal, sliding-window softmax of every (token, head) row of scores, in FP16: row {@code
     * token * heads + head} attends to positions {@code max(0, p - window + 1) .. p}, {@code p =
     * start + token}, scaled; every other entry up to the chunk's last position, and every row past
     * the active ones, is zero. A workgroup of {@link #SOFTMAX_LANES} lanes per row.
     */
    // @formatter:on
    public static void windowedSoftmax(
            KernelContext context,
            IntArray batchInfo,
            FloatArray scores,
            HalfFloatArray probs,
            int heads,
            int paddedPositions,
            float scale,
            int window) {
        int row = context.groupIdx;
        int tid = context.localIdx;
        int t = row / heads;
        int base = row * paddedPositions;
        float[] reduce = context.allocateFloatLocalArray(SOFTMAX_LANES);
        // A padding row attends to nothing, without a branch whose merged value is miscompiled.
        int active = Math.min(1, Math.max(0, batchInfo.get(1) - t));
        int pos = batchInfo.get(0) + t;
        int end = active * (pos + 1);
        int lo = Math.max(0, pos - window + 1);
        float localMax = Float.NEGATIVE_INFINITY;
        for (int j = lo + tid; j < end; j += SOFTMAX_LANES) {
            localMax = TornadoMath.max(localMax, scores.get(base + j) * scale);
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = SOFTMAX_LANES / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = TornadoMath.max(reduce[tid], reduce[tid + stride]);
            }
            context.localBarrier();
        }
        float max = reduce[0];
        context.localBarrier();
        float localSum = 0.0f;
        for (int j = lo + tid; j < end; j += SOFTMAX_LANES) {
            localSum += TornadoMath.exp(scores.get(base + j) * scale - max);
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = SOFTMAX_LANES / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float inverse = 1.0f / TornadoMath.max(reduce[0], 1.0e-30f);
        int filled = ((batchInfo.get(0) + batchInfo.get(1) + 127) / 128) * 128;
        int width = Math.min(paddedPositions, filled);
        for (int j = tid; j < width; j += SOFTMAX_LANES) {
            if (j >= lo && j < end) {
                probs.set(
                        base + j,
                        new HalfFloat(
                                TornadoMath.exp(scores.get(base + j) * scale - max) * inverse));
            }
            if (j < lo || j >= end) {
                probs.set(base + j, new HalfFloat(0.0f));
            }
        }
    }

    /** Query rows per workgroup of {@link #flashAttentionHead128}. */
    public static final int FLASH_ROWS = 64;

    /** Threads per workgroup of {@link #flashAttentionHead128}: 4 warps. */
    public static final int FLASH_THREADS = 128;

    // @formatter:off
    /**
     * {@link #gatherPagedKeyValues} in the layouts {@link #flashAttentionHead128} reads, 64
     * positions per block {@code b} so that a block is one contiguous copy: keys as {@code [g][b][t
     * / 8][d][t % 8]} and values as {@code [g][b][d / 8][t][d % 8]} ({@code t} within the block),
     * the 16 x 8 tiles the tensor cores read. Worker: {@code paddedPositions * kvDim} lanes.
     */
    // @formatter:on
    public static void gatherPagedKeysTransposed(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keysT,
            HalfFloatArray values,
            int layer,
            int kvDim,
            int headSize,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int block = (g * (paddedPositions / 64) + (t >> 6)) * 8192;
            int kIndex = block + (((t & 63) >> 3) * headSize + d) * 8 + (t & 7);
            int vIndex = block + ((d >> 3) * 64 + (t & 63)) * 8 + (d & 7);
            if (t < valid) {
                int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
                int off =
                        KvBlockAddress.offset(
                                        blockTable,
                                        batchInfo.get(2),
                                        t,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + c;
                keysT.set(kIndex, keyCache.get(off));
                values.set(vIndex, valueCache.get(off));
            }
            if (t >= valid) {
                keysT.set(kIndex, new HalfFloat(0.0f));
                values.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }

    // @formatter:off
    /**
     * Causal attention for heads 128 wide on tensor cores, without the scores in memory. A
     * workgroup takes 64 query rows of one head, 16 per warp. Keys go 64 at a time, twice: first
     * for each row's maximum and sum, then for the weighted values. Reads {@code queries[t][h * 128
     * + d]}, and the keys and values of {@link #gatherPagedKeysTransposed}, one block copied to
     * local memory at a time. Writes {@code out[t][h * 128 + d]}, zero for rows past the chunk's
     * tokens. Worker: {@code (rows / 64) * heads} workgroups of {@link #FLASH_THREADS}.
     */
    // @formatter:on
    public static void flashAttentionHead128(
            KernelContext ctx,
            IntArray batchInfo,
            HalfFloatArray queries,
            HalfFloatArray keysT,
            HalfFloatArray values,
            HalfFloatArray out,
            int heads,
            int kvMul,
            int qDim,
            int rowBlocks,
            int paddedPositions,
            float scale) {
        int[] kTile = ctx.allocateIntLocalArray(4096);
        int[] vTile = ctx.allocateIntLocalArray(4096);
        int[] pTile = ctx.allocateIntLocalArray(4 * 16 * 36);
        float[] rowMax = ctx.allocateFloatLocalArray(64);
        float[] rowInv = ctx.allocateFloatLocalArray(64);
        int tid = ctx.localIdx;
        int warp = tid >> 5;
        int lane = tid & 31;
        int quad = lane & 3;
        int h = ctx.groupIdx % heads;
        // The row blocks with the most keys start first.
        int wgRow = (rowBlocks - 1 - ctx.groupIdx / heads) * 64;
        int g = h / kvMul;
        int start = batchInfo.get(0);
        int tokens = batchInfo.get(1);
        int lastRow = Math.min(wgRow + 64, tokens);
        int blocks = lastRow > wgRow ? (start + lastRow + 63) / 64 : 0;
        int rowBase = wgRow + warp * 16;
        int r0 = rowBase + (lane >> 2);
        int p0 = start + r0;
        int p1 = p0 + 8;
        // A 64-key block of one key/value head: 8192 halves of keys, then 8192 of values.
        int gBase = g * (paddedPositions / 64) * 8192;
        int qCol = h * 128;
        HalfFloat[] q0 = ctx.mmaLoadA(queries, rowBase, qCol + 0, qDim);
        HalfFloat[] q1 = ctx.mmaLoadA(queries, rowBase, qCol + 16, qDim);
        HalfFloat[] q2 = ctx.mmaLoadA(queries, rowBase, qCol + 32, qDim);
        HalfFloat[] q3 = ctx.mmaLoadA(queries, rowBase, qCol + 48, qDim);
        HalfFloat[] q4 = ctx.mmaLoadA(queries, rowBase, qCol + 64, qDim);
        HalfFloat[] q5 = ctx.mmaLoadA(queries, rowBase, qCol + 80, qDim);
        HalfFloat[] q6 = ctx.mmaLoadA(queries, rowBase, qCol + 96, qDim);
        HalfFloat[] q7 = ctx.mmaLoadA(queries, rowBase, qCol + 112, qDim);
        float m0 = -1.0e30f;
        float l0 = 0.0f;
        float m1 = -1.0e30f;
        float l1 = 0.0f;
        for (int kb = 0; kb < blocks; kb++) {
            int key0 = kb * 64;
            int kSrc = gBase + kb * 8192;
            for (int i = tid; i < 1024; i += 128) {
                ctx.asyncCopyToLocal16(kTile, 4 * i, keysT, kSrc + 8 * i);
            }
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();
            float[] s0 = ctx.mmaFragment(0.0f);
            float[] s1 = ctx.mmaFragment(0.0f);
            float[] s2 = ctx.mmaFragment(0.0f);
            float[] s3 = ctx.mmaFragment(0.0f);
            float[] s4 = ctx.mmaFragment(0.0f);
            float[] s5 = ctx.mmaFragment(0.0f);
            float[] s6 = ctx.mmaFragment(0.0f);
            float[] s7 = ctx.mmaFragment(0.0f);
            s0 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 0), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 2048), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 4096), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 6144), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 8192), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 10240), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 12288), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 14336), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 256), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 2304), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 4352), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 6400), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 8448), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 10496), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 12544), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 14592), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 512), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 2560), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 4608), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 6656), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 8704), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 10752), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 12800), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 14848), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 768), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 2816), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 4864), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 6912), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 8960), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 11008), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 13056), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 15104), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 1024), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 3072), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 5120), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 7168), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 9216), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 11264), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 13312), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 15360), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 1280), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 3328), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 5376), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 7424), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 9472), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 11520), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 13568), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 15616), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 1536), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 3584), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 5632), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 7680), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 9728), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 11776), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 13824), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 15872), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 1792), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 3840), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 5888), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 7936), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 9984), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 12032), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 14080), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 16128), s7, MMAShape.M16N8K16);
            ctx.localBarrier();
            int c = key0 + 2 * quad;
            float a00 = c <= p0 ? s0[0] * scale : -1.0e30f;
            float a01 = c + 1 <= p0 ? s0[1] * scale : -1.0e30f;
            float a02 = c <= p1 ? s0[2] * scale : -1.0e30f;
            float a03 = c + 1 <= p1 ? s0[3] * scale : -1.0e30f;
            float a10 = c + 8 <= p0 ? s1[0] * scale : -1.0e30f;
            float a11 = c + 9 <= p0 ? s1[1] * scale : -1.0e30f;
            float a12 = c + 8 <= p1 ? s1[2] * scale : -1.0e30f;
            float a13 = c + 9 <= p1 ? s1[3] * scale : -1.0e30f;
            float a20 = c + 16 <= p0 ? s2[0] * scale : -1.0e30f;
            float a21 = c + 17 <= p0 ? s2[1] * scale : -1.0e30f;
            float a22 = c + 16 <= p1 ? s2[2] * scale : -1.0e30f;
            float a23 = c + 17 <= p1 ? s2[3] * scale : -1.0e30f;
            float a30 = c + 24 <= p0 ? s3[0] * scale : -1.0e30f;
            float a31 = c + 25 <= p0 ? s3[1] * scale : -1.0e30f;
            float a32 = c + 24 <= p1 ? s3[2] * scale : -1.0e30f;
            float a33 = c + 25 <= p1 ? s3[3] * scale : -1.0e30f;
            float a40 = c + 32 <= p0 ? s4[0] * scale : -1.0e30f;
            float a41 = c + 33 <= p0 ? s4[1] * scale : -1.0e30f;
            float a42 = c + 32 <= p1 ? s4[2] * scale : -1.0e30f;
            float a43 = c + 33 <= p1 ? s4[3] * scale : -1.0e30f;
            float a50 = c + 40 <= p0 ? s5[0] * scale : -1.0e30f;
            float a51 = c + 41 <= p0 ? s5[1] * scale : -1.0e30f;
            float a52 = c + 40 <= p1 ? s5[2] * scale : -1.0e30f;
            float a53 = c + 41 <= p1 ? s5[3] * scale : -1.0e30f;
            float a60 = c + 48 <= p0 ? s6[0] * scale : -1.0e30f;
            float a61 = c + 49 <= p0 ? s6[1] * scale : -1.0e30f;
            float a62 = c + 48 <= p1 ? s6[2] * scale : -1.0e30f;
            float a63 = c + 49 <= p1 ? s6[3] * scale : -1.0e30f;
            float a70 = c + 56 <= p0 ? s7[0] * scale : -1.0e30f;
            float a71 = c + 57 <= p0 ? s7[1] * scale : -1.0e30f;
            float a72 = c + 56 <= p1 ? s7[2] * scale : -1.0e30f;
            float a73 = c + 57 <= p1 ? s7[3] * scale : -1.0e30f;
            float b0 =
                    TornadoMath.max(
                            TornadoMath.max(
                                    TornadoMath.max(
                                            TornadoMath.max(
                                                    TornadoMath.max(
                                                            TornadoMath.max(
                                                                    TornadoMath.max(
                                                                            TornadoMath.max(
                                                                                    TornadoMath.max(
                                                                                            TornadoMath
                                                                                                    .max(
                                                                                                            TornadoMath
                                                                                                                    .max(
                                                                                                                            TornadoMath
                                                                                                                                    .max(
                                                                                                                                            TornadoMath
                                                                                                                                                    .max(
                                                                                                                                                            TornadoMath
                                                                                                                                                                    .max(
                                                                                                                                                                            TornadoMath
                                                                                                                                                                                    .max(
                                                                                                                                                                                            a00,
                                                                                                                                                                                            a01),
                                                                                                                                                                            a10),
                                                                                                                                                            a11),
                                                                                                                                            a20),
                                                                                                                            a21),
                                                                                                            a30),
                                                                                            a31),
                                                                                    a40),
                                                                            a41),
                                                                    a50),
                                                            a51),
                                                    a60),
                                            a61),
                                    a70),
                            a71);
            float b1 =
                    TornadoMath.max(
                            TornadoMath.max(
                                    TornadoMath.max(
                                            TornadoMath.max(
                                                    TornadoMath.max(
                                                            TornadoMath.max(
                                                                    TornadoMath.max(
                                                                            TornadoMath.max(
                                                                                    TornadoMath.max(
                                                                                            TornadoMath
                                                                                                    .max(
                                                                                                            TornadoMath
                                                                                                                    .max(
                                                                                                                            TornadoMath
                                                                                                                                    .max(
                                                                                                                                            TornadoMath
                                                                                                                                                    .max(
                                                                                                                                                            TornadoMath
                                                                                                                                                                    .max(
                                                                                                                                                                            TornadoMath
                                                                                                                                                                                    .max(
                                                                                                                                                                                            a02,
                                                                                                                                                                                            a03),
                                                                                                                                                                            a12),
                                                                                                                                                            a13),
                                                                                                                                            a22),
                                                                                                                            a23),
                                                                                                            a32),
                                                                                            a33),
                                                                                    a42),
                                                                            a43),
                                                                    a52),
                                                            a53),
                                                    a62),
                                            a63),
                                    a72),
                            a73);
            float n0 = TornadoMath.max(m0, b0);
            float n1 = TornadoMath.max(m1, b1);
            float e0 =
                    TornadoMath.exp(a00 - n0)
                            + TornadoMath.exp(a01 - n0)
                            + TornadoMath.exp(a10 - n0)
                            + TornadoMath.exp(a11 - n0)
                            + TornadoMath.exp(a20 - n0)
                            + TornadoMath.exp(a21 - n0)
                            + TornadoMath.exp(a30 - n0)
                            + TornadoMath.exp(a31 - n0)
                            + TornadoMath.exp(a40 - n0)
                            + TornadoMath.exp(a41 - n0)
                            + TornadoMath.exp(a50 - n0)
                            + TornadoMath.exp(a51 - n0)
                            + TornadoMath.exp(a60 - n0)
                            + TornadoMath.exp(a61 - n0)
                            + TornadoMath.exp(a70 - n0)
                            + TornadoMath.exp(a71 - n0);
            float e1 =
                    TornadoMath.exp(a02 - n1)
                            + TornadoMath.exp(a03 - n1)
                            + TornadoMath.exp(a12 - n1)
                            + TornadoMath.exp(a13 - n1)
                            + TornadoMath.exp(a22 - n1)
                            + TornadoMath.exp(a23 - n1)
                            + TornadoMath.exp(a32 - n1)
                            + TornadoMath.exp(a33 - n1)
                            + TornadoMath.exp(a42 - n1)
                            + TornadoMath.exp(a43 - n1)
                            + TornadoMath.exp(a52 - n1)
                            + TornadoMath.exp(a53 - n1)
                            + TornadoMath.exp(a62 - n1)
                            + TornadoMath.exp(a63 - n1)
                            + TornadoMath.exp(a72 - n1)
                            + TornadoMath.exp(a73 - n1);
            l0 = l0 * TornadoMath.exp(m0 - n0) + e0;
            l1 = l1 * TornadoMath.exp(m1 - n1) + e1;
            m0 = n0;
            m1 = n1;
        }
        // Each row's four lanes merge their maxima and sums into the first one.
        float om01 = ctx.simdShuffleDown(m0, 1);
        float ol01 = ctx.simdShuffleDown(l0, 1);
        float mm01 = TornadoMath.max(m0, om01);
        l0 = l0 * TornadoMath.exp(m0 - mm01) + ol01 * TornadoMath.exp(om01 - mm01);
        m0 = mm01;
        float om11 = ctx.simdShuffleDown(m1, 1);
        float ol11 = ctx.simdShuffleDown(l1, 1);
        float mm11 = TornadoMath.max(m1, om11);
        l1 = l1 * TornadoMath.exp(m1 - mm11) + ol11 * TornadoMath.exp(om11 - mm11);
        m1 = mm11;
        float om02 = ctx.simdShuffleDown(m0, 2);
        float ol02 = ctx.simdShuffleDown(l0, 2);
        float mm02 = TornadoMath.max(m0, om02);
        l0 = l0 * TornadoMath.exp(m0 - mm02) + ol02 * TornadoMath.exp(om02 - mm02);
        m0 = mm02;
        float om12 = ctx.simdShuffleDown(m1, 2);
        float ol12 = ctx.simdShuffleDown(l1, 2);
        float mm12 = TornadoMath.max(m1, om12);
        l1 = l1 * TornadoMath.exp(m1 - mm12) + ol12 * TornadoMath.exp(om12 - mm12);
        m1 = mm12;
        int statRow = warp * 16 + (lane >> 2);
        if (quad == 0) {
            rowMax[statRow] = m0;
            rowInv[statRow] = 1.0f / TornadoMath.max(l0, 1.0e-30f);
            rowMax[statRow + 8] = m1;
            rowInv[statRow + 8] = 1.0f / TornadoMath.max(l1, 1.0e-30f);
        }
        ctx.localBarrier();
        float x0 = rowMax[statRow];
        float i0 = rowInv[statRow];
        float x1 = rowMax[statRow + 8];
        float i1 = rowInv[statRow + 8];
        float[] o0 = ctx.mmaFragment(0.0f);
        float[] o1 = ctx.mmaFragment(0.0f);
        float[] o2 = ctx.mmaFragment(0.0f);
        float[] o3 = ctx.mmaFragment(0.0f);
        float[] o4 = ctx.mmaFragment(0.0f);
        float[] o5 = ctx.mmaFragment(0.0f);
        float[] o6 = ctx.mmaFragment(0.0f);
        float[] o7 = ctx.mmaFragment(0.0f);
        float[] o8 = ctx.mmaFragment(0.0f);
        float[] o9 = ctx.mmaFragment(0.0f);
        float[] o10 = ctx.mmaFragment(0.0f);
        float[] o11 = ctx.mmaFragment(0.0f);
        float[] o12 = ctx.mmaFragment(0.0f);
        float[] o13 = ctx.mmaFragment(0.0f);
        float[] o14 = ctx.mmaFragment(0.0f);
        float[] o15 = ctx.mmaFragment(0.0f);
        // The warp's probabilities in FP16, 16 rows of 64 keys, rows 72 halves apart.
        int pRow0 = warp * 576 + (lane >> 2) * 36 + quad;
        int pRow1 = pRow0 + 288;
        int pBytes = warp * 2304;
        for (int kb = 0; kb < blocks; kb++) {
            int key0 = kb * 64;
            int kSrc = gBase + kb * 8192;
            for (int i = tid; i < 1024; i += 128) {
                ctx.asyncCopyToLocal16(kTile, 4 * i, keysT, kSrc + 8 * i);
                ctx.asyncCopyToLocal16(vTile, 4 * i, values, kSrc + 8 * i);
            }
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();
            float[] s0 = ctx.mmaFragment(0.0f);
            float[] s1 = ctx.mmaFragment(0.0f);
            float[] s2 = ctx.mmaFragment(0.0f);
            float[] s3 = ctx.mmaFragment(0.0f);
            float[] s4 = ctx.mmaFragment(0.0f);
            float[] s5 = ctx.mmaFragment(0.0f);
            float[] s6 = ctx.mmaFragment(0.0f);
            float[] s7 = ctx.mmaFragment(0.0f);
            s0 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 0), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 2048), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 4096), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 6144), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 8192), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 10240), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 12288), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 14336), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 256), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 2304), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 4352), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 6400), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 8448), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 10496), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 12544), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 14592), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 512), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 2560), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 4608), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 6656), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 8704), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 10752), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 12800), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 14848), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 768), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 2816), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 4864), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 6912), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 8960), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 11008), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 13056), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 15104), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 1024), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 3072), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 5120), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 7168), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 9216), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 11264), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 13312), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q4, ctx.mmaLoadB(kTile, 16, 15360), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 1280), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 3328), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 5376), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 7424), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 9472), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 11520), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 13568), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q5, ctx.mmaLoadB(kTile, 16, 15616), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 1536), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 3584), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 5632), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 7680), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 9728), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 11776), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 13824), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q6, ctx.mmaLoadB(kTile, 16, 15872), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 1792), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 3840), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 5888), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 7936), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 9984), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 12032), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 14080), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q7, ctx.mmaLoadB(kTile, 16, 16128), s7, MMAShape.M16N8K16);
            int c = key0 + 2 * quad;
            float w00 = c <= p0 ? TornadoMath.exp(s0[0] * scale - x0) * i0 : 0.0f;
            float w01 = c + 1 <= p0 ? TornadoMath.exp(s0[1] * scale - x0) * i0 : 0.0f;
            float w02 = c <= p1 ? TornadoMath.exp(s0[2] * scale - x1) * i1 : 0.0f;
            float w03 = c + 1 <= p1 ? TornadoMath.exp(s0[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 0] =
                    (new HalfFloat(w00).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w01).getHalfFloatValue() << 16);
            pTile[pRow1 + 0] =
                    (new HalfFloat(w02).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w03).getHalfFloatValue() << 16);
            float w10 = c + 8 <= p0 ? TornadoMath.exp(s1[0] * scale - x0) * i0 : 0.0f;
            float w11 = c + 9 <= p0 ? TornadoMath.exp(s1[1] * scale - x0) * i0 : 0.0f;
            float w12 = c + 8 <= p1 ? TornadoMath.exp(s1[2] * scale - x1) * i1 : 0.0f;
            float w13 = c + 9 <= p1 ? TornadoMath.exp(s1[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 4] =
                    (new HalfFloat(w10).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w11).getHalfFloatValue() << 16);
            pTile[pRow1 + 4] =
                    (new HalfFloat(w12).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w13).getHalfFloatValue() << 16);
            float w20 = c + 16 <= p0 ? TornadoMath.exp(s2[0] * scale - x0) * i0 : 0.0f;
            float w21 = c + 17 <= p0 ? TornadoMath.exp(s2[1] * scale - x0) * i0 : 0.0f;
            float w22 = c + 16 <= p1 ? TornadoMath.exp(s2[2] * scale - x1) * i1 : 0.0f;
            float w23 = c + 17 <= p1 ? TornadoMath.exp(s2[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 8] =
                    (new HalfFloat(w20).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w21).getHalfFloatValue() << 16);
            pTile[pRow1 + 8] =
                    (new HalfFloat(w22).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w23).getHalfFloatValue() << 16);
            float w30 = c + 24 <= p0 ? TornadoMath.exp(s3[0] * scale - x0) * i0 : 0.0f;
            float w31 = c + 25 <= p0 ? TornadoMath.exp(s3[1] * scale - x0) * i0 : 0.0f;
            float w32 = c + 24 <= p1 ? TornadoMath.exp(s3[2] * scale - x1) * i1 : 0.0f;
            float w33 = c + 25 <= p1 ? TornadoMath.exp(s3[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 12] =
                    (new HalfFloat(w30).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w31).getHalfFloatValue() << 16);
            pTile[pRow1 + 12] =
                    (new HalfFloat(w32).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w33).getHalfFloatValue() << 16);
            float w40 = c + 32 <= p0 ? TornadoMath.exp(s4[0] * scale - x0) * i0 : 0.0f;
            float w41 = c + 33 <= p0 ? TornadoMath.exp(s4[1] * scale - x0) * i0 : 0.0f;
            float w42 = c + 32 <= p1 ? TornadoMath.exp(s4[2] * scale - x1) * i1 : 0.0f;
            float w43 = c + 33 <= p1 ? TornadoMath.exp(s4[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 16] =
                    (new HalfFloat(w40).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w41).getHalfFloatValue() << 16);
            pTile[pRow1 + 16] =
                    (new HalfFloat(w42).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w43).getHalfFloatValue() << 16);
            float w50 = c + 40 <= p0 ? TornadoMath.exp(s5[0] * scale - x0) * i0 : 0.0f;
            float w51 = c + 41 <= p0 ? TornadoMath.exp(s5[1] * scale - x0) * i0 : 0.0f;
            float w52 = c + 40 <= p1 ? TornadoMath.exp(s5[2] * scale - x1) * i1 : 0.0f;
            float w53 = c + 41 <= p1 ? TornadoMath.exp(s5[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 20] =
                    (new HalfFloat(w50).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w51).getHalfFloatValue() << 16);
            pTile[pRow1 + 20] =
                    (new HalfFloat(w52).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w53).getHalfFloatValue() << 16);
            float w60 = c + 48 <= p0 ? TornadoMath.exp(s6[0] * scale - x0) * i0 : 0.0f;
            float w61 = c + 49 <= p0 ? TornadoMath.exp(s6[1] * scale - x0) * i0 : 0.0f;
            float w62 = c + 48 <= p1 ? TornadoMath.exp(s6[2] * scale - x1) * i1 : 0.0f;
            float w63 = c + 49 <= p1 ? TornadoMath.exp(s6[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 24] =
                    (new HalfFloat(w60).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w61).getHalfFloatValue() << 16);
            pTile[pRow1 + 24] =
                    (new HalfFloat(w62).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w63).getHalfFloatValue() << 16);
            float w70 = c + 56 <= p0 ? TornadoMath.exp(s7[0] * scale - x0) * i0 : 0.0f;
            float w71 = c + 57 <= p0 ? TornadoMath.exp(s7[1] * scale - x0) * i0 : 0.0f;
            float w72 = c + 56 <= p1 ? TornadoMath.exp(s7[2] * scale - x1) * i1 : 0.0f;
            float w73 = c + 57 <= p1 ? TornadoMath.exp(s7[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 28] =
                    (new HalfFloat(w70).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w71).getHalfFloatValue() << 16);
            pTile[pRow1 + 28] =
                    (new HalfFloat(w72).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w73).getHalfFloatValue() << 16);
            ctx.localBarrier();
            HalfFloat[] pa0 = ctx.mmaLoadA(pTile, 72, pBytes + 0);
            o0 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 0), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 1024), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 2048), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 3072), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 4096), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 5120), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 6144), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 7168), o7, MMAShape.M16N8K16);
            o8 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 8192), o8, MMAShape.M16N8K16);
            o9 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 9216), o9, MMAShape.M16N8K16);
            o10 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 10240), o10, MMAShape.M16N8K16);
            o11 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 11264), o11, MMAShape.M16N8K16);
            o12 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 12288), o12, MMAShape.M16N8K16);
            o13 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 13312), o13, MMAShape.M16N8K16);
            o14 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 14336), o14, MMAShape.M16N8K16);
            o15 = ctx.mma(pa0, ctx.mmaLoadB(vTile, 16, 15360), o15, MMAShape.M16N8K16);
            HalfFloat[] pa1 = ctx.mmaLoadA(pTile, 72, pBytes + 32);
            o0 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 256), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 1280), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 2304), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 3328), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 4352), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 5376), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 6400), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 7424), o7, MMAShape.M16N8K16);
            o8 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 8448), o8, MMAShape.M16N8K16);
            o9 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 9472), o9, MMAShape.M16N8K16);
            o10 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 10496), o10, MMAShape.M16N8K16);
            o11 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 11520), o11, MMAShape.M16N8K16);
            o12 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 12544), o12, MMAShape.M16N8K16);
            o13 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 13568), o13, MMAShape.M16N8K16);
            o14 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 14592), o14, MMAShape.M16N8K16);
            o15 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 15616), o15, MMAShape.M16N8K16);
            HalfFloat[] pa2 = ctx.mmaLoadA(pTile, 72, pBytes + 64);
            o0 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 512), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 1536), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 2560), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 3584), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 4608), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 5632), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 6656), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 7680), o7, MMAShape.M16N8K16);
            o8 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 8704), o8, MMAShape.M16N8K16);
            o9 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 9728), o9, MMAShape.M16N8K16);
            o10 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 10752), o10, MMAShape.M16N8K16);
            o11 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 11776), o11, MMAShape.M16N8K16);
            o12 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 12800), o12, MMAShape.M16N8K16);
            o13 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 13824), o13, MMAShape.M16N8K16);
            o14 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 14848), o14, MMAShape.M16N8K16);
            o15 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 15872), o15, MMAShape.M16N8K16);
            HalfFloat[] pa3 = ctx.mmaLoadA(pTile, 72, pBytes + 96);
            o0 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 768), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 1792), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 2816), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 3840), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 4864), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 5888), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 6912), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 7936), o7, MMAShape.M16N8K16);
            o8 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 8960), o8, MMAShape.M16N8K16);
            o9 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 9984), o9, MMAShape.M16N8K16);
            o10 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 11008), o10, MMAShape.M16N8K16);
            o11 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 12032), o11, MMAShape.M16N8K16);
            o12 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 13056), o12, MMAShape.M16N8K16);
            o13 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 14080), o13, MMAShape.M16N8K16);
            o14 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 15104), o14, MMAShape.M16N8K16);
            o15 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 16128), o15, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int ob0 = r0 * qDim + qCol + 2 * quad;
        int ob1 = ob0 + 8 * qDim;
        boolean v0 = r0 < tokens;
        boolean v1 = r0 + 8 < tokens;
        // The value goes through a local: TC drops a conditional written inside new HalfFloat(...).
        float y00 = v0 ? o0[0] : 0.0f;
        out.set(ob0, new HalfFloat(y00));
        float y01 = v0 ? o0[1] : 0.0f;
        out.set(ob0 + 1, new HalfFloat(y01));
        float y02 = v1 ? o0[2] : 0.0f;
        out.set(ob1, new HalfFloat(y02));
        float y03 = v1 ? o0[3] : 0.0f;
        out.set(ob1 + 1, new HalfFloat(y03));
        float y10 = v0 ? o1[0] : 0.0f;
        out.set(ob0 + 8, new HalfFloat(y10));
        float y11 = v0 ? o1[1] : 0.0f;
        out.set(ob0 + 8 + 1, new HalfFloat(y11));
        float y12 = v1 ? o1[2] : 0.0f;
        out.set(ob1 + 8, new HalfFloat(y12));
        float y13 = v1 ? o1[3] : 0.0f;
        out.set(ob1 + 8 + 1, new HalfFloat(y13));
        float y20 = v0 ? o2[0] : 0.0f;
        out.set(ob0 + 16, new HalfFloat(y20));
        float y21 = v0 ? o2[1] : 0.0f;
        out.set(ob0 + 16 + 1, new HalfFloat(y21));
        float y22 = v1 ? o2[2] : 0.0f;
        out.set(ob1 + 16, new HalfFloat(y22));
        float y23 = v1 ? o2[3] : 0.0f;
        out.set(ob1 + 16 + 1, new HalfFloat(y23));
        float y30 = v0 ? o3[0] : 0.0f;
        out.set(ob0 + 24, new HalfFloat(y30));
        float y31 = v0 ? o3[1] : 0.0f;
        out.set(ob0 + 24 + 1, new HalfFloat(y31));
        float y32 = v1 ? o3[2] : 0.0f;
        out.set(ob1 + 24, new HalfFloat(y32));
        float y33 = v1 ? o3[3] : 0.0f;
        out.set(ob1 + 24 + 1, new HalfFloat(y33));
        float y40 = v0 ? o4[0] : 0.0f;
        out.set(ob0 + 32, new HalfFloat(y40));
        float y41 = v0 ? o4[1] : 0.0f;
        out.set(ob0 + 32 + 1, new HalfFloat(y41));
        float y42 = v1 ? o4[2] : 0.0f;
        out.set(ob1 + 32, new HalfFloat(y42));
        float y43 = v1 ? o4[3] : 0.0f;
        out.set(ob1 + 32 + 1, new HalfFloat(y43));
        float y50 = v0 ? o5[0] : 0.0f;
        out.set(ob0 + 40, new HalfFloat(y50));
        float y51 = v0 ? o5[1] : 0.0f;
        out.set(ob0 + 40 + 1, new HalfFloat(y51));
        float y52 = v1 ? o5[2] : 0.0f;
        out.set(ob1 + 40, new HalfFloat(y52));
        float y53 = v1 ? o5[3] : 0.0f;
        out.set(ob1 + 40 + 1, new HalfFloat(y53));
        float y60 = v0 ? o6[0] : 0.0f;
        out.set(ob0 + 48, new HalfFloat(y60));
        float y61 = v0 ? o6[1] : 0.0f;
        out.set(ob0 + 48 + 1, new HalfFloat(y61));
        float y62 = v1 ? o6[2] : 0.0f;
        out.set(ob1 + 48, new HalfFloat(y62));
        float y63 = v1 ? o6[3] : 0.0f;
        out.set(ob1 + 48 + 1, new HalfFloat(y63));
        float y70 = v0 ? o7[0] : 0.0f;
        out.set(ob0 + 56, new HalfFloat(y70));
        float y71 = v0 ? o7[1] : 0.0f;
        out.set(ob0 + 56 + 1, new HalfFloat(y71));
        float y72 = v1 ? o7[2] : 0.0f;
        out.set(ob1 + 56, new HalfFloat(y72));
        float y73 = v1 ? o7[3] : 0.0f;
        out.set(ob1 + 56 + 1, new HalfFloat(y73));
        float y80 = v0 ? o8[0] : 0.0f;
        out.set(ob0 + 64, new HalfFloat(y80));
        float y81 = v0 ? o8[1] : 0.0f;
        out.set(ob0 + 64 + 1, new HalfFloat(y81));
        float y82 = v1 ? o8[2] : 0.0f;
        out.set(ob1 + 64, new HalfFloat(y82));
        float y83 = v1 ? o8[3] : 0.0f;
        out.set(ob1 + 64 + 1, new HalfFloat(y83));
        float y90 = v0 ? o9[0] : 0.0f;
        out.set(ob0 + 72, new HalfFloat(y90));
        float y91 = v0 ? o9[1] : 0.0f;
        out.set(ob0 + 72 + 1, new HalfFloat(y91));
        float y92 = v1 ? o9[2] : 0.0f;
        out.set(ob1 + 72, new HalfFloat(y92));
        float y93 = v1 ? o9[3] : 0.0f;
        out.set(ob1 + 72 + 1, new HalfFloat(y93));
        float y100 = v0 ? o10[0] : 0.0f;
        out.set(ob0 + 80, new HalfFloat(y100));
        float y101 = v0 ? o10[1] : 0.0f;
        out.set(ob0 + 80 + 1, new HalfFloat(y101));
        float y102 = v1 ? o10[2] : 0.0f;
        out.set(ob1 + 80, new HalfFloat(y102));
        float y103 = v1 ? o10[3] : 0.0f;
        out.set(ob1 + 80 + 1, new HalfFloat(y103));
        float y110 = v0 ? o11[0] : 0.0f;
        out.set(ob0 + 88, new HalfFloat(y110));
        float y111 = v0 ? o11[1] : 0.0f;
        out.set(ob0 + 88 + 1, new HalfFloat(y111));
        float y112 = v1 ? o11[2] : 0.0f;
        out.set(ob1 + 88, new HalfFloat(y112));
        float y113 = v1 ? o11[3] : 0.0f;
        out.set(ob1 + 88 + 1, new HalfFloat(y113));
        float y120 = v0 ? o12[0] : 0.0f;
        out.set(ob0 + 96, new HalfFloat(y120));
        float y121 = v0 ? o12[1] : 0.0f;
        out.set(ob0 + 96 + 1, new HalfFloat(y121));
        float y122 = v1 ? o12[2] : 0.0f;
        out.set(ob1 + 96, new HalfFloat(y122));
        float y123 = v1 ? o12[3] : 0.0f;
        out.set(ob1 + 96 + 1, new HalfFloat(y123));
        float y130 = v0 ? o13[0] : 0.0f;
        out.set(ob0 + 104, new HalfFloat(y130));
        float y131 = v0 ? o13[1] : 0.0f;
        out.set(ob0 + 104 + 1, new HalfFloat(y131));
        float y132 = v1 ? o13[2] : 0.0f;
        out.set(ob1 + 104, new HalfFloat(y132));
        float y133 = v1 ? o13[3] : 0.0f;
        out.set(ob1 + 104 + 1, new HalfFloat(y133));
        float y140 = v0 ? o14[0] : 0.0f;
        out.set(ob0 + 112, new HalfFloat(y140));
        float y141 = v0 ? o14[1] : 0.0f;
        out.set(ob0 + 112 + 1, new HalfFloat(y141));
        float y142 = v1 ? o14[2] : 0.0f;
        out.set(ob1 + 112, new HalfFloat(y142));
        float y143 = v1 ? o14[3] : 0.0f;
        out.set(ob1 + 112 + 1, new HalfFloat(y143));
        float y150 = v0 ? o15[0] : 0.0f;
        out.set(ob0 + 120, new HalfFloat(y150));
        float y151 = v0 ? o15[1] : 0.0f;
        out.set(ob0 + 120 + 1, new HalfFloat(y151));
        float y152 = v1 ? o15[2] : 0.0f;
        out.set(ob1 + 120, new HalfFloat(y152));
        float y153 = v1 ? o15[3] : 0.0f;
        out.set(ob1 + 120 + 1, new HalfFloat(y153));
    }
}
