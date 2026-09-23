package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The batched-prefill kernels this family does not share with any other.
 *
 * <p>Every kernel here is the chunk-wide form of one {@link Gemma4Kernels} kernel: the same
 * arithmetic in the same order, with a row index added. A row is one prompt token; {@code
 * startPosHolder} carries the chunk's first sequence position at index 0 and the number of
 * <i>real</i> rows at index 1, because the grids launch a padded row count and the padding rows
 * must not rotate, must not write a key or a value, and must not address past the end of a layer's
 * KV slice.
 *
 * <p>They are a separate class rather than additions to {@link TransformerBatchPrefillKernels}
 * because the differences are this architecture's and not parameters: sandwich norms that normalize
 * a branch output and add it onto the residual, a per-head norm on the query, the key <i>and</i>
 * the value, a GeGLU where every other family has SwiGLU, an attention scale of exactly one, a
 * sliding window on four layers out of five, and the per-layer embedding block that has no
 * counterpart anywhere else in the repository. The generic kernels that <i>do</i> fit — the RMS
 * reductions, the FP32→FP16 cast and the whole {@code gemmMMA} family — are called from the layer
 * graph unchanged, not copied here.
 */
// @formatter:on
public final class Gemma4BatchPrefillKernels {

    private Gemma4BatchPrefillKernels() {}

    // @formatter:off
    /**
     * Dimensions per staged key tile.
     *
     * <p>Sixteen, and the reason is occupancy rather than coalescing. Thirty-two is the width that
     * makes a warp's load one 128-byte sector, and that is what this tile was first built with —
     * but the tile is then 32 × 129 floats, 16.5 KB, 87% of the block's shared memory, and Nsight
     * put the kernel's theoretical occupancy at 41.7% <i>limited by the required amount of shared
     * memory</i>. It also put DRAM throughput at 0.70% with a 99.55% L2 hit rate: the keys are in
     * cache, so what a wider tile buys in transaction shape it more than loses in warps resident.
     * Sixteen halves the tile and roughly doubles the occupancy; eight halves it again and measured
     * slightly worse, so this is a measured optimum and not a rounding.
     */
    // @formatter:on
    private static final int DIM_TILE = Integer.getInteger("jitllm.gemma4.dimTile", 16);

    // ── Norms ────────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Sandwich norm with residual, chunk-wide: {@code x[b,i] += weight[i] * (scale[b] *
     * delta[b,i])}.
     *
     * <p>The row-wise form of {@link Gemma4Kernels#rmsNormApplyWithResidual}, and the reason this
     * family cannot use {@code batchedRmsReduceFusedResidual}: the scale is the RMS of the branch
     * output {@code delta}, not of the residual {@code x} it is added to, so the reduce runs over a
     * different buffer than the add updates and the two cannot be one pass.
     *
     * <p>Worker: {@code B*size} threads, local 256.
     */
    // @formatter:on
    public static void batchedRmsApplyWithResidual(
            KernelContext context,
            FloatArray x,
            FloatArray delta,
            FloatArray weight,
            FloatArray scaleBatch,
            int size) {
        int gid = context.globalIdx;
        int b = gid / size;
        int i = gid - b * size;
        float scale = scaleBatch.get(b);
        x.set(gid, x.get(gid) + weight.get(i) * (scale * delta.get(gid)));
    }

    // @formatter:off
    /**
     * The query, key and value per-head norms over the packed {@code [q|k|v]} projection output, in
     * one launch.
     *
     * <p>One workgroup per (row, head slot), where the slots run {@code q} heads, then {@code k}
     * heads, then {@code v} heads — which is exactly the packed row's own order, so a slot's base
     * is {@code slot * headDim} from the row start and no per-kind branch on the address is needed.
     * Only the weight differs by kind, and Gemma 4 normalizes V without one; that is the whole
     * reason this is a single kernel with a three-way select rather than three launches.
     *
     * <p>Worker: {@code B*(nHeads + 2*nHeadKv)} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedQkvHeadNorms(
            KernelContext context,
            FloatArray qkv,
            FloatArray qNormWeight,
            FloatArray kNormWeight,
            int nHeads,
            int nHeadKv,
            int headDim,
            int qkvStride,
            int localMemSize,
            float rmsNormEps) {
        int group = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        int slotsPerRow = nHeads + 2 * nHeadKv;
        int b = group / slotsPerRow;
        int slot = group - b * slotsPerRow;
        int base = b * qkvStride + slot * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = qkv.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();

        if (slot < nHeads) {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, qNormWeight.get(i) * (ss * qkv.get(base + i)));
            }
        } else if (slot < nHeads + nHeadKv) {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, kNormWeight.get(i) * (ss * qkv.get(base + i)));
            }
        } else {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, ss * qkv.get(base + i));
            }
        }
    }

    // @formatter:off
    /**
     * The query-only per-head norm, for the twenty layers that reuse an earlier layer's KV and
     * therefore project no key and no value.
     *
     * <p>Worker: {@code B*nHeads} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedQHeadNorm(
            KernelContext context,
            FloatArray qkv,
            FloatArray qNormWeight,
            int nHeads,
            int headDim,
            int qkvStride,
            int localMemSize,
            float rmsNormEps) {
        int group = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int base = b * qkvStride + h * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = qkv.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            qkv.set(base + i, qNormWeight.get(i) * (ss * qkv.get(base + i)));
        }
    }

    // ── RoPE and the KV cache ────────────────────────────────────────────────

    // @formatter:off
    /**
     * NeoX RoPE over the packed {@code [q|k|v]} rows, fused with the key/value cache write.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#ropeNeoxRotateAndCacheCopy}, reading its angle
     * from the same precomputed tables at {@code startPos + b} rather than at a single position.
     * The cache is this family's flat one addressed by {@code cacheBaseOffset}, not a paged one:
     * the decode graphs that run after prefill read it that way, and a prefill that wrote pages
     * would leave them reading a cache nobody filled.
     *
     * <p>Padding rows return before writing anything. A padded row's position is past the chunk and
     * would be a valid index into the next layer's KV slice, so the guard is what keeps a padded
     * launch from corrupting a neighbouring layer rather than merely wasting work.
     *
     * <p>Worker: {@code B*nHeads*(headDim/2)} threads.
     */
    // @formatter:on
    public static void batchedRopeAndCache(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeads,
            int nHeadKv,
            int headDim,
            int kvDim,
            int qkvStride,
            int cacheBaseOffset) {
        int gid = context.globalIdx;
        int half = headDim / 2;
        int perRow = nHeads * half;
        int b = gid / perRow;
        int rem = gid - b * perRow;
        int h = rem / half;
        int ic = rem - h * half;

        if (b >= startPosHolder.get(1)) {
            return;
        }
        int pos = startPosHolder.get(0) + b;
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int rowBase = b * qkvStride;
        int qBase = rowBase + h * headDim;
        float v0q = qkv.get(qBase + ic);
        float v1q = qkv.get(qBase + ic + half);
        qkv.set(qBase + ic, v0q * fcr - v1q * fci);
        qkv.set(qBase + ic + half, v0q * fci + v1q * fcr);

        if (h < nHeadKv) {
            int kBase = rowBase + nHeads * headDim + h * headDim;
            int vBase = kBase + nHeadKv * headDim;
            float v0k = qkv.get(kBase + ic);
            float v1k = qkv.get(kBase + ic + half);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            int cacheOffset = cacheBaseOffset + pos * kvDim + h * headDim;
            keyCache.set(cacheOffset + ic, rotatedK0);
            keyCache.set(cacheOffset + ic + half, rotatedK1);
            valueCache.set(cacheOffset + ic, qkv.get(vBase + ic));
            valueCache.set(cacheOffset + ic + half, qkv.get(vBase + ic + half));
        }
    }

    // @formatter:off
    /**
     * NeoX RoPE on the query alone, for the layers that reuse an earlier layer's KV cache.
     *
     * <p>Worker: {@code B*nHeads*(headDim/2)} threads.
     */
    // @formatter:on
    public static void batchedRopeQOnly(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeads,
            int headDim,
            int qkvStride) {
        int gid = context.globalIdx;
        int half = headDim / 2;
        int perRow = nHeads * half;
        int b = gid / perRow;
        int rem = gid - b * perRow;
        int h = rem / half;
        int ic = rem - h * half;

        if (b >= startPosHolder.get(1)) {
            return;
        }
        int pos = startPosHolder.get(0) + b;
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int qBase = b * qkvStride + h * headDim;
        float v0 = qkv.get(qBase + ic);
        float v1 = qkv.get(qBase + ic + half);
        qkv.set(qBase + ic, v0 * fcr - v1 * fci);
        qkv.set(qBase + ic + half, v0 * fci + v1 * fcr);
    }

    // ── Attention ────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Causal attention over a (possibly sliding) window, one workgroup per (row, head), with the
     * result emitted in FP16 for the output projection's tensor-core GEMM.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#attentionWithSlidingWindowParallel}: three
     * phases, the score dot product and the weighted sum each serial and in dimension order, only
     * the softmax maximum and the sum of exponentials as trees. Row {@code b} attends positions
     * {@code [max(0, pos-windowSize+1), pos]} of the shared sequence cache with {@code pos =
     * startPos + b}, which is what makes the chunk causal without a mask: a row simply never reads
     * past its own position. Gemma 4's attention scale is one, so no score is divided.
     *
     * <p>The scores go to a global scratch rather than to shared memory because the window is up to
     * the whole context and a workgroup's shared memory is not. Each (row, head) owns {@code
     * scoreStride} floats of it and writes only inside its own slice.
     *
     * <p>Padding rows write zeros into their output and return: those rows still pass through the
     * output projection, and an uninitialized FP16 row would put NaNs into a GEMM that shares no
     * lanes with the real rows but does share its warps' execution.
     *
     * <p>Worker: {@code B*nHeads} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttention(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray out,
            FloatArray scores,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize,
            int scoreStride,
            int localMemSize) {
        int tid = context.localIdx;
        int group = context.groupIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int outBase = b * (nHeads * headDim) + h * headDim;

        if (b >= startPosHolder.get(1)) {
            for (int i = tid; i < headDim; i += localSize) {
                out.set(outBase + i, new HalfFloat(0.0f));
            }
            return;
        }

        int pos = startPosHolder.get(0) + b;
        int windowStart = Math.max(0, pos - windowSize + 1);
        int scoreBase = group * scoreStride;
        int kvHeadIdx = h / kvMul;
        int qOffset = b * qkvStride + h * headDim;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = qkv.get(qOffset + i);
        }
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            int keyOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
            float score = 0.0f;
            for (int i = 0; i < headDim; i++) {
                score += qShared[i] * keyCache.get(keyOffset + i);
            }
            scores.set(scoreBase + (t - windowStart), score);
        }
        context.localBarrier();

        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float v = scores.get(scoreBase + (t - windowStart));
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float maxScore = reduce[0];
        context.localBarrier();

        float localSum = 0.0f;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float e = TornadoMath.exp(scores.get(scoreBase + (t - windowStart)) - maxScore);
            scores.set(scoreBase + (t - windowStart), e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sum = reduce[0];
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            scores.set(
                    scoreBase + (t - windowStart),
                    scores.get(scoreBase + (t - windowStart)) * normFactor);
        }
        context.localBarrier();

        for (int i = tid; i < headDim; i += localSize) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum +=
                        scores.get(scoreBase + (t - windowStart)) * valueCache.get(valueOffset + i);
            }
            out.set(outBase + i, new HalfFloat(weightedSum));
        }
    }

    // @formatter:off
    /**
     * The same attention with the key tile staged through shared memory, coalesced.
     *
     * <p>Only the score phase differs from {@link #batchedSlidingWindowAttention}. There, one lane
     * owns one position and walks {@code headDim} straight out of global memory, so at a fixed
     * dimension consecutive lanes are {@code kvDim} floats apart and every load is its own sector.
     * Here the phase walks {@code headDim} in tiles of {@value #DIM_TILE}: a tile is loaded with
     * consecutive lanes reading consecutive <i>dimensions</i> of one position, which is contiguous,
     * and stored transposed at {@code keyTile[d * (lanes + 1) + p]} so that the read back — every
     * lane taking its own position at a fixed dimension — strides by one and hits no bank twice.
     *
     * <p><b>Bit-identical to the kernel it replaces.</b> A lane still owns the same position and
     * still accumulates that position's dot product over dimensions in increasing order — the tiles
     * are in order and the dimensions within a tile are in order — so it is the same sum of the
     * same products in the same sequence. The maximum, the sum of exponentials, the normalisation
     * and the whole value pass are untouched. This is a change to where the operands are read from,
     * and to nothing else.
     *
     * <p>The tile barriers sit outside the {@code t <= pos} guard on purpose: the lanes whose
     * position is past the end of the window still have to reach them, and a barrier inside a
     * divergent branch is the classic way to hang a workgroup rather than to skip work in it.
     *
     * <p>Worker: {@code B*nHeads} workgroups of {@code localMemSize} lanes, as before.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttentionStaged(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray out,
            FloatArray scores,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize,
            int scoreStride,
            int localMemSize) {
        int tid = context.localIdx;
        int group = context.groupIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int outBase = b * (nHeads * headDim) + h * headDim;

        if (b >= startPosHolder.get(1)) {
            for (int i = tid; i < headDim; i += localSize) {
                out.set(outBase + i, new HalfFloat(0.0f));
            }
            return;
        }

        int pos = startPosHolder.get(0) + b;
        int windowStart = Math.max(0, pos - windowSize + 1);
        int scoreBase = group * scoreStride;
        int kvHeadIdx = h / kvMul;
        int qOffset = b * qkvStride + h * headDim;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);
        float[] keyTile = context.allocateFloatLocalArray(DIM_TILE * (localMemSize + 1));

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = qkv.get(qOffset + i);
        }
        context.localBarrier();

        int tileStride = localSize + 1;
        int tileElements = DIM_TILE * localSize;
        for (int tBase = windowStart; tBase <= pos; tBase += localSize) {
            int t = tBase + tid;
            float score = 0.0f;
            for (int d0 = 0; d0 < headDim; d0 += DIM_TILE) {
                context.localBarrier();
                for (int idx = tid; idx < tileElements; idx += localSize) {
                    int d = idx % DIM_TILE;
                    int p = idx / DIM_TILE;
                    int tt = tBase + p;
                    float v =
                            (tt <= pos)
                                    ? keyCache.get(
                                            cacheBaseOffset
                                                    + tt * kvDim
                                                    + kvHeadIdx * headDim
                                                    + d0
                                                    + d)
                                    : 0.0f;
                    keyTile[d * tileStride + p] = v;
                }
                context.localBarrier();
                if (t <= pos) {
                    for (int d = 0; d < DIM_TILE; d++) {
                        score += qShared[d0 + d] * keyTile[d * tileStride + tid];
                    }
                }
            }
            if (t <= pos) {
                scores.set(scoreBase + (t - windowStart), score);
            }
        }
        context.localBarrier();

        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float v = scores.get(scoreBase + (t - windowStart));
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float maxScore = reduce[0];
        context.localBarrier();

        float localSum = 0.0f;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float e = TornadoMath.exp(scores.get(scoreBase + (t - windowStart)) - maxScore);
            scores.set(scoreBase + (t - windowStart), e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sum = reduce[0];
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            scores.set(
                    scoreBase + (t - windowStart),
                    scores.get(scoreBase + (t - windowStart)) * normFactor);
        }
        context.localBarrier();

        for (int i = tid; i < headDim; i += localSize) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum +=
                        scores.get(scoreBase + (t - windowStart)) * valueCache.get(valueOffset + i);
            }
            out.set(outBase + i, new HalfFloat(weightedSum));
        }
    }

    // ── The split-K projection ───────────────────────────────────────────────

    private static final int WARP_SIZE = 32;
    private static final int BM = 128, BN = 128, BK = 16;
    private static final int WARPS_N = 2;
    private static final int WM = 32, WN = 64;
    private static final int B_SUBTILE_BYTES = 256;

    /** Two consecutive FP16 values in one int, for the shared ldmatrix tiles. */
    private static int packHalves(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    /** Two vertically adjacent Q8_0 weights at depth k, dequantized, in one int. */
    private static int packQ8Halves(ByteArray w, int col, int k, int blocksPerRow) {
        int kBlock = k >>> 5;
        int kIn = k & 31;
        int off0 = (col * blocksPerRow + kBlock) * 34;
        int off1 = off0 + blocksPerRow * 34;
        float v0 = w.getHalfFloat(off0).getFloat32() * w.get(off0 + 2 + kIn);
        float v1 = w.getHalfFloat(off1).getFloat32() * w.get(off1 + 2 + kIn);
        return TransformerBatchPrefillKernels.fp16BitsOf(v0)
                | (TransformerBatchPrefillKernels.fp16BitsOf(v1) << 16);
    }

    // @formatter:off
    /**
     * A Q8_0 weight matrix decoded into FP16, one element per thread.
     *
     * <p><b>Why this exists.</b> The Q8_0 tensor-core GEMMs decode their weights inside the K-loop,
     * and TornadoVM can only reach the hardware float-to-half conversion through a <i>store</i> to
     * a half array — as a value in a register it does not lower at all ({@code address origin
     * unimplemented: MulNode}). So those kernels convert in software, by binary search on the
     * exponent, eight times per lane per K-step. It shows in the emitted CUDA: {@code
     * gemmMMAGateUpQ8} is 8,666 lines carrying sixteen {@code mma.sync}, against 1,179 lines for
     * the FP16 {@code gemmMMA} that does the same arithmetic. The GEMM is not doing matrix
     * multiplication; it is doing float-to-half conversion with a little matrix multiplication
     * attached.
     *
     * <p>Decoding into a scratch first is a store, so the conversion is the one hardware
     * instruction it should be, and the GEMM that follows stages FP16 operands with nothing but
     * integer packing. The cost is the scratch traffic, paid once per chunk per layer against a
     * GEMM that reads the same weights for every one of the chunk's rows.
     *
     * <p>Bit-identical to what the Q8_0 GEMM computes for the same element: the same product of the
     * same block scale and the same quant, rounded to half once, round-to-nearest-even both ways.
     *
     * <p>{@code destOffset} places a matrix inside a larger scratch, which is what lets the gate
     * and the up projection share one buffer and one GEMM.
     *
     * <p>Worker: one thread per element, local 256. Requires the row length to be a whole number of
     * 32-weight blocks, which every projection this family has is.
     */
    // @formatter:on
    public static void dequantizeQ8ToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset) {
        int gid = context.globalIdx;
        int blk = gid >>> 5;
        int within = gid & 31;
        int off = blk * 34;
        out.set(
                destOffset + gid,
                new HalfFloat(w.getHalfFloat(off).getFloat32() * w.get(off + 2 + within)));
    }

    // @formatter:off
    /**
     * The Q4_0 twin of {@link #dequantizeQ8ToFP16}: 18 bytes to thirty-two weights, an FP16 block
     * scale then sixteen bytes of packed nibbles, the unsigned nibble recentred by eight.
     *
     * <p>Elements 0-15 of a block are the low nibbles of the sixteen bytes and 16-31 the high ones,
     * which is the convention {@link TransformerComputeKernelsQ4_0#decode} reads and the same one
     * the host-side embedding decode uses. Decoding to the same value as that kernel is what makes
     * this interchangeable with it.
     *
     * <p>Worker: one thread per element, local 256.
     */
    // @formatter:on
    public static void dequantizeQ4_0ToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset) {
        int gid = context.globalIdx;
        int blk = gid >>> 5;
        int within = gid & 31;
        int off = blk * 18;
        int half = within >>> 4;
        int byteIndex = within - (half << 4);
        int packed = w.get(off + 2 + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        out.set(destOffset + gid, new HalfFloat(w.getHalfFloat(off).getFloat32() * (q - 8)));
    }

    // @formatter:off
    /**
     * The Q4_1 twin: 20 bytes to thirty-two weights, an FP16 scale and an FP16 minimum then the
     * same sixteen packed bytes, and {@code scale * q + minimum} with no recentring.
     *
     * <p>This family needs it for four tensors and no more: {@code ffn_down} is Q4_1 on blocks 0-3
     * of the Q4_0 file and Q4_0 on the other thirty-one. A projection's kernel comes from the
     * tensor's own representation for exactly that reason — a model-wide answer would read a
     * 20-byte block as an 18-byte one on four layers and produce a plausible, wrong activation.
     *
     * <p>Worker: one thread per element, local 256.
     */
    // @formatter:on
    public static void dequantizeQ4_1ToFP16(
            KernelContext context, ByteArray w, HalfFloatArray out, int destOffset) {
        int gid = context.globalIdx;
        int blk = gid >>> 5;
        int within = gid & 31;
        int off = blk * 20;
        int half = within >>> 4;
        int byteIndex = within - (half << 4);
        int packed = w.get(off + 4 + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        float d = w.getHalfFloat(off).getFloat32();
        float m = w.getHalfFloat(off + 2).getFloat32();
        out.set(destOffset + gid, new HalfFloat(d * q + m));
    }

    // @formatter:off
    /**
     * {@code gemmMMAQ8} with the depth split across blocks, for the two projections whose output is
     * too narrow to fill the device.
     *
     * <p><b>Why.</b> The tile is 128×128, so a projection into {@code dim} = 1536 has twelve
     * N-blocks, and a chunk of 512 rows has four M-blocks: forty-eight thread blocks on a
     * hundred-and-twenty-eight-SM device, with the rest idle. Measured on this family, doubling the
     * chunk width — which doubles the M-blocks and nothing else — took {@code w2Proj} from 628.5 ms
     * to 338.7 and {@code woProj} from 161.6 to 87.7, while {@code gateUpProj}, whose output is
     * already twelve thousand wide and was never starved, got 6% <i>worse</i>. That is the
     * signature, cleanly separated: these two are starved, not slow. Padding the chunk to a wider
     * one is not the fix — it buys the blocks with wasted rows, and measured worse (pp512 3128 at
     * width 512 against 2489 at width 1024).
     *
     * <p><b>What.</b> Each block covers one slice of K and writes its own partial product, so the
     * block count is multiplied by the number of slices without touching M or N. {@code slices}
     * comes from {@code groupIdz}; the partials land in one buffer of {@code slices*M} rows, which
     * {@link #splitKReduce} then sums into C.
     *
     * <p><b>This reassociates the sum over K</b> — the slices are summed pairwise at the end rather
     * than accumulated in one running total — so it is not bit-identical, and is gated as an
     * arithmetic-order change with the parity bounds unchanged. Everything else is {@code
     * gemmMMAQ8}: tile geometry, the software pipeline, the staging, barriers, MMA order, scale
     * conversion and FP32 accumulation.
     *
     * <p>Requires {@code M % 128 == 0}, {@code N % 128 == 0} and {@code (K / slices) % 32 == 0}.
     * Worker: {@code WorkerGrid3D((M/128)*256, N/128, slices)}, local (256,1,1).
     */
    // @formatter:on
    public static void gemmMMAQ8SplitK(
            KernelContext ctx,
            HalfFloatArray A,
            ByteArray B,
            FloatArray partial,
            int M,
            int N,
            int K,
            int slices) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int slice = ctx.groupIdz;
        int blocksPerRow = K / 32;
        int kSlice = K / slices;
        int kBase = slice * kSlice;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + kBase + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + kBase + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + kBase + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + kBase + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int bCol0 = blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1);
        int bK0 = kBase + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int bCol1 = blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1);
        int bK1 = kBase + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int bCol2 = blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1);
        int bK2 = kBase + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int bCol3 = blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1);
        int bK3 = kBase + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packQ8Halves(B, bCol0, bK0, blocksPerRow);
        int bReg1 = packQ8Halves(B, bCol1, bK1, blocksPerRow);
        int bReg2 = packQ8Halves(B, bCol2, bK2, blocksPerRow);
        int bReg3 = packQ8Halves(B, bCol3, bK3, blocksPerRow);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = kSlice / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packQ8Halves(B, bCol0, kOff + bK0, blocksPerRow);
                bReg1 = packQ8Halves(B, bCol1, kOff + bK1, blocksPerRow);
                bReg2 = packQ8Halves(B, bCol2, kOff + bK2, blocksPerRow);
                bReg3 = packQ8Halves(B, bCol3, kOff + bK3, blocksPerRow);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = slice * M + blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, partial, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, partial, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, partial, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, partial, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, partial, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, partial, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, partial, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, partial, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, partial, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, partial, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, partial, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, partial, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, partial, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, partial, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, partial, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, partial, rBase + 16, cBase + 56, N);
    }

    // @formatter:off
    /**
     * {@link #gemmMMAQ8SplitK} with an FP16 B operand: the depth split for the two narrow
     * projections, over weights decoded once instead of decoded per K-step.
     *
     * <p>The two fixes compose and neither subsumes the other. Splitting the depth answers the grid
     * — a projection into {@code dim} gives forty-eight thread blocks on a
     * hundred-and-twenty-eight-SM device. Decoding the weights first answers the pipeline, since a
     * Q8_0 staging converts float to half in software for want of a register-level conversion. This
     * is both.
     *
     * <p>Requires {@code M % 128 == 0}, {@code N % 128 == 0}, {@code (K / slices) % 16 == 0}.
     * Worker: {@code WorkerGrid3D((M/128)*256, N/128, slices)}, local (256,1,1).
     */
    // @formatter:on
    public static void gemmMMASplitK(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray partial,
            int M,
            int N,
            int K,
            int slices) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int slice = ctx.groupIdz;
        int kSlice = K / slices;
        int kBase = slice * kSlice;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + kBase + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + kBase + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + kBase + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + kBase + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 =
                (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K
                        + kBase
                        + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 =
                (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K
                        + kBase
                        + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 =
                (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K
                        + kBase
                        + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 =
                (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K
                        + kBase
                        + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + K);
        int bReg1 = packHalves(B, gB1, gB1 + K);
        int bReg2 = packHalves(B, gB2, gB2 + K);
        int bReg3 = packHalves(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = kSlice / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = slice * M + blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, partial, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, partial, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, partial, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, partial, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, partial, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, partial, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, partial, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, partial, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, partial, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, partial, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, partial, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, partial, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, partial, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, partial, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, partial, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, partial, rBase + 16, cBase + 56, N);
    }

    // @formatter:off
    /**
     * Sums the depth slices {@link #gemmMMAQ8SplitK} left behind into the projection's output.
     *
     * <p>One thread per output element, slices added in increasing order. The traffic is {@code
     * slices} reads and one write per element — 12.6 MB per call at four slices and a chunk of 512,
     * against the tens of milliseconds the GEMM itself takes, so the pass is not where the time
     * goes.
     *
     * <p>Worker: {@code M*N} threads, local 256.
     */
    // @formatter:on
    public static void splitKReduce(
            KernelContext context, FloatArray partial, FloatArray out, int elements, int slices) {
        int gid = context.globalIdx;
        float sum = 0.0f;
        for (int s = 0; s < slices; s++) {
            sum += partial.get(s * elements + gid);
        }
        out.set(gid, sum);
    }

    // ── Feed-forward ─────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * GeGLU over the packed {@code [gate|up]} rows the fused gate/up GEMM produced, emitted in FP16
     * for the down projection's GEMM: {@code hb[b,i] = gelu(gateUp[b,i]) * gateUp[b,ffnLen+i]}.
     *
     * <p>The GeGLU, not a SwiGLU with the activation swapped: {@link
     * TransformerComputeKernelsLayered#geluActivation} is the same function {@link
     * Gemma4Kernels#fusedGateUpGeGLUQ8} applies on the single-token path, so the two paths differ
     * in where the products come from and not in what is computed from them.
     *
     * <p>Worker: {@code B*ffnLen} threads, local 256.
     */
    // @formatter:on
    public static void batchedGeGLUFP16Packed(
            KernelContext context, HalfFloatArray hbFP16, FloatArray gateUp, int ffnLen) {
        int gid = context.globalIdx;
        int b = gid / ffnLen;
        int i = gid - b * ffnLen;
        int rowBase = b * 2 * ffnLen;
        float gate = gateUp.get(rowBase + i);
        float up = gateUp.get(rowBase + ffnLen + i);
        hbFP16.set(gid, new HalfFloat(TransformerComputeKernelsLayered.geluActivation(gate) * up));
    }

    // ── Per-layer embeddings ─────────────────────────────────────────────────

    // @formatter:off
    /**
     * The per-layer-embedding gate, chunk-wide, emitted in FP16 for the projection that follows:
     * {@code out[b,i] = gelu(gate[b,i]) * perLayerInputs[b, peOffset + i]}.
     *
     * <p>{@code perLayerInputs} is one row of {@code numLayers*segmentSize} per prompt token, so
     * {@code peOffset} selects this layer's segment within a row exactly as it selects it within
     * the single vector on the decode path.
     *
     * <p>Worker: {@code B*segmentSize} threads.
     */
    // @formatter:on
    public static void batchedPleGateGeluMul(
            KernelContext context,
            HalfFloatArray out,
            FloatArray gate,
            FloatArray perLayerInputs,
            int peOffset,
            int segmentSize,
            int perLayerTotal) {
        int gid = context.globalIdx;
        int b = gid / segmentSize;
        int i = gid - b * segmentSize;
        float gated = TransformerComputeKernelsLayered.geluActivation(gate.get(gid));
        out.set(gid, new HalfFloat(gated * perLayerInputs.get(b * perLayerTotal + peOffset + i)));
    }

    // @formatter:off
    /**
     * The per-layer projection's pre-scale and per-segment RMS norm, chunk-wide.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#pleProjScaleAndNormalize}: a row holds {@code
     * numLayers} segments of {@code segmentSize}, one workgroup normalizes one segment of one row,
     * and the single learned weight is reused for every segment as it is on the decode path.
     *
     * <p>Worker: {@code B*numLayers} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedPleProjScaleAndNormalize(
            KernelContext context,
            FloatArray x,
            FloatArray weight,
            int segmentSize,
            int localMemSize,
            float preScale,
            float rmsNormEps) {
        int segIdx = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int base = segIdx * segmentSize;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < segmentSize; i += localSize) {
            float v = x.get(base + i) * preScale;
            x.set(base + i, v);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / segmentSize + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < segmentSize; i += localSize) {
            x.set(base + i, weight.get(i) * (ss * x.get(base + i)));
        }
    }

    // ── Elementwise ──────────────────────────────────────────────────────────

    /**
     * {@code out[i] = (a[i] + b[i]) * scale} — the chunk-wide {@link Gemma4Kernels#addAndScale}.
     */
    public static void batchedAddAndScale(
            KernelContext context, FloatArray out, FloatArray a, FloatArray b, float scale) {
        int gid = context.globalIdx;
        out.set(gid, (a.get(gid) + b.get(gid)) * scale);
    }

    /** {@code x[i] *= scale} — the embedding scale Gemma 4 applies on input. */
    public static void batchedScaleInPlace(KernelContext context, FloatArray x, float scale) {
        int gid = context.globalIdx;
        x.set(gid, x.get(gid) * scale);
    }

    /**
     * {@code x[i] *= scaleTensor[0]} — the chunk-wide {@link Gemma4Kernels#scaleInPlaceFromTensor},
     * for the layers that carry a learned output scale.
     */
    public static void batchedScaleInPlaceFromTensor(
            KernelContext context, FloatArray x, FloatArray scaleTensor) {
        int gid = context.globalIdx;
        x.set(gid, x.get(gid) * scaleTensor.get(0));
    }
}
