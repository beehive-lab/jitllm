package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched-prefill device kernels for gpt-oss. A chunk launches a fixed number of rows; {@code
 * batchInfo[0]} is the chunk's first position and {@code batchInfo[1]} its real rows, so padding
 * rows write nothing.
 *
 * <p>The experts run as grouped int8 tensor-core GEMMs over tiles of up to 64 rows of one expert
 * ({@link Qwen35MoeBatchKernels#groupByExpert} sorts the assignments). MXFP4 needs no rescaling to
 * get there: twice an E2M1 value is an integer in {@code [-12, 12]}, so a block is exactly 32 int8
 * quants times {@code 2^(e - 128)}. The weight staging decodes the nibbles into the int8 tile; the
 * rest is the block-scaled arithmetic of {@link Qwen35MoeBatchKernels#groupedGateUpQ8_0}. Output
 * columns that are not a multiple of the 128-column tile (2880 for gpt-oss) are clamped when staged
 * and masked when written.
 */
public final class GptOssBatchKernels {

    private GptOssBatchKernels() {}

    public static final int TILE_ROWS = 64;
    private static final int SUB_TILES = TILE_ROWS / 16;
    public static final int TILE_COLS = 128;
    public static final int GEMM_THREADS = 256;
    private static final int QK = 32;
    private static final int MX_BYTES = 17;
    private static final int A_WORDS = 512;
    private static final int B_WORDS = 1024;
    static final int HEAD = GptOssKernels.HEAD;
    static final float MASKED = GptOssKernels.MASKED;

    /** RMS-normalizes each active row of {@code x} into {@code out}. One work-group per row. */
    public static void rmsNormRows(
            KernelContext ctx,
            FloatArray out,
            FloatArray x,
            FloatArray weight,
            IntArray batchInfo,
            int dim,
            float eps) {
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int size = ctx.localGroupSizeX;
        float[] red = ctx.allocateFloatLocalArray(256);
        if (row < batchInfo.get(1)) {
            float partial = 0.0f;
            for (int j = lid; j < dim; j += size) {
                float v = x.get(row * dim + j);
                partial += v * v;
            }
            red[lid] = partial;
            ctx.localBarrier();
            for (int stride = size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    red[lid] = red[lid] + red[lid + stride];
                }
                ctx.localBarrier();
            }
            float scale = 1.0f / TornadoMath.sqrt(red[0] / dim + eps);
            for (int j = lid; j < dim; j += size) {
                out.set(row * dim + j, weight.get(j) * scale * x.get(row * dim + j));
            }
        }
    }

    /**
     * Adds the Q, K and V biases, rotates Q and K from the YaRN tables at each row's position, and
     * writes K and V to the paged cache. One thread per (row, head, pair).
     */
    public static void biasRopeCache(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray bq,
            FloatArray bk,
            FloatArray bv,
            FloatArray cosTable,
            FloatArray sinTable,
            FloatArray keyCache,
            FloatArray valueCache,
            int heads,
            int kvHeads,
            int kvDim,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int half = HEAD / 2;
        int perRow = heads * half;
        int id = ctx.globalIdx;
        int row = id / perRow;
        if (row < batchInfo.get(1)) {
            int rem = id - row * perRow;
            int h = rem / half;
            int i = rem - h * half;
            int pos = batchInfo.get(0) + row;
            float c = cosTable.get(pos * half + i);
            float s = sinTable.get(pos * half + i);
            int o = h * HEAD + i;
            int qo = row * heads * HEAD + o;
            float q0 = q.get(qo) + bq.get(o);
            float q1 = q.get(qo + half) + bq.get(o + half);
            q.set(qo, q0 * c - q1 * s);
            q.set(qo + half, q0 * s + q1 * c);
            if (h < kvHeads) {
                int ko = row * kvDim + o;
                float k0 = k.get(ko) + bk.get(o);
                float k1 = k.get(ko + half) + bk.get(o + half);
                int slot = batchInfo.get(2);
                int base =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(base + o, k0 * c - k1 * s);
                keyCache.set(base + o + half, k0 * s + k1 * c);
                valueCache.set(base + o, v.get(ko) + bv.get(o));
                valueCache.set(base + o + half, v.get(ko + half) + bv.get(o + half));
            }
        }
    }

    /**
     * Causal attention with sinks for every active row of the chunk, over the cache the chunk has
     * just written. One work-group of {@link GptOssKernels#HEAD} threads per (row, head), the same
     * online softmax as {@link GptOssKernels#attentionSplit}, with the sink folded in at the end.
     * The fallback where the cache is FP32 or the device has no tensor cores.
     */
    public static void attention(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray sinks,
            FloatArray out,
            int heads,
            int kvDim,
            int kvMul,
            int window,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int g = ctx.groupIdx;
        int row = g / heads;
        int h = g - row * heads;
        int t = ctx.localIdx;
        float[] qs = ctx.allocateFloatLocalArray(HEAD);
        float[] p = ctx.allocateFloatLocalArray(HEAD);
        float[] red = ctx.allocateFloatLocalArray(HEAD);
        if (row < batchInfo.get(1)) {
            int pos = batchInfo.get(0) + row;
            int slot = batchInfo.get(2);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
            int kvHead = (h / kvMul) * HEAD;
            float scale = 1.0f / TornadoMath.sqrt((float) HEAD);
            int qBase = (row * heads + h) * HEAD;
            qs[t] = q.get(qBase + t);
            ctx.localBarrier();
            float runMax = MASKED;
            float runSum = 0.0f;
            float acc = 0.0f;
            for (int c = start; c <= pos; c += HEAD) {
                int pt = c + t;
                int valid = 1 - ((pos - pt) >>> 31);
                int pc = Math.min(pt, pos);
                int kb =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        pc,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHead;
                float dot = 0.0f;
                for (int d = 0; d < HEAD; d++) {
                    dot += qs[d] * keyCache.get(kb + d);
                }
                float score = dot * scale * valid + MASKED * (1 - valid);
                red[t] = score;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = Math.max(red[t], red[t + stride]);
                    }
                    ctx.localBarrier();
                }
                float newMax = Math.max(runMax, red[0]);
                ctx.localBarrier();
                float e = TornadoMath.exp(score - newMax);
                p[t] = e;
                red[t] = e;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = red[t] + red[t + stride];
                    }
                    ctx.localBarrier();
                }
                float correction = TornadoMath.exp(runMax - newMax);
                runSum = runSum * correction + red[0];
                acc = acc * correction;
                runMax = newMax;
                int n = Math.min(HEAD, pos - c + 1);
                for (int i = 0; i < n; i++) {
                    int vb =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            c + i,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead;
                    acc += p[i] * valueCache.get(vb + t);
                }
                ctx.localBarrier();
            }
            float sink = sinks.get(h);
            float finalMax = Math.max(runMax, sink);
            float correction = TornadoMath.exp(runMax - finalMax);
            float denominator = runSum * correction + TornadoMath.exp(sink - finalMax);
            out.set(qBase + t, acc * correction / denominator);
        }
    }

    /**
     * The rows {@code [firstRow, rows)} of a Q8_0 projection the 128-column GEMM does not cover,
     * added into {@code x} for every active row. One warp per (row, tail output row).
     */
    public static void tailResidualQ8_0(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray in,
            ByteArray w,
            FloatArray x,
            int rows,
            int cols,
            int firstRow) {
        int lane = ctx.localIdx & 31;
        int warp = ctx.groupIdx * (ctx.localGroupSizeX / 32) + ctx.localIdx / 32;
        int tail = rows - firstRow;
        int token = warp / tail;
        int r = firstRow + warp - token * tail;
        if (token < batchInfo.get(1)) {
            int blocks = cols / 32;
            int base = r * blocks * 34;
            float sum = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int off = base + b * 34;
                float s = w.getHalfFloat(off).getFloat32();
                sum += s * w.get(off + 2 + lane) * in.get(token * cols + b * 32 + lane);
            }
            sum = GptOssKernels.warpSum(ctx, sum);
            if (lane == 0) {
                int o = token * rows + r;
                x.set(o, x.get(o) + sum);
            }
        }
    }

    /** {@code x[t][c] += bias[c]} for every active row. */
    public static void biasRows(
            KernelContext ctx, IntArray batchInfo, FloatArray bias, FloatArray x, int dim) {
        int i = ctx.globalIdx;
        if (i / dim < batchInfo.get(1)) {
            x.set(i, x.get(i) + bias.get(i - (i / dim) * dim));
        }
    }

    /** Router logits for every active row, plus bias. One warp per (row, expert). */
    public static void router(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray in,
            FloatArray w,
            FloatArray bias,
            FloatArray out,
            int experts,
            int dim) {
        int lane = ctx.localIdx & 31;
        int warp = ctx.groupIdx * (ctx.localGroupSizeX / 32) + ctx.localIdx / 32;
        int token = warp / experts;
        int e = warp - token * experts;
        if (token < batchInfo.get(1)) {
            float partial = 0.0f;
            for (int j = lane; j < dim; j += 32) {
                partial += w.get(e * dim + j) * in.get(token * dim + j);
            }
            float sum = GptOssKernels.warpSum(ctx, partial);
            if (lane == 0) {
                out.set(warp, sum + bias.get(e));
            }
        }
    }

    /**
     * Each active row's top {@code used} experts and a softmax over their logits. A thread a row.
     */
    public static void topKSoftmax(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray logits,
            IntArray ids,
            FloatArray weights,
            int experts,
            int used) {
        int token = ctx.globalIdx;
        if (token < batchInfo.get(1)) {
            int base = token * experts;
            for (int slot = 0; slot < used; slot++) {
                float best = Float.NEGATIVE_INFINITY;
                int index = 0;
                for (int e = 0; e < experts; e++) {
                    if (logits.get(base + e) > best) {
                        best = logits.get(base + e);
                        index = e;
                    }
                }
                ids.set(token * used + slot, index);
                weights.set(token * used + slot, best);
                logits.set(base + index, Float.NEGATIVE_INFINITY);
            }
            float max = weights.get(token * used);
            float sum = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                float e = TornadoMath.exp(weights.get(token * used + slot) - max);
                weights.set(token * used + slot, e);
                sum += e;
            }
            for (int slot = 0; slot < used; slot++) {
                weights.set(token * used + slot, weights.get(token * used + slot) / sum);
            }
        }
    }

    /**
     * {@code residual[t][c] += sum over slots of weight[t][slot] * routed[position(t, slot)][c]},
     * the slots in order. Worker: {@code batch * dim} lanes.
     */
    public static void combine(
            KernelContext ctx,
            FloatArray routed,
            IntArray position,
            FloatArray weights,
            FloatArray residual,
            IntArray batchInfo,
            int dim,
            int used) {
        int i = ctx.globalIdx;
        int token = i / dim;
        if (token < batchInfo.get(1)) {
            int c = i - token * dim;
            float sum = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                int a = token * used + slot;
                sum += weights.get(a) * routed.get(position.get(a) * dim + c);
            }
            residual.set(i, residual.get(i) + sum);
        }
    }

    /** Stages one round's activation tile and scales: 64 rows, two blocks, four words a lane. */
    private static void stageA(
            KernelContext ctx,
            int[] aTile,
            float[] sA,
            ByteArray a8,
            FloatArray dA,
            IntArray sortedToken,
            int gather,
            int rowStart,
            int rows,
            int round,
            int buf,
            int k,
            int tid) {
        int kBlocks = k / QK;
        int kb0 = round * 2;
        for (int s = 0; s < 4; s++) {
            int linear = tid + (s << 8);
            int b = linear >> 9;
            int row = (linear >> 3) & 63;
            int quad = linear & 7;
            int sorted = rowStart + Math.min(row, rows - 1);
            int source = gather * sortedToken.get(sorted) + (1 - gather) * sorted;
            ctx.asyncCopyToLocal(
                    aTile,
                    buf * 2 * A_WORDS + linear,
                    a8,
                    source * k + (kb0 + b) * QK + (quad << 2));
        }
        if (tid < 2 * TILE_ROWS) {
            int sb = tid >> 6;
            int sr = tid & 63;
            int sSorted = rowStart + Math.min(sr, rows - 1);
            int sSource = gather * sortedToken.get(sSorted) + (1 - gather) * sSorted;
            sA[buf * 2 * TILE_ROWS + sb * TILE_ROWS + sr] = dA.get(sSource * kBlocks + kb0 + sb);
        }
    }

    /**
     * Stages one round's MXFP4 weight tile, 128 columns from weight row {@code firstRow} and two
     * blocks, with the blocks' scales, in the word layout of {@link
     * Qwen35MoeBatchKernels#groupedGateUpQ8_0}'s tile (a word: two k of two adjacent columns).
     * Thread {@code t} decodes bytes {@code 8h .. 8h + 7} ({@code h = t >> 7}) of block {@code (t
     * >> 6) & 1} of column pair {@code t & 63}: the low nibbles are k {@code 8h .. 8h + 7}, the
     * high ones {@code 16 + 8h ..}, through the shared int8 table {@code lut}. Columns at or past
     * {@code colLimit} read column {@code colLimit - 1} and are never written.
     */
    private static void stageBMXFP4(
            int[] bTile,
            float[] sW,
            int[] lut,
            ByteArray w,
            int firstRow,
            int colLimit,
            int rowBytes,
            int round,
            int buf,
            int tid) {
        int cp = tid & 63;
        int bl = (tid >> 6) & 1;
        int h = tid >> 7;
        int blk = round * 2 + bl;
        int c0 = Math.min(2 * cp, colLimit - 1);
        int c1 = Math.min(2 * cp + 1, colLimit - 1);
        int off0 = (firstRow + c0) * rowBytes + blk * MX_BYTES;
        int off1 = (firstRow + c1) * rowBytes + blk * MX_BYTES;
        int colPair = cp & 15;
        int group = cp >> 4;
        int base =
                buf * 2 * B_WORDS
                        + bl * B_WORDS
                        + group * 256
                        + ((colPair >> 2) << 6)
                        + (colPair & 3);
        for (int p = 0; p < 4; p++) {
            int j = 8 * h + 2 * p;
            int a0 = w.get(off0 + 1 + j) & 0xFF;
            int a1 = w.get(off0 + 2 + j) & 0xFF;
            int b0 = w.get(off1 + 1 + j) & 0xFF;
            int b1 = w.get(off1 + 2 + j) & 0xFF;
            // k = j, j + 1 from the low nibbles: kRow = j / 2.
            int lowWord =
                    (lut[a0 & 15] & 0xFF)
                            | ((lut[a1 & 15] & 0xFF) << 8)
                            | ((lut[b0 & 15] & 0xFF) << 16)
                            | ((lut[b1 & 15] & 0xFF) << 24);
            // k = 16 + j, 17 + j from the high nibbles: kRow = 8 + j / 2.
            int highWord =
                    (lut[a0 >> 4] & 0xFF)
                            | ((lut[a1 >> 4] & 0xFF) << 8)
                            | ((lut[b0 >> 4] & 0xFF) << 16)
                            | ((lut[b1 >> 4] & 0xFF) << 24);
            bTile[base + ((j >> 1) << 2)] = lowWord;
            bTile[base + ((8 + (j >> 1)) << 2)] = highWord;
        }
        if (h == 0) {
            sW[buf * 256 + bl * 128 + 2 * cp] = GptOssKernels.e8m0Half(w.get(off0) & 0xFF);
            sW[buf * 256 + bl * 128 + 2 * cp + 1] = GptOssKernels.e8m0Half(w.get(off1) & 0xFF);
        }
    }

    /** Fills the shared int8 nibble table: twice the E2M1 values. */
    private static void stageLut(KernelContext ctx, int[] lut) {
        if (ctx.localIdx < 16) {
            lut[ctx.localIdx] = (int) GptOssKernels.fp4(ctx.localIdx);
        }
        ctx.localBarrier();
    }

    /**
     * {@code hidden[row][c] = swiglu(gate_e[c]·x + gb_e[c], up_e[c]·x + ub_e[c])} for every sorted
     * row of every tile, {@code e} the tile's expert, over the int8-quantized normed chunk. Worker:
     * {@code WorkerGrid2D(ceil(hiddenDim / 128) * 256, maxTiles)}, local 256.
     */
    public static void groupedGateUpMXFP4(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            IntArray sortedToken,
            IntArray tiles,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray gateBias,
            FloatArray upBias,
            FloatArray hidden,
            int dim,
            int hiddenDim) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        if (tile < tiles.get(0) && blockCol < hiddenDim) {
            int expert = tiles.get(1 + 3 * tile);
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = dim / (2 * QK);
            int rowBytes = (dim / QK) * MX_BYTES;
            int firstRow = expert * hiddenDim + blockCol;
            int colLimit = Math.min(TILE_COLS, hiddenDim - blockCol);

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] gTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            int[] uTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sG = ctx.allocateFloatLocalArray(2 * 256);
            float[] sU = ctx.allocateFloatLocalArray(2 * 256);
            int[] lut = ctx.allocateIntLocalArray(16);
            stageLut(ctx, lut);

            float[] gate = new float[SUB_TILES * 8];
            float[] up = new float[SUB_TILES * 8];
            for (int i = 0; i < SUB_TILES * 8; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int activeSub = (rows + 15) >> 4;

            stageA(ctx, aTile, sA, a8, dA, sortedToken, 1, rowStart, rows, 0, 0, dim, tid);
            stageBMXFP4(gTile, sG, lut, gateExperts, firstRow, colLimit, rowBytes, 0, 0, tid);
            stageBMXFP4(uTile, sU, lut, upExperts, firstRow, colLimit, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    stageA(
                            ctx,
                            aTile,
                            sA,
                            a8,
                            dA,
                            sortedToken,
                            1,
                            rowStart,
                            rows,
                            round + 1,
                            next,
                            dim,
                            tid);
                    ctx.asyncCopyCommit();
                    stageBMXFP4(
                            gTile,
                            sG,
                            lut,
                            gateExperts,
                            firstRow,
                            colLimit,
                            rowBytes,
                            round + 1,
                            next,
                            tid);
                    stageBMXFP4(
                            uTile, sU, lut, upExperts, firstRow, colLimit, rowBytes, round + 1,
                            next, tid);
                }
                for (int b = 0; b < 2; b++) {
                    for (int j = 0; j < 2; j++) {
                        int bOff = (buf * 2 + b) * B_WORDS * 4 + (warp * 2 + j) * 256;
                        int sCol = buf * 256 + b * 128 + warp * 16 + j * 8 + colInWarp;
                        byte[] fg = ctx.mmaLoadBInt8(gTile, 32, bOff);
                        byte[] fu = ctx.mmaLoadBInt8(uTile, 32, bOff);
                        float g0 = sG[sCol];
                        float g1 = sG[sCol + 1];
                        float u0 = sU[sCol];
                        float u1 = sU[sCol + 1];
                        for (int t = 0; t < SUB_TILES; t++) {
                            if (t < activeSub) {
                                byte[] fa =
                                        ctx.mmaLoadAInt8(
                                                aTile, 32, (buf * 2 + b) * A_WORDS * 4 + t * 512);
                                int sRow = buf * 2 * TILE_ROWS + b * TILE_ROWS + t * 16 + rowInWarp;
                                float dA0 = sA[sRow];
                                float dA1 = sA[sRow + 8];
                                int[] dg =
                                        ctx.mmaInt8(
                                                fa, fg, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                                int[] du =
                                        ctx.mmaInt8(
                                                fa, fu, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                                int o = t * 8 + j * 4;
                                gate[o] += (float) dg[0] * dA0 * g0;
                                gate[o + 1] += (float) dg[1] * dA0 * g1;
                                gate[o + 2] += (float) dg[2] * dA1 * g0;
                                gate[o + 3] += (float) dg[3] * dA1 * g1;
                                up[o] += (float) du[0] * dA0 * u0;
                                up[o + 1] += (float) du[1] * dA0 * u1;
                                up[o + 2] += (float) du[2] * dA1 * u0;
                                up[o + 3] += (float) du[3] * dA1 * u1;
                            }
                        }
                    }
                }
                if (round + 1 < rounds) {
                    ctx.asyncCopyWaitGroup(0);
                }
                ctx.localBarrier();
            }

            for (int t = 0; t < SUB_TILES; t++) {
                for (int j = 0; j < 2; j++) {
                    int local = warp * 16 + j * 8 + colInWarp;
                    for (int e = 0; e < 4; e++) {
                        int row = t * 16 + rowInWarp + 8 * (e >> 1);
                        int c = local + (e & 1);
                        if (row < rows && c < colLimit) {
                            int wrow = firstRow + c;
                            float g = gate[t * 8 + j * 4 + e] + gateBias.get(wrow);
                            float u = up[t * 8 + j * 4 + e] + upBias.get(wrow);
                            hidden.set(
                                    (rowStart + row) * hiddenDim + blockCol + c,
                                    GptOssKernels.swiglu(g, u));
                        }
                    }
                }
            }
        }
    }

    /**
     * {@code out[row][c] = down_e[c]·hidden[row] + db_e[c]} for every sorted row of every tile,
     * over the int8-quantized hidden rows (sorted order). Worker: {@code WorkerGrid2D(ceil(dim /
     * 128) * 256, maxTiles)}, local 256.
     */
    public static void groupedDownMXFP4(
            KernelContext ctx,
            ByteArray h8,
            FloatArray dH,
            IntArray tiles,
            ByteArray downExperts,
            FloatArray downBias,
            FloatArray out,
            int dim,
            int hiddenDim) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        if (tile < tiles.get(0) && blockCol < dim) {
            int expert = tiles.get(1 + 3 * tile);
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = hiddenDim / (2 * QK);
            int rowBytes = (hiddenDim / QK) * MX_BYTES;
            int firstRow = expert * dim + blockCol;
            int colLimit = Math.min(TILE_COLS, dim - blockCol);

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sW = ctx.allocateFloatLocalArray(2 * 256);
            int[] lut = ctx.allocateIntLocalArray(16);
            stageLut(ctx, lut);

            float[] acc = new float[SUB_TILES * 8];
            for (int i = 0; i < SUB_TILES * 8; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int activeSub = (rows + 15) >> 4;

            stageA(ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, 0, 0, hiddenDim, tid);
            stageBMXFP4(bTile, sW, lut, downExperts, firstRow, colLimit, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    stageA(
                            ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, round + 1, next,
                            hiddenDim, tid);
                    ctx.asyncCopyCommit();
                    stageBMXFP4(
                            bTile,
                            sW,
                            lut,
                            downExperts,
                            firstRow,
                            colLimit,
                            rowBytes,
                            round + 1,
                            next,
                            tid);
                }
                for (int b = 0; b < 2; b++) {
                    for (int j = 0; j < 2; j++) {
                        int bOff = (buf * 2 + b) * B_WORDS * 4 + (warp * 2 + j) * 256;
                        int sCol = buf * 256 + b * 128 + warp * 16 + j * 8 + colInWarp;
                        byte[] fb = ctx.mmaLoadBInt8(bTile, 32, bOff);
                        float w0 = sW[sCol];
                        float w1 = sW[sCol + 1];
                        for (int t = 0; t < SUB_TILES; t++) {
                            if (t < activeSub) {
                                byte[] fa =
                                        ctx.mmaLoadAInt8(
                                                aTile, 32, (buf * 2 + b) * A_WORDS * 4 + t * 512);
                                int sRow = buf * 2 * TILE_ROWS + b * TILE_ROWS + t * 16 + rowInWarp;
                                float dA0 = sA[sRow];
                                float dA1 = sA[sRow + 8];
                                int[] d =
                                        ctx.mmaInt8(
                                                fa, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                                int o = t * 8 + j * 4;
                                acc[o] += (float) d[0] * dA0 * w0;
                                acc[o + 1] += (float) d[1] * dA0 * w1;
                                acc[o + 2] += (float) d[2] * dA1 * w0;
                                acc[o + 3] += (float) d[3] * dA1 * w1;
                            }
                        }
                    }
                }
                if (round + 1 < rounds) {
                    ctx.asyncCopyWaitGroup(0);
                }
                ctx.localBarrier();
            }

            for (int t = 0; t < SUB_TILES; t++) {
                for (int j = 0; j < 2; j++) {
                    int local = warp * 16 + j * 8 + colInWarp;
                    for (int e = 0; e < 4; e++) {
                        int row = t * 16 + rowInWarp + 8 * (e >> 1);
                        int c = local + (e & 1);
                        if (row < rows && c < colLimit) {
                            out.set(
                                    (rowStart + row) * dim + blockCol + c,
                                    acc[t * 8 + j * 4 + e] + downBias.get(firstRow + c));
                        }
                    }
                }
            }
        }
    }

    /** The FP16-cache form of {@link #biasRopeCache}. */
    /**
     * Adds the Q, K and V biases, rotates Q and K from the YaRN tables at each row's position, and
     * writes K and V to the paged cache. One thread per (row, head, pair).
     */
    /** {@link #biasRopeCache} over an FP16 key/value cache. */
    public static void biasRopeCacheFP16(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray bq,
            FloatArray bk,
            FloatArray bv,
            FloatArray cosTable,
            FloatArray sinTable,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            int heads,
            int kvHeads,
            int kvDim,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int half = HEAD / 2;
        int perRow = heads * half;
        int id = ctx.globalIdx;
        int row = id / perRow;
        if (row < batchInfo.get(1)) {
            int rem = id - row * perRow;
            int h = rem / half;
            int i = rem - h * half;
            int pos = batchInfo.get(0) + row;
            float c = cosTable.get(pos * half + i);
            float s = sinTable.get(pos * half + i);
            int o = h * HEAD + i;
            int qo = row * heads * HEAD + o;
            float q0 = q.get(qo) + bq.get(o);
            float q1 = q.get(qo + half) + bq.get(o + half);
            q.set(qo, q0 * c - q1 * s);
            q.set(qo + half, q0 * s + q1 * c);
            if (h < kvHeads) {
                int ko = row * kvDim + o;
                float k0 = k.get(ko) + bk.get(o);
                float k1 = k.get(ko + half) + bk.get(o + half);
                int slot = batchInfo.get(2);
                int base =
                        KvBlockAddress.offset(
                                blockTable,
                                slot,
                                pos,
                                KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                                kvDim,
                                blockCfg,
                                blockStride);
                keyCache.set(base + o, new HalfFloat(k0 * c - k1 * s));
                keyCache.set(base + o + half, new HalfFloat(k0 * s + k1 * c));
                valueCache.set(base + o, new HalfFloat(v.get(ko) + bv.get(o)));
                valueCache.set(base + o + half, new HalfFloat(v.get(ko + half) + bv.get(o + half)));
            }
        }
    }

    /** {@link #attention} over an FP16 key/value cache. */
    public static void attentionFP16(
            KernelContext ctx,
            IntArray batchInfo,
            FloatArray q,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray sinks,
            FloatArray out,
            int heads,
            int kvDim,
            int kvMul,
            int window,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {
        int g = ctx.groupIdx;
        int row = g / heads;
        int h = g - row * heads;
        int t = ctx.localIdx;
        float[] qs = ctx.allocateFloatLocalArray(HEAD);
        float[] p = ctx.allocateFloatLocalArray(HEAD);
        float[] red = ctx.allocateFloatLocalArray(HEAD);
        if (row < batchInfo.get(1)) {
            int pos = batchInfo.get(0) + row;
            int slot = batchInfo.get(2);
            int start = Math.max(0, pos - window + 1) * Math.min(window, 1);
            int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
            int kvHead = (h / kvMul) * HEAD;
            float scale = 1.0f / TornadoMath.sqrt((float) HEAD);
            int qBase = (row * heads + h) * HEAD;
            qs[t] = q.get(qBase + t);
            ctx.localBarrier();
            float runMax = MASKED;
            float runSum = 0.0f;
            float acc = 0.0f;
            for (int c = start; c <= pos; c += HEAD) {
                int pt = c + t;
                int valid = 1 - ((pos - pt) >>> 31);
                int pc = Math.min(pt, pos);
                int kb =
                        KvBlockAddress.offset(
                                        blockTable,
                                        slot,
                                        pc,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + kvHead;
                float dot = 0.0f;
                for (int d = 0; d < HEAD; d++) {
                    dot += qs[d] * keyCache.get(kb + d).getFloat32();
                }
                float score = dot * scale * valid + MASKED * (1 - valid);
                red[t] = score;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = Math.max(red[t], red[t + stride]);
                    }
                    ctx.localBarrier();
                }
                float newMax = Math.max(runMax, red[0]);
                ctx.localBarrier();
                float e = TornadoMath.exp(score - newMax);
                p[t] = e;
                red[t] = e;
                ctx.localBarrier();
                for (int stride = HEAD / 2; stride > 0; stride >>= 1) {
                    if (t < stride) {
                        red[t] = red[t] + red[t + stride];
                    }
                    ctx.localBarrier();
                }
                float correction = TornadoMath.exp(runMax - newMax);
                runSum = runSum * correction + red[0];
                acc = acc * correction;
                runMax = newMax;
                int n = Math.min(HEAD, pos - c + 1);
                for (int i = 0; i < n; i++) {
                    int vb =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            c + i,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead;
                    acc += p[i] * valueCache.get(vb + t).getFloat32();
                }
                ctx.localBarrier();
            }
            float sink = sinks.get(h);
            float finalMax = Math.max(runMax, sink);
            float correction = TornadoMath.exp(runMax - finalMax);
            float denominator = runSum * correction + TornadoMath.exp(sink - finalMax);
            out.set(qBase + t, acc * correction / denominator);
        }
    }

    /**
     * gpt-oss's batched-prefill attention on the FP16 tensor cores: {@link
     * TensorCoreAttentionKernels#flashAttentionHead64} with a sliding window ({@code window} keys
     * up to and including each row's own position; 0 = causal over everything) and a learned sink
     * per head that joins each row's softmax denominator. Writes FP32 rows for the output
     * projection. Grid and inputs as {@code flashAttentionHead64}.
     */
    public static void flashAttentionSinks64(
            KernelContext ctx,
            IntArray batchInfo,
            HalfFloatArray queries,
            HalfFloatArray keysT,
            HalfFloatArray values,
            FloatArray out,
            int heads,
            int kvMul,
            int qDim,
            int rowBlocks,
            int paddedPositions,
            float scale,
            FloatArray sinks,
            int window) {
        int[] kTile = ctx.allocateIntLocalArray(2048);
        int[] vTile = ctx.allocateIntLocalArray(2048);
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
        // The sliding window: keys before lo are masked; the blocks wholly before the tile's first
        // row's window are skipped.
        int lo0 = window > 0 ? p0 - window + 1 : -1000000000;
        int lo1 = window > 0 ? p1 - window + 1 : -1000000000;
        int kbFirst = window > 0 ? Math.max(0, start + wgRow - window + 1) / 64 : 0;
        // A 64-key block of one key/value head: 4096 halves of keys, then 4096 of values.
        int gBase = g * (paddedPositions / 64) * 4096;
        int qCol = h * 64;
        HalfFloat[] q0 = ctx.mmaLoadA(queries, rowBase, qCol + 0, qDim);
        HalfFloat[] q1 = ctx.mmaLoadA(queries, rowBase, qCol + 16, qDim);
        HalfFloat[] q2 = ctx.mmaLoadA(queries, rowBase, qCol + 32, qDim);
        HalfFloat[] q3 = ctx.mmaLoadA(queries, rowBase, qCol + 48, qDim);
        float m0 = -1.0e30f;
        float l0 = 0.0f;
        float m1 = -1.0e30f;
        float l1 = 0.0f;
        for (int kb = kbFirst; kb < blocks; kb++) {
            int key0 = kb * 64;
            int kSrc = gBase + kb * 4096;
            for (int i = tid; i < 512; i += 128) {
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
            s1 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 1024), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 2048), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 3072), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 4096), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 5120), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 6144), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 7168), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 256), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 1280), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 2304), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 3328), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 4352), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 5376), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 6400), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 7424), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 512), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 1536), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 2560), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 3584), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 4608), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 5632), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 6656), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 7680), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 768), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 1792), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 2816), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 3840), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 4864), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 5888), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 6912), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 7936), s7, MMAShape.M16N8K16);
            ctx.localBarrier();
            int c = key0 + 2 * quad;
            float a00 = c <= p0 && c >= lo0 ? s0[0] * scale : -1.0e30f;
            float a01 = c + 1 <= p0 && c + 1 >= lo0 ? s0[1] * scale : -1.0e30f;
            float a02 = c <= p1 && c >= lo1 ? s0[2] * scale : -1.0e30f;
            float a03 = c + 1 <= p1 && c + 1 >= lo1 ? s0[3] * scale : -1.0e30f;
            float a10 = c + 8 <= p0 && c + 8 >= lo0 ? s1[0] * scale : -1.0e30f;
            float a11 = c + 9 <= p0 && c + 9 >= lo0 ? s1[1] * scale : -1.0e30f;
            float a12 = c + 8 <= p1 && c + 8 >= lo1 ? s1[2] * scale : -1.0e30f;
            float a13 = c + 9 <= p1 && c + 9 >= lo1 ? s1[3] * scale : -1.0e30f;
            float a20 = c + 16 <= p0 && c + 16 >= lo0 ? s2[0] * scale : -1.0e30f;
            float a21 = c + 17 <= p0 && c + 17 >= lo0 ? s2[1] * scale : -1.0e30f;
            float a22 = c + 16 <= p1 && c + 16 >= lo1 ? s2[2] * scale : -1.0e30f;
            float a23 = c + 17 <= p1 && c + 17 >= lo1 ? s2[3] * scale : -1.0e30f;
            float a30 = c + 24 <= p0 && c + 24 >= lo0 ? s3[0] * scale : -1.0e30f;
            float a31 = c + 25 <= p0 && c + 25 >= lo0 ? s3[1] * scale : -1.0e30f;
            float a32 = c + 24 <= p1 && c + 24 >= lo1 ? s3[2] * scale : -1.0e30f;
            float a33 = c + 25 <= p1 && c + 25 >= lo1 ? s3[3] * scale : -1.0e30f;
            float a40 = c + 32 <= p0 && c + 32 >= lo0 ? s4[0] * scale : -1.0e30f;
            float a41 = c + 33 <= p0 && c + 33 >= lo0 ? s4[1] * scale : -1.0e30f;
            float a42 = c + 32 <= p1 && c + 32 >= lo1 ? s4[2] * scale : -1.0e30f;
            float a43 = c + 33 <= p1 && c + 33 >= lo1 ? s4[3] * scale : -1.0e30f;
            float a50 = c + 40 <= p0 && c + 40 >= lo0 ? s5[0] * scale : -1.0e30f;
            float a51 = c + 41 <= p0 && c + 41 >= lo0 ? s5[1] * scale : -1.0e30f;
            float a52 = c + 40 <= p1 && c + 40 >= lo1 ? s5[2] * scale : -1.0e30f;
            float a53 = c + 41 <= p1 && c + 41 >= lo1 ? s5[3] * scale : -1.0e30f;
            float a60 = c + 48 <= p0 && c + 48 >= lo0 ? s6[0] * scale : -1.0e30f;
            float a61 = c + 49 <= p0 && c + 49 >= lo0 ? s6[1] * scale : -1.0e30f;
            float a62 = c + 48 <= p1 && c + 48 >= lo1 ? s6[2] * scale : -1.0e30f;
            float a63 = c + 49 <= p1 && c + 49 >= lo1 ? s6[3] * scale : -1.0e30f;
            float a70 = c + 56 <= p0 && c + 56 >= lo0 ? s7[0] * scale : -1.0e30f;
            float a71 = c + 57 <= p0 && c + 57 >= lo0 ? s7[1] * scale : -1.0e30f;
            float a72 = c + 56 <= p1 && c + 56 >= lo1 ? s7[2] * scale : -1.0e30f;
            float a73 = c + 57 <= p1 && c + 57 >= lo1 ? s7[3] * scale : -1.0e30f;
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
        // The head's sink joins each row's maximum and denominator, with no value.
        float sink = sinks.get(h);
        float sm0 = TornadoMath.max(m0, sink);
        l0 = l0 * TornadoMath.exp(m0 - sm0) + TornadoMath.exp(sink - sm0);
        m0 = sm0;
        float sm1 = TornadoMath.max(m1, sink);
        l1 = l1 * TornadoMath.exp(m1 - sm1) + TornadoMath.exp(sink - sm1);
        m1 = sm1;
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
        // The warp's probabilities in FP16, 16 rows of 64 keys, rows 72 halves apart.
        int pRow0 = warp * 576 + (lane >> 2) * 36 + quad;
        int pRow1 = pRow0 + 288;
        int pBytes = warp * 2304;
        for (int kb = kbFirst; kb < blocks; kb++) {
            int key0 = kb * 64;
            int kSrc = gBase + kb * 4096;
            for (int i = tid; i < 512; i += 128) {
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
            s1 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 1024), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 2048), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 3072), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 4096), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 5120), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 6144), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q0, ctx.mmaLoadB(kTile, 16, 7168), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 256), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 1280), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 2304), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 3328), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 4352), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 5376), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 6400), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q1, ctx.mmaLoadB(kTile, 16, 7424), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 512), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 1536), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 2560), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 3584), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 4608), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 5632), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 6656), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q2, ctx.mmaLoadB(kTile, 16, 7680), s7, MMAShape.M16N8K16);
            s0 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 768), s0, MMAShape.M16N8K16);
            s1 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 1792), s1, MMAShape.M16N8K16);
            s2 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 2816), s2, MMAShape.M16N8K16);
            s3 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 3840), s3, MMAShape.M16N8K16);
            s4 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 4864), s4, MMAShape.M16N8K16);
            s5 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 5888), s5, MMAShape.M16N8K16);
            s6 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 6912), s6, MMAShape.M16N8K16);
            s7 = ctx.mma(q3, ctx.mmaLoadB(kTile, 16, 7936), s7, MMAShape.M16N8K16);
            int c = key0 + 2 * quad;
            float w00 = c <= p0 && c >= lo0 ? TornadoMath.exp(s0[0] * scale - x0) * i0 : 0.0f;
            float w01 =
                    c + 1 <= p0 && c + 1 >= lo0 ? TornadoMath.exp(s0[1] * scale - x0) * i0 : 0.0f;
            float w02 = c <= p1 && c >= lo1 ? TornadoMath.exp(s0[2] * scale - x1) * i1 : 0.0f;
            float w03 =
                    c + 1 <= p1 && c + 1 >= lo1 ? TornadoMath.exp(s0[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 0] =
                    (new HalfFloat(w00).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w01).getHalfFloatValue() << 16);
            pTile[pRow1 + 0] =
                    (new HalfFloat(w02).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w03).getHalfFloatValue() << 16);
            float w10 =
                    c + 8 <= p0 && c + 8 >= lo0 ? TornadoMath.exp(s1[0] * scale - x0) * i0 : 0.0f;
            float w11 =
                    c + 9 <= p0 && c + 9 >= lo0 ? TornadoMath.exp(s1[1] * scale - x0) * i0 : 0.0f;
            float w12 =
                    c + 8 <= p1 && c + 8 >= lo1 ? TornadoMath.exp(s1[2] * scale - x1) * i1 : 0.0f;
            float w13 =
                    c + 9 <= p1 && c + 9 >= lo1 ? TornadoMath.exp(s1[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 4] =
                    (new HalfFloat(w10).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w11).getHalfFloatValue() << 16);
            pTile[pRow1 + 4] =
                    (new HalfFloat(w12).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w13).getHalfFloatValue() << 16);
            float w20 =
                    c + 16 <= p0 && c + 16 >= lo0 ? TornadoMath.exp(s2[0] * scale - x0) * i0 : 0.0f;
            float w21 =
                    c + 17 <= p0 && c + 17 >= lo0 ? TornadoMath.exp(s2[1] * scale - x0) * i0 : 0.0f;
            float w22 =
                    c + 16 <= p1 && c + 16 >= lo1 ? TornadoMath.exp(s2[2] * scale - x1) * i1 : 0.0f;
            float w23 =
                    c + 17 <= p1 && c + 17 >= lo1 ? TornadoMath.exp(s2[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 8] =
                    (new HalfFloat(w20).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w21).getHalfFloatValue() << 16);
            pTile[pRow1 + 8] =
                    (new HalfFloat(w22).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w23).getHalfFloatValue() << 16);
            float w30 =
                    c + 24 <= p0 && c + 24 >= lo0 ? TornadoMath.exp(s3[0] * scale - x0) * i0 : 0.0f;
            float w31 =
                    c + 25 <= p0 && c + 25 >= lo0 ? TornadoMath.exp(s3[1] * scale - x0) * i0 : 0.0f;
            float w32 =
                    c + 24 <= p1 && c + 24 >= lo1 ? TornadoMath.exp(s3[2] * scale - x1) * i1 : 0.0f;
            float w33 =
                    c + 25 <= p1 && c + 25 >= lo1 ? TornadoMath.exp(s3[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 12] =
                    (new HalfFloat(w30).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w31).getHalfFloatValue() << 16);
            pTile[pRow1 + 12] =
                    (new HalfFloat(w32).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w33).getHalfFloatValue() << 16);
            float w40 =
                    c + 32 <= p0 && c + 32 >= lo0 ? TornadoMath.exp(s4[0] * scale - x0) * i0 : 0.0f;
            float w41 =
                    c + 33 <= p0 && c + 33 >= lo0 ? TornadoMath.exp(s4[1] * scale - x0) * i0 : 0.0f;
            float w42 =
                    c + 32 <= p1 && c + 32 >= lo1 ? TornadoMath.exp(s4[2] * scale - x1) * i1 : 0.0f;
            float w43 =
                    c + 33 <= p1 && c + 33 >= lo1 ? TornadoMath.exp(s4[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 16] =
                    (new HalfFloat(w40).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w41).getHalfFloatValue() << 16);
            pTile[pRow1 + 16] =
                    (new HalfFloat(w42).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w43).getHalfFloatValue() << 16);
            float w50 =
                    c + 40 <= p0 && c + 40 >= lo0 ? TornadoMath.exp(s5[0] * scale - x0) * i0 : 0.0f;
            float w51 =
                    c + 41 <= p0 && c + 41 >= lo0 ? TornadoMath.exp(s5[1] * scale - x0) * i0 : 0.0f;
            float w52 =
                    c + 40 <= p1 && c + 40 >= lo1 ? TornadoMath.exp(s5[2] * scale - x1) * i1 : 0.0f;
            float w53 =
                    c + 41 <= p1 && c + 41 >= lo1 ? TornadoMath.exp(s5[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 20] =
                    (new HalfFloat(w50).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w51).getHalfFloatValue() << 16);
            pTile[pRow1 + 20] =
                    (new HalfFloat(w52).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w53).getHalfFloatValue() << 16);
            float w60 =
                    c + 48 <= p0 && c + 48 >= lo0 ? TornadoMath.exp(s6[0] * scale - x0) * i0 : 0.0f;
            float w61 =
                    c + 49 <= p0 && c + 49 >= lo0 ? TornadoMath.exp(s6[1] * scale - x0) * i0 : 0.0f;
            float w62 =
                    c + 48 <= p1 && c + 48 >= lo1 ? TornadoMath.exp(s6[2] * scale - x1) * i1 : 0.0f;
            float w63 =
                    c + 49 <= p1 && c + 49 >= lo1 ? TornadoMath.exp(s6[3] * scale - x1) * i1 : 0.0f;
            pTile[pRow0 + 24] =
                    (new HalfFloat(w60).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w61).getHalfFloatValue() << 16);
            pTile[pRow1 + 24] =
                    (new HalfFloat(w62).getHalfFloatValue() & 0xFFFF)
                            | (new HalfFloat(w63).getHalfFloatValue() << 16);
            float w70 =
                    c + 56 <= p0 && c + 56 >= lo0 ? TornadoMath.exp(s7[0] * scale - x0) * i0 : 0.0f;
            float w71 =
                    c + 57 <= p0 && c + 57 >= lo0 ? TornadoMath.exp(s7[1] * scale - x0) * i0 : 0.0f;
            float w72 =
                    c + 56 <= p1 && c + 56 >= lo1 ? TornadoMath.exp(s7[2] * scale - x1) * i1 : 0.0f;
            float w73 =
                    c + 57 <= p1 && c + 57 >= lo1 ? TornadoMath.exp(s7[3] * scale - x1) * i1 : 0.0f;
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
            HalfFloat[] pa1 = ctx.mmaLoadA(pTile, 72, pBytes + 32);
            o0 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 256), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 1280), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 2304), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 3328), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 4352), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 5376), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 6400), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa1, ctx.mmaLoadB(vTile, 16, 7424), o7, MMAShape.M16N8K16);
            HalfFloat[] pa2 = ctx.mmaLoadA(pTile, 72, pBytes + 64);
            o0 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 512), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 1536), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 2560), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 3584), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 4608), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 5632), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 6656), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa2, ctx.mmaLoadB(vTile, 16, 7680), o7, MMAShape.M16N8K16);
            HalfFloat[] pa3 = ctx.mmaLoadA(pTile, 72, pBytes + 96);
            o0 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 768), o0, MMAShape.M16N8K16);
            o1 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 1792), o1, MMAShape.M16N8K16);
            o2 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 2816), o2, MMAShape.M16N8K16);
            o3 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 3840), o3, MMAShape.M16N8K16);
            o4 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 4864), o4, MMAShape.M16N8K16);
            o5 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 5888), o5, MMAShape.M16N8K16);
            o6 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 6912), o6, MMAShape.M16N8K16);
            o7 = ctx.mma(pa3, ctx.mmaLoadB(vTile, 16, 7936), o7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int ob0 = r0 * qDim + qCol + 2 * quad;
        int ob1 = ob0 + 8 * qDim;
        boolean v0 = r0 < tokens;
        boolean v1 = r0 + 8 < tokens;
        // The value goes through a local: TC drops a conditional written inside new HalfFloat(...).
        float y00 = v0 ? o0[0] : 0.0f;
        out.set(ob0, y00);
        float y01 = v0 ? o0[1] : 0.0f;
        out.set(ob0 + 1, y01);
        float y02 = v1 ? o0[2] : 0.0f;
        out.set(ob1, y02);
        float y03 = v1 ? o0[3] : 0.0f;
        out.set(ob1 + 1, y03);
        float y10 = v0 ? o1[0] : 0.0f;
        out.set(ob0 + 8, y10);
        float y11 = v0 ? o1[1] : 0.0f;
        out.set(ob0 + 8 + 1, y11);
        float y12 = v1 ? o1[2] : 0.0f;
        out.set(ob1 + 8, y12);
        float y13 = v1 ? o1[3] : 0.0f;
        out.set(ob1 + 8 + 1, y13);
        float y20 = v0 ? o2[0] : 0.0f;
        out.set(ob0 + 16, y20);
        float y21 = v0 ? o2[1] : 0.0f;
        out.set(ob0 + 16 + 1, y21);
        float y22 = v1 ? o2[2] : 0.0f;
        out.set(ob1 + 16, y22);
        float y23 = v1 ? o2[3] : 0.0f;
        out.set(ob1 + 16 + 1, y23);
        float y30 = v0 ? o3[0] : 0.0f;
        out.set(ob0 + 24, y30);
        float y31 = v0 ? o3[1] : 0.0f;
        out.set(ob0 + 24 + 1, y31);
        float y32 = v1 ? o3[2] : 0.0f;
        out.set(ob1 + 24, y32);
        float y33 = v1 ? o3[3] : 0.0f;
        out.set(ob1 + 24 + 1, y33);
        float y40 = v0 ? o4[0] : 0.0f;
        out.set(ob0 + 32, y40);
        float y41 = v0 ? o4[1] : 0.0f;
        out.set(ob0 + 32 + 1, y41);
        float y42 = v1 ? o4[2] : 0.0f;
        out.set(ob1 + 32, y42);
        float y43 = v1 ? o4[3] : 0.0f;
        out.set(ob1 + 32 + 1, y43);
        float y50 = v0 ? o5[0] : 0.0f;
        out.set(ob0 + 40, y50);
        float y51 = v0 ? o5[1] : 0.0f;
        out.set(ob0 + 40 + 1, y51);
        float y52 = v1 ? o5[2] : 0.0f;
        out.set(ob1 + 40, y52);
        float y53 = v1 ? o5[3] : 0.0f;
        out.set(ob1 + 40 + 1, y53);
        float y60 = v0 ? o6[0] : 0.0f;
        out.set(ob0 + 48, y60);
        float y61 = v0 ? o6[1] : 0.0f;
        out.set(ob0 + 48 + 1, y61);
        float y62 = v1 ? o6[2] : 0.0f;
        out.set(ob1 + 48, y62);
        float y63 = v1 ? o6[3] : 0.0f;
        out.set(ob1 + 48 + 1, y63);
        float y70 = v0 ? o7[0] : 0.0f;
        out.set(ob0 + 56, y70);
        float y71 = v0 ? o7[1] : 0.0f;
        out.set(ob0 + 56 + 1, y71);
        float y72 = v1 ? o7[2] : 0.0f;
        out.set(ob1 + 56, y72);
        float y73 = v1 ? o7[3] : 0.0f;
        out.set(ob1 + 56 + 1, y73);
    }
}
