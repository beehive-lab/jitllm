package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The mixture-of-experts feed-forward of {@code qwen35moe} over a chunk of prompt tokens.
 *
 * <ol>
 *   <li>{@link #routerBatch} and {@link #routerTopKBatch}: every token's expert scores and shared
 *       gate, then its {@code used} experts and their weights, as {@link Qwen35MoeKernels} does for
 *       one token.
 *   <li>{@link #groupByExpert}: the chunk's (token, slot) assignments sorted by expert, in token
 *       order within an expert, and cut into tiles of at most {@link #TILE_ROWS} rows of one
 *       expert. A chunk of 256 tokens routes 2048 assignments over 256 experts, about eight an
 *       expert, which is why the tiles are 16 rows and not the dense GEMM's 128.
 *   <li>{@link #groupedGateUpQ8_0} and {@link #groupedDownQ8_0}: a block-scaled int8 tensor-core
 *       GEMM per tile, reading the expert's rows in place from the stacked Q8_0 tensors; the
 *       gate/up one gathers its activation rows by token and writes {@code silu(gate) * up}.
 *   <li>{@link #combine}: each token's routed outputs, weighted, in slot order, and its shared
 *       expert's output times its gate, added to the residual stream.
 * </ol>
 *
 * <p>The tile table is {@code tiles[0]} = the number of tiles, then {@code (expert, first sorted
 * row, rows)} per tile. A GEMM is launched for the most tiles a chunk can produce ({@link
 * #maxTiles}); a block whose tile does not exist returns before its first barrier.
 *
 * <p>The GEMMs are the arithmetic of {@link Int8GemmKernels#gemmInt8Q8_0} on a 64 x 128 tile: eight
 * warps, each the tile's rows by 16 columns, a round of two 32-blocks, {@code m16n8k32} int8 MMAs,
 * and each block's product scaled by the activation's and the weight's block scales before it is
 * accumulated in FP32.
 */
// @formatter:on
public final class Qwen35MoeBatchKernels {

    private Qwen35MoeBatchKernels() {}

    /** Rows of one expert tile: four 16-row MMA sub-tiles sharing each staged weight tile. */
    public static final int TILE_ROWS = 64;

    private static final int SUB_TILES = TILE_ROWS / 16;

    /** Output columns of one GEMM block. */
    public static final int TILE_COLS = 128;

    /** Threads of one GEMM block: eight warps. */
    public static final int GEMM_THREADS = 256;

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;

    /** Words of one round's activation tile per 32-block: 64 rows of 32 bytes. */
    private static final int A_WORDS = 512;

    /** Words of one round's weight tile per 32-block: 128 columns of 32 bytes. */
    private static final int B_WORDS = 1024;

    /** The most tiles a chunk of {@code assignments} over {@code experts} experts can cut into. */
    public static int maxTiles(int assignments, int experts) {
        return experts + (assignments + TILE_ROWS - 1) / TILE_ROWS;
    }

    private static float warpSum(KernelContext context, float value) {
        float v = value;
        v += context.simdShuffleDown(v, 16);
        v += context.simdShuffleDown(v, 8);
        v += context.simdShuffleDown(v, 4);
        v += context.simdShuffleDown(v, 2);
        v += context.simdShuffleDown(v, 1);
        return v;
    }

    /**
     * {@code logits[t][e] = router[e] . x[t]} and {@code sharedGate[t] = sigmoid(gate . x[t])} for
     * the active tokens, a warp per (token, expert or gate). Worker: {@code batch * (experts + 1) *
     * 32} lanes, local 128.
     */
    public static void routerBatch(
            KernelContext context,
            FloatArray x,
            FloatArray router,
            FloatArray sharedGateInput,
            FloatArray logits,
            FloatArray sharedGate,
            IntArray batchInfo,
            int dim,
            int experts) {
        int warp = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        int token = warp / (experts + 1);
        int e = warp - token * (experts + 1);
        // Each branch reduces and stores its own sum: a value merged from an if/else is
        // miscompiled by the CUDA backend (the merge re-assigns the else branch's value).
        if (token < batchInfo.get(1) && e < experts) {
            int xBase = token * dim;
            int base = e * dim;
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += router.get(base + i) * x.get(xBase + i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                logits.set(token * experts + e, sum);
            }
        }
        if (token < batchInfo.get(1) && e == experts) {
            int xBase = token * dim;
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += sharedGateInput.get(i) * x.get(xBase + i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                sharedGate.set(token, 1.0f / (1.0f + TornadoMath.exp(-sum)));
            }
        }
    }

    /** Tokens and experts of one {@link #routerTiled} block. */
    public static final int ROUTER_TILE = 64;

    private static final int ROUTER_K = 32;

    // @formatter:off
    /**
     * {@code logits[t][e] = router[e] . x[t]}, in FP32, as a tiled matrix product: block {@code
     * (groupIdx, groupIdy)} is 64 tokens by 64 experts, staged 32 columns at a time through shared
     * memory, each of the 256 lanes accumulating a 4 x 4 patch. Tokens past the active ones are
     * computed and not stored. Worker: {@code WorkerGrid2D((batch / 64) * 256, experts / 64)},
     * local 256; {@code experts} and {@code dim} multiples of 64 and 32.
     */
    // @formatter:on
    public static void routerTiled(
            KernelContext ctx,
            FloatArray x,
            FloatArray router,
            FloatArray logits,
            IntArray batchInfo,
            int dim,
            int experts) {
        int tid = ctx.localIdx;
        int tokenBase = ROUTER_TILE * ctx.groupIdx;
        int expertBase = ROUTER_TILE * ctx.groupIdy;
        if (tokenBase < batchInfo.get(1) && expertBase < experts) {
            float[] xs = ctx.allocateFloatLocalArray(ROUTER_TILE * (ROUTER_K + 1));
            float[] rs = ctx.allocateFloatLocalArray(ROUTER_TILE * (ROUTER_K + 1));
            int ty = tid >> 4;
            int tx = tid & 15;
            float[] acc = new float[16];
            for (int i = 0; i < 16; i++) {
                acc[i] = 0.0f;
            }
            for (int k0 = 0; k0 < dim; k0 += ROUTER_K) {
                for (int s = 0; s < 8; s++) {
                    int linear = tid + (s << 8);
                    int row = linear >> 5;
                    int col = linear & 31;
                    xs[row * (ROUTER_K + 1) + col] = x.get((tokenBase + row) * dim + k0 + col);
                    rs[row * (ROUTER_K + 1) + col] =
                            router.get((expertBase + row) * dim + k0 + col);
                }
                ctx.localBarrier();
                for (int kk = 0; kk < ROUTER_K; kk++) {
                    float x0 = xs[(ty * 4) * (ROUTER_K + 1) + kk];
                    float x1 = xs[(ty * 4 + 1) * (ROUTER_K + 1) + kk];
                    float x2 = xs[(ty * 4 + 2) * (ROUTER_K + 1) + kk];
                    float x3 = xs[(ty * 4 + 3) * (ROUTER_K + 1) + kk];
                    float r0 = rs[(tx * 4) * (ROUTER_K + 1) + kk];
                    float r1 = rs[(tx * 4 + 1) * (ROUTER_K + 1) + kk];
                    float r2 = rs[(tx * 4 + 2) * (ROUTER_K + 1) + kk];
                    float r3 = rs[(tx * 4 + 3) * (ROUTER_K + 1) + kk];
                    acc[0] += x0 * r0;
                    acc[1] += x0 * r1;
                    acc[2] += x0 * r2;
                    acc[3] += x0 * r3;
                    acc[4] += x1 * r0;
                    acc[5] += x1 * r1;
                    acc[6] += x1 * r2;
                    acc[7] += x1 * r3;
                    acc[8] += x2 * r0;
                    acc[9] += x2 * r1;
                    acc[10] += x2 * r2;
                    acc[11] += x2 * r3;
                    acc[12] += x3 * r0;
                    acc[13] += x3 * r1;
                    acc[14] += x3 * r2;
                    acc[15] += x3 * r3;
                }
                ctx.localBarrier();
            }
            int active = batchInfo.get(1);
            for (int i = 0; i < 4; i++) {
                int token = tokenBase + ty * 4 + i;
                if (token < active) {
                    for (int j = 0; j < 4; j++) {
                        logits.set(token * experts + expertBase + tx * 4 + j, acc[i * 4 + j]);
                    }
                }
            }
        }
    }

    /**
     * {@code sharedGate[t] = sigmoid(gate . x[t])} for the active tokens, a warp each. Worker:
     * {@code batch * 32} lanes, local 128.
     */
    public static void sharedGateBatch(
            KernelContext context,
            FloatArray x,
            FloatArray sharedGateInput,
            FloatArray sharedGate,
            IntArray batchInfo,
            int dim) {
        int token = context.globalIdx >> 5;
        int lane = context.localIdx & 31;
        if (token < batchInfo.get(1)) {
            float sum = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                sum += sharedGateInput.get(i) * x.get(token * dim + i);
            }
            sum = warpSum(context, sum);
            if (lane == 0) {
                sharedGate.set(token, 1.0f / (1.0f + TornadoMath.exp(-sum)));
            }
        }
    }

    /**
     * {@link Qwen35MoeKernels#routerTopK} for every active token, a 32-lane block each. Worker:
     * {@code batch * 32} lanes, local 32.
     */
    public static void routerTopKBatch(
            KernelContext context,
            FloatArray logits,
            IntArray ids,
            FloatArray weights,
            IntArray batchInfo,
            int experts,
            int used) {
        int token = context.groupIdx;
        int lane = context.localIdx;
        float[] scores = context.allocateFloatLocalArray(Qwen35MoeKernels.MAX_EXPERTS);
        if (token < batchInfo.get(1)) {
            for (int e = lane; e < experts; e += 32) {
                scores[e] = logits.get(token * experts + e);
            }
        }
        context.localBarrier();
        if (token < batchInfo.get(1)) {
            float top = 0.0f;
            float total = 0.0f;
            float mine = 0.0f;
            for (int k = 0; k < used; k++) {
                float best = Float.NEGATIVE_INFINITY;
                float bestIndex = 1.0e9f;
                for (int e = lane; e < experts; e += 32) {
                    float s = scores[e];
                    if (s > best) {
                        best = s;
                        bestIndex = e;
                    }
                }
                float max = best;
                max = TornadoMath.max(max, context.simdShuffleDown(max, 16));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 8));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 4));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 2));
                max = TornadoMath.max(max, context.simdShuffleDown(max, 1));
                max = context.simdBroadcastFirst(max);
                float index = best == max ? bestIndex : 1.0e9f;
                index = TornadoMath.min(index, context.simdShuffleDown(index, 16));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 8));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 4));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 2));
                index = TornadoMath.min(index, context.simdShuffleDown(index, 1));
                int chosen = (int) context.simdBroadcastFirst(index);
                if (k == 0) {
                    top = max;
                }
                float w = TornadoMath.exp(max - top);
                total += w;
                if (lane == k) {
                    mine = w;
                }
                if ((chosen & 31) == lane) {
                    scores[chosen] = Float.NEGATIVE_INFINITY;
                }
                if (lane == 0) {
                    ids.set(token * used + k, chosen);
                }
            }
            if (lane < used) {
                weights.set(token * used + lane, mine / total);
            }
        }
    }

    /** Lanes of the {@link #groupByExpert} block. */
    public static final int GROUP_THREADS = 1024;

    // @formatter:off
    /**
     * Sorts the active tokens' assignments by expert and cuts them into tiles, in one block.
     *
     * <p>The lanes count every expert's assignments with shared-memory atomics, lane zero turns the
     * counts into offsets and the tile table, and the lanes then claim each assignment a row of its
     * expert's range with an atomic increment: {@code sortedToken[row]} is the token of sorted row
     * {@code row}, and {@code position[token * used + slot]} the sorted row of that assignment.
     * Which row of its expert's range an assignment gets depends on timing, and nothing computed
     * depends on it: every row of a tile is computed from its own token alone, and {@link #combine}
     * reads each assignment back through {@code position}. Worker: one block of {@link
     * #GROUP_THREADS} lanes ({@code experts} at most that).
     */
    // @formatter:on
    public static void groupByExpert(
            KernelContext context,
            IntArray ids,
            IntArray batchInfo,
            IntArray sortedToken,
            IntArray position,
            IntArray tiles,
            int experts,
            int used,
            int batch) {
        int tid = context.localIdx;
        int[] counts = context.allocateIntLocalArray(GROUP_THREADS);
        int[] cursor = context.allocateIntLocalArray(GROUP_THREADS);
        int total = batchInfo.get(1) * used;
        if (tid < experts) {
            counts[tid] = 0;
        }
        context.localBarrier();
        for (int a = tid; a < total; a += GROUP_THREADS) {
            context.atomicAdd(counts, ids.get(a), 1);
        }
        context.localBarrier();
        if (tid == 0) {
            int running = 0;
            int tileCount = 0;
            for (int x = 0; x < experts; x++) {
                int c = counts[x];
                cursor[x] = running;
                for (int start = 0; start < c; start += TILE_ROWS) {
                    tiles.set(1 + 3 * tileCount, x);
                    tiles.set(2 + 3 * tileCount, running + start);
                    tiles.set(3 + 3 * tileCount, Math.min(TILE_ROWS, c - start));
                    tileCount++;
                }
                running += c;
            }
            tiles.set(0, tileCount);
        }
        context.localBarrier();
        for (int a = tid; a < total; a += GROUP_THREADS) {
            int e = ids.get(a);
            int expected = cursor[e];
            int seen = context.atomicCAS(cursor, e, expected, expected + 1);
            while (seen != expected) {
                expected = seen;
                seen = context.atomicCAS(cursor, e, expected, expected + 1);
            }
            sortedToken.set(expected, a / used);
            position.set(a, expected);
        }
    }

    private static int halfBits(ByteArray w, int offset) {
        return w.getHalfFloat(offset).getHalfFloatValue() & 0xFFFF;
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
            int sorted = rowStart + (row < rows ? row : 0);
            int source = gather != 0 ? sortedToken.get(sorted) : sorted;
            ctx.asyncCopyToLocal(
                    aTile,
                    buf * 2 * A_WORDS + linear,
                    a8,
                    source * k + (kb0 + b) * QK + (quad << 2));
        }
        if (tid < 2 * TILE_ROWS) {
            int sb = tid >> 6;
            int sr = tid & 63;
            int sSorted = rowStart + (sr < rows ? sr : 0);
            int sSource = gather != 0 ? sortedToken.get(sSorted) : sSorted;
            sA[buf * 2 * TILE_ROWS + sb * TILE_ROWS + sr] = dA.get(sSource * kBlocks + kb0 + sb);
        }
    }

    /**
     * Stages one round's weight tile of one matrix — 128 columns starting at weight row {@code
     * firstRow}, two blocks — and its scales: eight words a lane, read with 16-bit loads.
     */
    private static void stageB(
            int[] bTile,
            float[] sW,
            ByteArray w,
            int firstRow,
            int rowBytes,
            int round,
            int buf,
            int tid) {
        int kRow = tid & 15;
        int colPair = (tid >> 4) & 15;
        int slot = ((colPair >> 2) << 6) + (kRow << 2) + (colPair & 3);
        int base =
                (firstRow + (colPair << 1)) * rowBytes + round * 2 * BLOCK_BYTES + 2 + (kRow << 1);
        int dst = buf * 2 * B_WORDS + slot;
        for (int s = 0; s < 8; s++) {
            int offset = base + (s >> 2) * BLOCK_BYTES + (s & 3) * 32 * rowBytes;
            bTile[dst + (s >> 2) * B_WORDS + (s & 3) * 256] =
                    halfBits(w, offset) | (halfBits(w, offset + rowBytes) << 16);
        }
        int b = tid >> 7;
        int col = tid & 127;
        sW[buf * 256 + b * 128 + col] =
                w.getHalfFloat((firstRow + col) * rowBytes + (round * 2 + b) * BLOCK_BYTES)
                        .getFloat32();
    }

    // @formatter:off
    /**
     * {@code hidden[row][c] = silu(gate_e[c] . x[token(row)]) * (up_e[c] . x[token(row)])} for
     * every sorted row of every tile, {@code e} the tile's expert, over the quantized chunk.
     *
     * <p>Block {@code (groupIdx, groupIdy)} is column block {@code groupIdx} of tile {@code
     * groupIdy}. Warp {@code w} computes the tile's 16 rows by columns {@code 16 w .. 16 w + 15} of
     * both projections. Worker: {@code WorkerGrid2D((hiddenDim / 128) * 256, maxTiles)}, local 256.
     */
    // @formatter:on
    public static void groupedGateUpQ8_0(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            IntArray sortedToken,
            IntArray tiles,
            ByteArray gateExperts,
            ByteArray upExperts,
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
            int rowBytes = (dim / QK) * BLOCK_BYTES;
            int firstRow = expert * hiddenDim + blockCol;

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] gTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            int[] uTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sG = ctx.allocateFloatLocalArray(2 * 256);
            float[] sU = ctx.allocateFloatLocalArray(2 * 256);

            // Sub-tile t, n8 tile j, element e at [t * 8 + j * 4 + e].
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
            stageB(gTile, sG, gateExperts, firstRow, rowBytes, 0, 0, tid);
            stageB(uTile, sU, upExperts, firstRow, rowBytes, 0, 0, tid);
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
                    stageB(gTile, sG, gateExperts, firstRow, rowBytes, round + 1, next, tid);
                    stageB(uTile, sU, upExperts, firstRow, rowBytes, round + 1, next, tid);
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
                    int col = blockCol + warp * 16 + j * 8 + colInWarp;
                    for (int e = 0; e < 4; e++) {
                        int row = t * 16 + rowInWarp + 8 * (e >> 1);
                        if (row < rows) {
                            float g = gate[t * 8 + j * 4 + e];
                            hidden.set(
                                    (rowStart + row) * hiddenDim + col + (e & 1),
                                    g / (1.0f + TornadoMath.exp(-g)) * up[t * 8 + j * 4 + e]);
                        }
                    }
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@code out[row][c] = down_e[c] . hidden[row]} for every sorted row of every tile, over the
     * quantized hidden activations (sorted order, so nothing is gathered). Block {@code (groupIdx,
     * groupIdy)} is column block {@code groupIdx} of tile {@code groupIdy}. Worker: {@code
     * WorkerGrid2D((dim / 128) * 256, maxTiles)}, local 256.
     */
    // @formatter:on
    public static void groupedDownQ8_0(
            KernelContext ctx,
            ByteArray h8,
            FloatArray dH,
            IntArray tiles,
            ByteArray downExperts,
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
            int rowBytes = (hiddenDim / QK) * BLOCK_BYTES;
            int firstRow = expert * dim + blockCol;

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sW = ctx.allocateFloatLocalArray(2 * 256);

            float[] acc = new float[SUB_TILES * 8];
            for (int i = 0; i < SUB_TILES * 8; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int activeSub = (rows + 15) >> 4;

            stageA(ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, 0, 0, hiddenDim, tid);
            stageB(bTile, sW, downExperts, firstRow, rowBytes, 0, 0, tid);
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
                    stageB(bTile, sW, downExperts, firstRow, rowBytes, round + 1, next, tid);
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
                    int col = blockCol + warp * 16 + j * 8 + colInWarp;
                    for (int e = 0; e < 4; e++) {
                        int row = t * 16 + rowInWarp + 8 * (e >> 1);
                        if (row < rows) {
                            out.set((rowStart + row) * dim + col + (e & 1), acc[t * 8 + j * 4 + e]);
                        }
                    }
                }
            }
        }
    }

    /**
     * {@code residual[t][c] += sum over slots of weight[t][slot] * routed[position(t, slot)][c] +
     * sharedGate[t] * shared[t][c]}, the slots in order, for the active tokens. Worker: {@code
     * batch * dim} lanes, local 256.
     */
    public static void combine(
            KernelContext context,
            FloatArray routed,
            IntArray position,
            FloatArray weights,
            FloatArray shared,
            FloatArray sharedGate,
            FloatArray residual,
            IntArray batchInfo,
            int dim,
            int used) {
        int i = context.globalIdx;
        int token = i / dim;
        if (token < batchInfo.get(1)) {
            int c = i - token * dim;
            float sum = 0.0f;
            for (int slot = 0; slot < used; slot++) {
                int a = token * used + slot;
                sum += weights.get(a) * routed.get(position.get(a) * dim + c);
            }
            sum += sharedGate.get(token) * shared.get(i);
            residual.set(i, residual.get(i) + sum);
        }
    }
}
