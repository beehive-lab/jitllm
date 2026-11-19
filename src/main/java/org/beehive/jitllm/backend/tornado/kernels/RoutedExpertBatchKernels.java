package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched-prefill kernels of a mixture of experts on int8 tensor cores, shared by the families that
 * route with sigmoid scores (GLM-5.3-Flash, Kimi-Linear): the grouped gate/up and down GEMMs over
 * the tiles {@link Qwen35MoeBatchKernels#groupByExpert} cuts, with Q8_0 experts or with Q4_0
 * experts recentred to int8 as their tiles are staged, an optional SwiGLU clamp, and the weighted
 * combination with the shared expert. The staging and fragment helpers are {@link
 * Qwen35MoeBatchKernels}'.
 */
public final class RoutedExpertBatchKernels {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 34;
    private static final int Q4_BLOCK_BYTES = 18;
    private static final int TILE_ROWS = Qwen35MoeBatchKernels.TILE_ROWS;
    private static final int TILE_COLS = Qwen35MoeBatchKernels.TILE_COLS;
    private static final int SUB_TILES = Qwen35MoeBatchKernels.SUB_TILES;
    private static final int A_WORDS = Qwen35MoeBatchKernels.A_WORDS;
    private static final int B_WORDS = Qwen35MoeBatchKernels.B_WORDS;

    private RoutedExpertBatchKernels() {}

    // @formatter:off
    /**
     * {@link Qwen35MoeBatchKernels#groupedGateUpQ8_0} with the SwiGLU clamp, over one stack of
     * experts: the tiles of experts {@code [lo, hi)} only, which this stack holds from its first
     * row. A model with two stacks launches it once per stack, so no launch chooses between them.
     * {@code hidden[row][c] = silu(min(g, limit)) * clamp(u, -limit, limit)} for every sorted row.
     * Worker: {@code WorkerGrid2D((hiddenDim / 128) * 256, maxTiles)}, local 256.
     */
    // @formatter:on
    public static void groupedGateUpClamped(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            IntArray sortedToken,
            IntArray tiles,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int lo,
            int hi,
            float limit) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        int expert = tiles.get(1 + 3 * tile);
        if (tile < tiles.get(0) && blockCol < hiddenDim && expert >= lo && expert < hi) {
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = dim / (2 * QK);
            int rowBytes = (dim / QK) * BLOCK_BYTES;
            int firstRow = (expert - lo) * hiddenDim + blockCol;

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] gTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            int[] uTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sG = ctx.allocateFloatLocalArray(2 * 256);
            float[] sU = ctx.allocateFloatLocalArray(2 * 256);

            float[] gate = new float[SUB_TILES * 8];
            float[] up = new float[SUB_TILES * 8];
            for (int i = 0; i < SUB_TILES * 8; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int activeSub = (rows + 15) >> 4;

            Qwen35MoeBatchKernels.stageA(
                    ctx, aTile, sA, a8, dA, sortedToken, 1, rowStart, rows, 0, 0, dim, tid);
            Qwen35MoeBatchKernels.stageB(gTile, sG, gateExperts, firstRow, rowBytes, 0, 0, tid);
            Qwen35MoeBatchKernels.stageB(uTile, sU, upExperts, firstRow, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    Qwen35MoeBatchKernels.stageA(
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
                    Qwen35MoeBatchKernels.stageB(
                            gTile, sG, gateExperts, firstRow, rowBytes, round + 1, next, tid);
                    Qwen35MoeBatchKernels.stageB(
                            uTile, sU, upExperts, firstRow, rowBytes, round + 1, next, tid);
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
                            hidden.set(
                                    (rowStart + row) * hiddenDim + col + (e & 1),
                                    clampedSilu(
                                            gate[t * 8 + j * 4 + e], up[t * 8 + j * 4 + e], limit));
                        }
                    }
                }
            }
        }
    }

    // @formatter:off
    /**
     * {@link Qwen35MoeBatchKernels#groupedDownQ8_0} over one stack of experts, the tiles of experts
     * {@code [lo, hi)}: {@code out[row][c] = down_e[c] . hidden[row]} over the quantized hidden
     * activations in sorted order. Worker: {@code WorkerGrid2D((dim / 128) * 256, maxTiles)}, local
     * 256.
     */
    // @formatter:on
    public static void groupedDown(
            KernelContext ctx,
            ByteArray h8,
            FloatArray dH,
            IntArray tiles,
            ByteArray downExperts,
            FloatArray out,
            int dim,
            int hiddenDim,
            int lo,
            int hi) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        int expert = tiles.get(1 + 3 * tile);
        if (tile < tiles.get(0) && blockCol < dim && expert >= lo && expert < hi) {
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = hiddenDim / (2 * QK);
            int rowBytes = (hiddenDim / QK) * BLOCK_BYTES;
            int firstRow = (expert - lo) * dim + blockCol;

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

            Qwen35MoeBatchKernels.stageA(
                    ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, 0, 0, hiddenDim, tid);
            Qwen35MoeBatchKernels.stageB(bTile, sW, downExperts, firstRow, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    Qwen35MoeBatchKernels.stageA(
                            ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, round + 1, next,
                            hiddenDim, tid);
                    ctx.asyncCopyCommit();
                    Qwen35MoeBatchKernels.stageB(
                            bTile, sW, downExperts, firstRow, rowBytes, round + 1, next, tid);
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
     * {@code branch[t] = sum over slots of weight[t][slot] * routed[position(t, slot)] + shared[t]}
     * for the real rows, the slots in order. Worker: {@code batch * dim} lanes.
     */
    public static void combineExperts(
            KernelContext context,
            FloatArray routed,
            IntArray position,
            FloatArray weights,
            FloatArray shared,
            FloatArray branch,
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
            branch.set(i, sum + shared.get(i));
        }
    }

    /** {@link #groupedGateUpClamped} over a stack of Q4_0 experts. */
    public static void groupedGateUpClampedQ4_0(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            IntArray sortedToken,
            IntArray tiles,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray hidden,
            int dim,
            int hiddenDim,
            int lo,
            int hi,
            float limit) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        int expert = tiles.get(1 + 3 * tile);
        if (tile < tiles.get(0) && blockCol < hiddenDim && expert >= lo && expert < hi) {
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = dim / (2 * QK);
            int rowBytes = (dim / QK) * Q4_BLOCK_BYTES;
            int firstRow = (expert - lo) * hiddenDim + blockCol;

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * A_WORDS);
            int[] gTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            int[] uTile = ctx.allocateIntLocalArray(2 * 2 * B_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * TILE_ROWS);
            float[] sG = ctx.allocateFloatLocalArray(2 * 256);
            float[] sU = ctx.allocateFloatLocalArray(2 * 256);

            float[] gate = new float[SUB_TILES * 8];
            float[] up = new float[SUB_TILES * 8];
            for (int i = 0; i < SUB_TILES * 8; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int activeSub = (rows + 15) >> 4;

            Qwen35MoeBatchKernels.stageA(
                    ctx, aTile, sA, a8, dA, sortedToken, 1, rowStart, rows, 0, 0, dim, tid);
            Qwen35MoeBatchKernels.stageBQ4_0(gTile, sG, gateExperts, firstRow, rowBytes, 0, 0, tid);
            Qwen35MoeBatchKernels.stageBQ4_0(uTile, sU, upExperts, firstRow, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    Qwen35MoeBatchKernels.stageA(
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
                    Qwen35MoeBatchKernels.stageBQ4_0(
                            gTile, sG, gateExperts, firstRow, rowBytes, round + 1, next, tid);
                    Qwen35MoeBatchKernels.stageBQ4_0(
                            uTile, sU, upExperts, firstRow, rowBytes, round + 1, next, tid);
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
                            hidden.set(
                                    (rowStart + row) * hiddenDim + col + (e & 1),
                                    clampedSilu(
                                            gate[t * 8 + j * 4 + e], up[t * 8 + j * 4 + e], limit));
                        }
                    }
                }
            }
        }
    }

    /** {@link #groupedDown} over a stack of Q4_0 experts. */
    public static void groupedDownQ4_0(
            KernelContext ctx,
            ByteArray h8,
            FloatArray dH,
            IntArray tiles,
            ByteArray downExperts,
            FloatArray out,
            int dim,
            int hiddenDim,
            int lo,
            int hi) {
        int tid = ctx.localIdx;
        int tile = ctx.groupIdy;
        int blockCol = TILE_COLS * ctx.groupIdx;
        int expert = tiles.get(1 + 3 * tile);
        if (tile < tiles.get(0) && blockCol < dim && expert >= lo && expert < hi) {
            int rowStart = tiles.get(2 + 3 * tile);
            int rows = tiles.get(3 + 3 * tile);
            int warp = tid >> 5;
            int lane = tid & 31;
            int rounds = hiddenDim / (2 * QK);
            int rowBytes = (hiddenDim / QK) * Q4_BLOCK_BYTES;
            int firstRow = (expert - lo) * dim + blockCol;

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

            Qwen35MoeBatchKernels.stageA(
                    ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, 0, 0, hiddenDim, tid);
            Qwen35MoeBatchKernels.stageBQ4_0(bTile, sW, downExperts, firstRow, rowBytes, 0, 0, tid);
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            for (int round = 0; round < rounds; round++) {
                int buf = round & 1;
                int next = 1 - buf;
                if (round + 1 < rounds) {
                    Qwen35MoeBatchKernels.stageA(
                            ctx, aTile, sA, h8, dH, tiles, 0, rowStart, rows, round + 1, next,
                            hiddenDim, tid);
                    ctx.asyncCopyCommit();
                    Qwen35MoeBatchKernels.stageBQ4_0(
                            bTile, sW, downExperts, firstRow, rowBytes, round + 1, next, tid);
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

    private static float clampedSilu(float gate, float up, float limit) {
        float g = TornadoMath.min(gate, limit);
        float u = TornadoMath.max(-limit, TornadoMath.min(up, limit));
        return g / (1.0f + TornadoMath.exp(-g)) * u;
    }
}
