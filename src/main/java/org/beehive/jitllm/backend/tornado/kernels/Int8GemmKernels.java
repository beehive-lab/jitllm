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
 * Research kernels: Q4_0 projections on the int8 tensor path with per-block FP32 scaling (the
 * llama.cpp MMQ arithmetic), second design.
 *
 * <p><b>Arithmetic, fixed and explicit.</b> Activations are quantized per 32-wide block to int8 as
 * {@code round-half-away(x * 127 / amax)} with the FP32 scale {@code dA = amax / 127} (a zero block
 * keeps {@code dA = 0} and {@code q = 0}). Weights are the Q4_0 nibbles recentred to {@code q - 8}
 * with their FP16 block scale {@code dW} widened to FP32. Each 32-block's dot product is exact in
 * int32 on {@code m16n8k32 s8.s8.s32}; the output accumulates {@code (float) dot * dA * dW} in FP32
 * over the blocks in k order, {@code ((float) dot * dA)} rounded then fused-multiplied into the
 * accumulator. No FP16 expansion, no combining differently scaled blocks before scaling.
 *
 * <p><b>Against llama.cpp's MMQ (e2d2c0d6a, {@code quantize_mmq_q8_1} and the Q4_0 mma path with
 * the DS4 layout), read rather than assumed.</b> Same weight recentring ({@code q - 8} as int8),
 * same FP16 weight scale widened to FP32, same activation rounding ({@code roundf(x * (127 /
 * amax))}), same per-block association ({@code sum += C * dA * dB} with the block products in k
 * order). Two differences: MMQ stores the activation scale as {@code 1 / (127 / amax)} in
 * <b>FP16</b> ({@code __low2float(y_ds)}), this path keeps {@code amax / 127} in FP32; and MMQ's
 * k-iteration groups blocks by 256 within a tile while this path takes them two at a time, which
 * changes nothing in the per-element association. Not bit-identical to MMQ; the activation scale
 * here is the more precise of the two.
 *
 * <p><b>What differs from the first prototype.</b> That kernel staged 32 k per round with plain
 * global loads (four words of activations and sixteen weight bytes per lane, decoded on the way),
 * read every block scale from global memory inside the column loop and reloaded its A fragments for
 * each column tile: two barriers and an exposed load latency per 32 k. Here the weights are decoded
 * once per chunk by {@link #decodeQ4_0ToInt8Tiled} into the B operand's word layout with their
 * scales as FP32, the activations once by {@link #quantizeActivationsQ8Warp}, and the GEMM copies
 * both int8 tiles global-to-shared with {@code cp.async} in 64-k rounds, double-buffered, with the
 * round's 512 scales staged beside them; A fragments are loaded once per block and held across the
 * eight column tiles.
 *
 * <p>These kernels use the constant-index int32 accumulator reads and byte-offset int8 fragment
 * loads TornadoVM develop 65f06c5d1 lowers (PRs #1101 and #1094). The batched prefill dispatches
 * them for every Q4_0 projection the FP16 pair would otherwise take, on a tensor-core device.
 */
// @formatter:on
public final class Int8GemmKernels {

    private Int8GemmKernels() {}

    private static final int Q8_BLOCK = 32;

    private static final int BLOCK_BYTES = 18;

    /** Rows, columns and k of one GEMM round. */
    public static final int I8_BM = 128;

    public static final int I8_BN = 128;

    /** k per staged round: two 32-blocks. */
    public static final int I8_BK = 64;

    private static final int I8_WARPS_N = 2;

    private static final int I8_WM = 32;

    private static final int I8_WN = 64;

    /** Words of one 128 x 32 int8 tile. */
    private static final int TILE_WORDS = 1024;

    // @formatter:off
    /**
     * Per-32-block int8 quantization of the FP32 chunk, a lane per element: the 32 lanes of a block
     * find the block's {@code amax} by a shuffle-max (exact), then each quantizes its own element
     * and lane zero writes the scale. Worker: {@code m * k} lanes, local a multiple of 32.
     *
     * <p><b>Definition, including the small values.</b> {@code d = amax / 127} in FP32. If {@code
     * d} is zero — the block is all zeros, or {@code amax} is below about 8.9e-44 so the scale has
     * no FP32 representation — every element quantizes to zero and the scale is zero. Otherwise
     * {@code q = round-half-away-from-zero(v)} clamped to {@code [-127, 127]}, where {@code v = x *
     * (127 / amax)} when that reciprocal is finite ({@code amax >= 4e-37}, every ordinary
     * activation; this is the arithmetic the exactness tests pin) and {@code v = (x / amax) * 127}
     * below it, where {@code 127 / amax} would overflow to infinity and the product would saturate
     * the integer conversion. {@code x / amax} is in {@code [-1, 1]} for any finite input, so the
     * slow form never leaves the byte range; the clamp is a guard that ordinary inputs never reach.
     * The scale may be subnormal (amax below 1.5e-36); the GEMM multiplies it in FP32 and the
     * contribution is simply tiny. A NaN element is not defined here: the device's {@code max}
     * ignores it and its own byte converts to zero, the host reference's does not — a NaN
     * activation is a fault upstream of this kernel, not an input it accepts.
     */
    // @formatter:on
    public static void quantizeActivationsQ8Warp(
            KernelContext ctx, FloatArray x, ByteArray q8, FloatArray scales, int k) {
        int lane = ctx.globalIdx;
        int sub = ctx.localIdx & 31;
        float v0 = x.get(lane);
        float amax = TornadoMath.abs(v0);
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 16));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 8));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 4));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 2));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 1));
        amax = ctx.simdBroadcastFirst(amax);
        float d = amax / 127.0f;
        int qv = 0;
        if (d > 0.0f) {
            float v;
            if (amax >= RECIPROCAL_FINITE_AMAX) {
                v = v0 * (127.0f / amax);
            } else {
                v = (v0 / amax) * 127.0f;
            }
            qv = (int) TornadoMath.floor(TornadoMath.abs(v) + 0.5f);
            if (qv > 127) {
                qv = 127;
            }
            if (v < 0.0f) {
                qv = -qv;
            }
        }
        q8.set(lane, (byte) qv);
        if (sub == 0) {
            scales.set(lane >> 5, d);
        }
    }

    /**
     * The smallest {@code amax} for which {@code 127 / amax} is a finite FP32 (127 / FLT_MAX is
     * 3.73e-37), rounded up.
     */
    public static final float RECIPROCAL_FINITE_AMAX = 4.0e-37f;

    /**
     * Diagnostic only (never dispatched by production): the FP16 staging of an activation with the
     * int8 path's quantization applied and undone — {@code half(q * d)} with the same block {@code
     * amax}, rounding and scale as {@link #quantizeActivationsQ8Warp} — so the FP16 pair can be run
     * on quantized inputs. Isolates the activation quantization from everything else the int8 path
     * changes. Same worker as the conversion it stands in for (a lane per element, local a multiple
     * of 32).
     */
    public static void fakeQuantizeNormedToFP16(
            KernelContext ctx, FloatArray in, HalfFloatArray out, int n, IntArray batchInfo) {
        int index = ctx.globalIdx;
        int row = index / n;
        float v0 = row < batchInfo.get(1) ? in.get(index) : 0.0f;
        float amax = TornadoMath.abs(v0);
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 16));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 8));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 4));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 2));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 1));
        amax = ctx.simdBroadcastFirst(amax);
        float d = amax / 127.0f;
        float v = amax >= RECIPROCAL_FINITE_AMAX ? v0 * (127.0f / amax) : (v0 / amax) * 127.0f;
        int qv = (int) TornadoMath.floor(TornadoMath.abs(v) + 0.5f);
        qv = qv > 127 ? 127 : qv;
        qv = v < 0.0f ? -qv : qv;
        qv = d > 0.0f ? qv : 0;
        out.set(index, new HalfFloat((float) qv * d));
    }

    /** {@link #fakeQuantizeNormedToFP16} without the active-row mask. */
    public static void fakeQuantizeToFP16(KernelContext ctx, FloatArray in, HalfFloatArray out) {
        int index = ctx.globalIdx;
        float v0 = in.get(index);
        float amax = TornadoMath.abs(v0);
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 16));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 8));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 4));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 2));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 1));
        amax = ctx.simdBroadcastFirst(amax);
        float d = amax / 127.0f;
        float v = amax >= RECIPROCAL_FINITE_AMAX ? v0 * (127.0f / amax) : (v0 / amax) * 127.0f;
        int qv = (int) TornadoMath.floor(TornadoMath.abs(v) + 0.5f);
        qv = qv > 127 ? 127 : qv;
        qv = v < 0.0f ? -qv : qv;
        qv = d > 0.0f ? qv : 0;
        out.set(index, new HalfFloat((float) qv * d));
    }

    // @formatter:off
    /**
     * Q4_0 weights decoded once into the int8 B-operand word layout the GEMM copies.
     *
     * <p>Word {@code (colBlock, kBlock, sub, kRow, pair)} — flat index {@code ((colBlock * (k / 32)
     * + kBlock) * 1024 + sub * 64 + kRow * 4 + pair)} — holds, as bytes low to high, weights {@code
     * (col, kk)}, {@code (col, kk + 1)}, {@code (col + 1, kk)}, {@code (col + 1, kk + 1)} with
     * {@code col = 128 colBlock + 8 sub + 2 pair} and {@code kk = 32 kBlock + 2 kRow}, each as the
     * signed byte {@code q - 8}. The two nibbles of a column come from one 16-bit read of its
     * block. Scales: {@code dW[col][kBlock]} as FP32, written by the lanes with {@code kRow == 0 &&
     * pair == 0} for their eight columns.
     *
     * <p>Worker: {@code n * k / 4} lanes ({@code n % 128 == 0}, {@code k % 32 == 0}).
     */
    // @formatter:on
    public static void decodeQ4_0ToInt8Tiled(
            KernelContext ctx, ByteArray w, ByteArray w8, FloatArray dW, int n, int k) {
        int word = ctx.globalIdx;
        int kBlocks = k / Q8_BLOCK;
        int idx = word & (TILE_WORDS - 1);
        int tile = word >> 10;
        int colBlock = tile / kBlocks;
        int kBlock = tile - colBlock * kBlocks;
        int sub = idx >> 6;
        int kRow = (idx >> 2) & 15;
        int pair = idx & 3;
        int col = (colBlock << 7) + (sub << 3) + (pair << 1);
        int kk = kRow << 1;
        int base0 = (col * kBlocks + kBlock) * BLOCK_BYTES;
        int base1 = base0 + kBlocks * BLOCK_BYTES;
        int byteIndex = kk & 15;
        int two0 = w.getHalfFloat(base0 + 2 + byteIndex).getHalfFloatValue() & 0xFFFF;
        int two1 = w.getHalfFloat(base1 + 2 + byteIndex).getHalfFloatValue() & 0xFFFF;
        int q00;
        int q01;
        int q10;
        int q11;
        if (kk >= 16) {
            q00 = (two0 & 0xF0) >>> 4;
            q01 = (two0 & 0xF000) >>> 12;
            q10 = (two1 & 0xF0) >>> 4;
            q11 = (two1 & 0xF000) >>> 12;
        } else {
            q00 = two0 & 0xF;
            q01 = (two0 >>> 8) & 0xF;
            q10 = two1 & 0xF;
            q11 = (two1 >>> 8) & 0xF;
        }
        w8.set(word << 2, (byte) (q00 - 8));
        w8.set((word << 2) + 1, (byte) (q01 - 8));
        w8.set((word << 2) + 2, (byte) (q10 - 8));
        w8.set((word << 2) + 3, (byte) (q11 - 8));
        if (kRow == 0 && pair == 0) {
            for (int c = 0; c < 8; c++) {
                int cc = (colBlock << 7) + (sub << 3) + c;
                dW.set(
                        cc * kBlocks + kBlock,
                        w.getHalfFloat((cc * kBlocks + kBlock) * BLOCK_BYTES).getFloat32());
            }
        }
    }

    private static final int Q8_0_BLOCK_BYTES = 34;

    // @formatter:off
    /**
     * {@code out[M][N] = sum over 32-blocks b of dA[m][b] * dW[n][b] * (A8[m][b] . W8[n][b])}.
     *
     * <p>Tile 128 x 128 outputs, eight warps as 4 x 2, each warp 32 rows x 64 columns: two 16-row
     * tiles by eight 8-column tiles, 64 FP32 accumulators a lane. A round is 64 k (two blocks): the
     * activation tile (2 x [128 rows][32 bytes], row-major, the quantized bytes as stored) and the
     * weight tile (2 x 1024 words in the decoded layout) are copied global-to-shared with {@code
     * cp.async} into the buffer the current round is not reading, and the round's scales ({@code
     * dA} for 128 rows x 2 blocks, {@code dW} for 128 columns x 2 blocks) are staged next to them
     * by plain loads; the copies are waited for at the end of the round, before the barrier that
     * publishes them. Per block a warp loads its two A fragments once, then for each column tile
     * one B fragment, two MMAs, and eight {@code acc += (float) e * dA * dW} from shared scales.
     * Accumulator element {@code i} of lane {@code l}: row {@code (l >> 2) + 8 (i >> 1)}, column
     * {@code 2 (l & 3) + (i & 1)}.
     *
     * <p>Requires M % 128 == 0, N % 128 == 0, K % 64 == 0. Worker: WorkerGrid2D((M/128)*256,
     * N/128), local 256.
     */
    // @formatter:on
    public static void gemmInt8BlockScaled(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray w8,
            FloatArray dW,
            FloatArray out,
            int m,
            int n,
            int k) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / I8_WARPS_N;
        int warpN = warpId - warpM * I8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int blockCol = I8_BN * ctx.groupIdy;
        int kBlocks = k / Q8_BLOCK;
        int rounds = k / I8_BK;

        int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
        float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

        float[] acc = new float[64];
        for (int i = 0; i < 64; i++) {
            acc[i] = 0.0f;
        }
        int rowInWarp = lane >> 2;
        int colInWarp = (lane & 3) << 1;
        int r0 = blockRow + warpM * I8_WM + rowInWarp;
        int c0 = blockCol + warpN * I8_WN + colInWarp;
        // The weight tiles of this column block: tile t = kBlock t of column block groupIdy.
        int wTileBase = ctx.groupIdy * kBlocks * TILE_WORDS;

        stageRound(
                ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, 0, 0, blockRow, blockCol, k, kBlocks,
                wTileBase, tid);
        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        for (int round = 0; round < rounds; round++) {
            int buf = round & 1;
            int bufNext = 1 - buf;
            boolean hasNext = round + 1 < rounds;
            if (hasNext) {
                stageRound(
                        ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, round + 1, bufNext, blockRow,
                        blockCol, k, kBlocks, wTileBase, tid);
                ctx.asyncCopyCommit();
            }
            int aBuf = buf * 2 * TILE_WORDS;
            int sBuf = buf * 2 * I8_BM;
            for (int b = 0; b < 2; b++) {
                int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                float dA0 = sA[sRow];
                float dA1 = sA[sRow + 8];
                float dA2 = sA[sRow + 16];
                float dA3 = sA[sRow + 24];
                int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 2048;
                int sCol = sBuf + b * I8_BN + warpN * I8_WN + colInWarp;
                for (int j = 0; j < 8; j++) {
                    byte[] fb = ctx.mmaLoadBInt8(bTile, 32, bOff + j * 256);
                    int[] d0 = ctx.mmaInt8(a0, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    int[] d1 = ctx.mmaInt8(a1, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    float dW0 = sW[sCol + (j << 3)];
                    float dW1 = sW[sCol + (j << 3) + 1];
                    acc[j * 8 + 0] += (float) d0[0] * dA0 * dW0;
                    acc[j * 8 + 1] += (float) d0[1] * dA0 * dW1;
                    acc[j * 8 + 2] += (float) d0[2] * dA1 * dW0;
                    acc[j * 8 + 3] += (float) d0[3] * dA1 * dW1;
                    acc[j * 8 + 4] += (float) d1[0] * dA2 * dW0;
                    acc[j * 8 + 5] += (float) d1[1] * dA2 * dW1;
                    acc[j * 8 + 6] += (float) d1[2] * dA3 * dW0;
                    acc[j * 8 + 7] += (float) d1[3] * dA3 * dW1;
                }
            }
            if (hasNext) {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
        }

        for (int j = 0; j < 8; j++) {
            int col = c0 + (j << 3);
            out.set(r0 * n + col, acc[j * 8 + 0]);
            out.set(r0 * n + col + 1, acc[j * 8 + 1]);
            out.set((r0 + 8) * n + col, acc[j * 8 + 2]);
            out.set((r0 + 8) * n + col + 1, acc[j * 8 + 3]);
            out.set((r0 + 16) * n + col, acc[j * 8 + 4]);
            out.set((r0 + 16) * n + col + 1, acc[j * 8 + 5]);
            out.set((r0 + 24) * n + col, acc[j * 8 + 6]);
            out.set((r0 + 24) * n + col + 1, acc[j * 8 + 7]);
        }
    }

    /** {@link #gemmInt8BlockScaled} adding the product into {@code out}: {@code out += A x W}. */
    public static void gemmInt8BlockScaledResidual(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray w8,
            FloatArray dW,
            FloatArray out,
            int m,
            int n,
            int k) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / I8_WARPS_N;
        int warpN = warpId - warpM * I8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int blockCol = I8_BN * ctx.groupIdy;
        int kBlocks = k / Q8_BLOCK;
        int rounds = k / I8_BK;

        int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
        float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

        float[] acc = new float[64];
        for (int i = 0; i < 64; i++) {
            acc[i] = 0.0f;
        }
        int rowInWarp = lane >> 2;
        int colInWarp = (lane & 3) << 1;
        int r0 = blockRow + warpM * I8_WM + rowInWarp;
        int c0 = blockCol + warpN * I8_WN + colInWarp;
        // The weight tiles of this column block: tile t = kBlock t of column block groupIdy.
        int wTileBase = ctx.groupIdy * kBlocks * TILE_WORDS;

        stageRound(
                ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, 0, 0, blockRow, blockCol, k, kBlocks,
                wTileBase, tid);
        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        for (int round = 0; round < rounds; round++) {
            int buf = round & 1;
            int bufNext = 1 - buf;
            boolean hasNext = round + 1 < rounds;
            if (hasNext) {
                stageRound(
                        ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, round + 1, bufNext, blockRow,
                        blockCol, k, kBlocks, wTileBase, tid);
                ctx.asyncCopyCommit();
            }
            int aBuf = buf * 2 * TILE_WORDS;
            int sBuf = buf * 2 * I8_BM;
            for (int b = 0; b < 2; b++) {
                int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                float dA0 = sA[sRow];
                float dA1 = sA[sRow + 8];
                float dA2 = sA[sRow + 16];
                float dA3 = sA[sRow + 24];
                int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 2048;
                int sCol = sBuf + b * I8_BN + warpN * I8_WN + colInWarp;
                for (int j = 0; j < 8; j++) {
                    byte[] fb = ctx.mmaLoadBInt8(bTile, 32, bOff + j * 256);
                    int[] d0 = ctx.mmaInt8(a0, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    int[] d1 = ctx.mmaInt8(a1, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    float dW0 = sW[sCol + (j << 3)];
                    float dW1 = sW[sCol + (j << 3) + 1];
                    acc[j * 8 + 0] += (float) d0[0] * dA0 * dW0;
                    acc[j * 8 + 1] += (float) d0[1] * dA0 * dW1;
                    acc[j * 8 + 2] += (float) d0[2] * dA1 * dW0;
                    acc[j * 8 + 3] += (float) d0[3] * dA1 * dW1;
                    acc[j * 8 + 4] += (float) d1[0] * dA2 * dW0;
                    acc[j * 8 + 5] += (float) d1[1] * dA2 * dW1;
                    acc[j * 8 + 6] += (float) d1[2] * dA3 * dW0;
                    acc[j * 8 + 7] += (float) d1[3] * dA3 * dW1;
                }
            }
            if (hasNext) {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
        }

        for (int j = 0; j < 8; j++) {
            int col = c0 + (j << 3);
            out.set(r0 * n + col, out.get(r0 * n + col) + acc[j * 8 + 0]);
            out.set(r0 * n + col + 1, out.get(r0 * n + col + 1) + acc[j * 8 + 1]);
            out.set((r0 + 8) * n + col, out.get((r0 + 8) * n + col) + acc[j * 8 + 2]);
            out.set((r0 + 8) * n + col + 1, out.get((r0 + 8) * n + col + 1) + acc[j * 8 + 3]);
            out.set((r0 + 16) * n + col, out.get((r0 + 16) * n + col) + acc[j * 8 + 4]);
            out.set((r0 + 16) * n + col + 1, out.get((r0 + 16) * n + col + 1) + acc[j * 8 + 5]);
            out.set((r0 + 24) * n + col, out.get((r0 + 24) * n + col) + acc[j * 8 + 6]);
            out.set((r0 + 24) * n + col + 1, out.get((r0 + 24) * n + col + 1) + acc[j * 8 + 7]);
        }
    }

    /**
     * {@link #gemmInt8BlockScaled} writing {@code hb = silu(gate) * (A x W)} in FP32, {@code gate}
     * the FP32 gate projection of the same shape; FP32 because the int8 down projection quantizes
     * it next, which needs the whole 32-block.
     */
    public static void gemmInt8BlockScaledSwiGLU(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray w8,
            FloatArray dW,
            FloatArray gate,
            FloatArray hb,
            int m,
            int n,
            int k) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / I8_WARPS_N;
        int warpN = warpId - warpM * I8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int blockCol = I8_BN * ctx.groupIdy;
        int kBlocks = k / Q8_BLOCK;
        int rounds = k / I8_BK;

        int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
        float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
        float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

        float[] acc = new float[64];
        for (int i = 0; i < 64; i++) {
            acc[i] = 0.0f;
        }
        int rowInWarp = lane >> 2;
        int colInWarp = (lane & 3) << 1;
        int r0 = blockRow + warpM * I8_WM + rowInWarp;
        int c0 = blockCol + warpN * I8_WN + colInWarp;
        // The weight tiles of this column block: tile t = kBlock t of column block groupIdy.
        int wTileBase = ctx.groupIdy * kBlocks * TILE_WORDS;

        stageRound(
                ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, 0, 0, blockRow, blockCol, k, kBlocks,
                wTileBase, tid);
        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        for (int round = 0; round < rounds; round++) {
            int buf = round & 1;
            int bufNext = 1 - buf;
            boolean hasNext = round + 1 < rounds;
            if (hasNext) {
                stageRound(
                        ctx, aTile, bTile, sA, sW, a8, dA, w8, dW, round + 1, bufNext, blockRow,
                        blockCol, k, kBlocks, wTileBase, tid);
                ctx.asyncCopyCommit();
            }
            int aBuf = buf * 2 * TILE_WORDS;
            int sBuf = buf * 2 * I8_BM;
            for (int b = 0; b < 2; b++) {
                int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                float dA0 = sA[sRow];
                float dA1 = sA[sRow + 8];
                float dA2 = sA[sRow + 16];
                float dA3 = sA[sRow + 24];
                int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 2048;
                int sCol = sBuf + b * I8_BN + warpN * I8_WN + colInWarp;
                for (int j = 0; j < 8; j++) {
                    byte[] fb = ctx.mmaLoadBInt8(bTile, 32, bOff + j * 256);
                    int[] d0 = ctx.mmaInt8(a0, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    int[] d1 = ctx.mmaInt8(a1, fb, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                    float dW0 = sW[sCol + (j << 3)];
                    float dW1 = sW[sCol + (j << 3) + 1];
                    acc[j * 8 + 0] += (float) d0[0] * dA0 * dW0;
                    acc[j * 8 + 1] += (float) d0[1] * dA0 * dW1;
                    acc[j * 8 + 2] += (float) d0[2] * dA1 * dW0;
                    acc[j * 8 + 3] += (float) d0[3] * dA1 * dW1;
                    acc[j * 8 + 4] += (float) d1[0] * dA2 * dW0;
                    acc[j * 8 + 5] += (float) d1[1] * dA2 * dW1;
                    acc[j * 8 + 6] += (float) d1[2] * dA3 * dW0;
                    acc[j * 8 + 7] += (float) d1[3] * dA3 * dW1;
                }
            }
            if (hasNext) {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
        }

        for (int j = 0; j < 8; j++) {
            int col = c0 + (j << 3);
            {
                float g = gate.get(r0 * n + col);
                hb.set(r0 * n + col, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 0]);
            }
            {
                float g = gate.get(r0 * n + col + 1);
                hb.set(r0 * n + col + 1, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 1]);
            }
            {
                float g = gate.get((r0 + 8) * n + col);
                hb.set((r0 + 8) * n + col, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 2]);
            }
            {
                float g = gate.get((r0 + 8) * n + col + 1);
                hb.set((r0 + 8) * n + col + 1, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 3]);
            }
            {
                float g = gate.get((r0 + 16) * n + col);
                hb.set((r0 + 16) * n + col, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 4]);
            }
            {
                float g = gate.get((r0 + 16) * n + col + 1);
                hb.set(
                        (r0 + 16) * n + col + 1,
                        (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 5]);
            }
            {
                float g = gate.get((r0 + 24) * n + col);
                hb.set((r0 + 24) * n + col, (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 6]);
            }
            {
                float g = gate.get((r0 + 24) * n + col + 1);
                hb.set(
                        (r0 + 24) * n + col + 1,
                        (g / (1.0f + TornadoMath.exp(-g))) * acc[j * 8 + 7]);
            }
        }
    }

    /** Threads of a {@link #gemmInt8Q8_0} block: sixteen warps as four by four. */
    public static final int Q8_GEMM_THREADS = 512;

    private static final int Q8_WARPS_N = 4;

    /** Columns of one {@link #gemmInt8Q8_0} warp's tile: four 8-column MMA tiles. */
    private static final int Q8_WN = 32;

    /** {@link #gemmInt8Q8_0}'s epilogues. */
    public static final int EPILOGUE_STORE = 0;

    public static final int EPILOGUE_RESIDUAL = 1;

    public static final int EPILOGUE_SWIGLU = 2;

    // @formatter:off
    /**
     * The block-scaled int8 GEMM of {@link #gemmInt8BlockScaled} reading Q8_0 weights where they
     * lie, with no decoded copy: the same 128 x 128 tile, 64-k rounds, fragments and arithmetic,
     * but sixteen warps as 4 x 4, each 32 rows x 32 columns, rather than eight with 64 columns.
     *
     * <p>Why sixteen. Every pair of MMAs is followed by eight conversions and scalings of their
     * results, which wait on the MMAs; with eight warps a block needed 170 registers a lane, one
     * block filled a multiprocessor, and two warps a scheduler could not cover the MMA latency
     * behind each pair (ptxas reused one set of result registers for every pair, so nothing
     * overlapped). A 32-column warp tile halves the accumulators, and sixteen warps give each
     * scheduler four to switch between.
     *
     * <p>A Q8_0 block's quants start two bytes into a 34-byte block, so they cannot be copied with
     * {@code cp.async}; each round's two weight tiles are read with 16-bit loads instead (lanes of
     * a warp take consecutive {@code kRow}s of two column pairs, four rows of 32 contiguous bytes)
     * and written to shared memory in the decoded layout of {@link #decodeQ4_0ToInt8Tiled}, the
     * block scales read from the same blocks. The next round's words are loaded into registers
     * before the current round's MMAs and stored after them, so the loads are in flight while the
     * tensor cores work. The activation tile is still copied asynchronously. Measured on the 27B's
     * Q8_0 prefill (two A10s), reading the blocks directly beat a separate rearranging pass
     * followed by the GEMM by 23%: the pass was bound by memory bandwidth and this staging is not.
     *
     * <p>{@code splits > 1} splits K: block row {@code groupIdy} is column tile {@code groupIdy /
     * splits} and split {@code groupIdy % splits}, which runs its share of the rounds and writes
     * its partial sums to {@code partial[split]} for {@link #reduceSplitsQ8_0}; {@code partial} is
     * not read with one split.
     *
     * <p>{@code epilogue}: {@link #EPILOGUE_STORE} writes {@code out}, {@link #EPILOGUE_RESIDUAL}
     * adds into it, {@link #EPILOGUE_SWIGLU} writes {@code out = silu(gate) * product} ({@code
     * gate} is not read otherwise). Only the first {@code rowsHolder[1]} rows are computed: a
     * chunk's padding rows are left as they were. Row {@code r} of the result is written at {@code
     * out[r * ldo + colOffset]}, so several GEMMs can fill one packed row; {@code ldo} is {@code n}
     * and {@code colOffset} zero for a plain one, and a split GEMM must be plain. Requires M % 128
     * == 0, N % 128 == 0, K % 64 == 0. Worker: WorkerGrid2D((M/128) * 512, N/128 * splits), local
     * {@link #Q8_GEMM_THREADS}.
     */
    // @formatter:on
    public static void gemmInt8Q8_0(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray w,
            FloatArray out,
            FloatArray gate,
            int m,
            int n,
            int k,
            int epilogue,
            FloatArray partial,
            int splits,
            IntArray rowsHolder,
            int ldo,
            int colOffset) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / Q8_WARPS_N;
        int warpN = warpId - warpM * Q8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int colTile = ctx.groupIdy / splits;
        int split = ctx.groupIdy - colTile * splits;
        int blockCol = I8_BN * colTile;
        // A chunk's real rows are rowsHolder[1]; a block wholly past them has nothing to do.
        if (blockRow < m && blockRow < rowsHolder.get(1) && blockCol < n) {
            int kBlocks = k / Q8_BLOCK;
            int rounds = k / I8_BK;
            // This block's share of the rounds; with one split, all of them.
            int roundsPerSplit = (rounds + splits - 1) / splits;
            int roundBegin = split * roundsPerSplit;
            int roundEnd = Math.min(rounds, roundBegin + roundsPerSplit);

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
            float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

            float[] acc = new float[32];
            for (int i = 0; i < 32; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int r0 = blockRow + warpM * I8_WM + rowInWarp;
            int c0 = blockCol + warpN * Q8_WN + colInWarp;

            int rowBytes = kBlocks * Q8_0_BLOCK_BYTES;
            // This lane's four weight words of a round: lanes of a warp on consecutive kRows of
            // two column pairs, so each load is part of a run of 32 contiguous bytes. Word 0 is
            // column pair colPair of the round's first block, word 1 the pair 32 further on (64
            // columns, 512 words), words 2 and 3 the same in the second block.
            int kRow = tid & 15;
            int colPair = (tid >> 4) & 31;
            int slot = ((colPair >> 2) << 6) + (kRow << 2) + (colPair & 3);
            int laneOffset = (blockCol + (colPair << 1)) * rowBytes + 2 + (kRow << 1);
            int halfStride = 64 * rowBytes;
            if (roundBegin < roundEnd) {
                stageActivationsQ8_0(
                        ctx, aTile, sA, a8, dA, roundBegin, 0, blockRow, k, kBlocks, tid);
                stageWeightsQ8_0(
                        bTile,
                        sW,
                        w,
                        roundBegin,
                        0,
                        blockCol,
                        rowBytes,
                        laneOffset,
                        halfStride,
                        slot,
                        tid);
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(0);
                ctx.localBarrier();
            }

            for (int round = roundBegin; round < roundEnd; round++) {
                int buf = (round - roundBegin) & 1;
                int bufNext = 1 - buf;
                boolean hasNext = round + 1 < roundEnd;
                // The next round's weight words, loaded now and stored after this round's MMAs,
                // so the loads are in flight while the tensor cores work.
                int next = (round + 1) * 2 * Q8_0_BLOCK_BYTES + laneOffset;
                int b1 = Q8_0_BLOCK_BYTES;
                int l0 = 0, h0 = 0, l1 = 0, h1 = 0, l2 = 0, h2 = 0, l3 = 0, h3 = 0;
                float scaleNext = 0.0f;
                if (hasNext) {
                    stageActivationsQ8_0(
                            ctx, aTile, sA, a8, dA, round + 1, bufNext, blockRow, k, kBlocks, tid);
                    ctx.asyncCopyCommit();
                    l0 = halfBits(w, next);
                    h0 = halfBits(w, next + rowBytes);
                    l1 = halfBits(w, next + halfStride);
                    h1 = halfBits(w, next + halfStride + rowBytes);
                    l2 = halfBits(w, next + b1);
                    h2 = halfBits(w, next + b1 + rowBytes);
                    l3 = halfBits(w, next + b1 + halfStride);
                    h3 = halfBits(w, next + b1 + halfStride + rowBytes);
                    if (tid >= 256) {
                        scaleNext =
                                w.getHalfFloat(
                                                (blockCol + (tid & 127)) * rowBytes
                                                        + ((round + 1) * 2 + ((tid >> 7) & 1))
                                                                * Q8_0_BLOCK_BYTES)
                                        .getFloat32();
                    }
                }
                int aBuf = buf * 2 * TILE_WORDS;
                int sBuf = buf * 2 * I8_BM;
                for (int b = 0; b < 2; b++) {
                    int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                    byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                    byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                    int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                    float dA0 = sA[sRow];
                    float dA1 = sA[sRow + 8];
                    float dA2 = sA[sRow + 16];
                    float dA3 = sA[sRow + 24];
                    int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 1024;
                    int sCol = sBuf + b * I8_BN + warpN * Q8_WN + colInWarp;
                    // Two column steps at a time: their four MMAs issue before either's results
                    // are scaled, so the scaling of one waits behind the other's MMAs rather than
                    // stalling on its own.
                    for (int jj = 0; jj < 4; jj += 2) {
                        byte[] fb0 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256);
                        byte[] fb1 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256 + 256);
                        int[] d0 = ctx.mmaInt8(a0, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] d1 = ctx.mmaInt8(a1, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e0 = ctx.mmaInt8(a0, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e1 = ctx.mmaInt8(a1, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        float dW0 = sW[sCol + (jj << 3)];
                        float dW1 = sW[sCol + (jj << 3) + 1];
                        float eW0 = sW[sCol + (jj << 3) + 8];
                        float eW1 = sW[sCol + (jj << 3) + 9];
                        acc[jj * 8 + 0] += (float) d0[0] * dA0 * dW0;
                        acc[jj * 8 + 1] += (float) d0[1] * dA0 * dW1;
                        acc[jj * 8 + 2] += (float) d0[2] * dA1 * dW0;
                        acc[jj * 8 + 3] += (float) d0[3] * dA1 * dW1;
                        acc[jj * 8 + 4] += (float) d1[0] * dA2 * dW0;
                        acc[jj * 8 + 5] += (float) d1[1] * dA2 * dW1;
                        acc[jj * 8 + 6] += (float) d1[2] * dA3 * dW0;
                        acc[jj * 8 + 7] += (float) d1[3] * dA3 * dW1;
                        acc[jj * 8 + 8] += (float) e0[0] * dA0 * eW0;
                        acc[jj * 8 + 9] += (float) e0[1] * dA0 * eW1;
                        acc[jj * 8 + 10] += (float) e0[2] * dA1 * eW0;
                        acc[jj * 8 + 11] += (float) e0[3] * dA1 * eW1;
                        acc[jj * 8 + 12] += (float) e1[0] * dA2 * eW0;
                        acc[jj * 8 + 13] += (float) e1[1] * dA2 * eW1;
                        acc[jj * 8 + 14] += (float) e1[2] * dA3 * eW0;
                        acc[jj * 8 + 15] += (float) e1[3] * dA3 * eW1;
                    }
                }
                if (hasNext) {
                    int dst = bufNext * 2 * TILE_WORDS + slot;
                    bTile[dst] = l0 | (h0 << 16);
                    bTile[dst + 512] = l1 | (h1 << 16);
                    bTile[dst + TILE_WORDS] = l2 | (h2 << 16);
                    bTile[dst + TILE_WORDS + 512] = l3 | (h3 << 16);
                    if (tid >= 256) {
                        sW[bufNext * 2 * I8_BN + tid - 256] = scaleNext;
                    }
                    ctx.asyncCopyWaitGroup(0);
                }
                ctx.localBarrier();
            }

            int splitBase = split * m * n;
            for (int j = 0; j < 4; j++) {
                int col = c0 + (j << 3);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 0],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 1],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 2],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 3],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 4],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 5],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 6],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 7],
                        epilogue,
                        splits);
            }
        }
    }

    /**
     * Bytes of one packed weight tile of {@link #packQ8_0Tiles}: 8 KB of quants and 512 bytes of
     * FP16 scales, the size of the 256 Q8_0 blocks it holds.
     */
    public static final int PACKED_TILE_BYTES = 8704;

    /** Bytes of a packed tile's quants, ahead of its scales. */
    public static final int PACKED_QUANT_BYTES = 8192;

    // @formatter:off
    /**
     * Repacks Q8_0 weights ({@code n} rows of {@code k}) for {@link #gemmInt8Q8_0Packed}: one
     * 9216-byte tile per 128 columns and 64-k round, column tile major, so that a round's weights
     * are one contiguous, 16-byte-aligned run that {@code cp.async} can copy. A tile holds the
     * round's 2048 quant words in the shared-memory order {@link #stageWeightsQ8_0} writes (two
     * 32-blocks of 1024 words; per block two 64-column halves of 512 words, word {@code ((colPair
     * >> 2) << 6) + (kRow << 2) + (colPair & 3)} holding quants {@code k, k + 1} of columns {@code
     * c, c + 1}), then the 256 block scales as FP16, block-major (scale {@code j} is column {@code
     * j & 127} of the round's block {@code j >> 7}). The same bytes as the Q8_0 blocks, reordered.
     * Runs on the host, once per weight.
     *
     * <p>Requires {@code n % 128 == 0} and {@code k % 64 == 0}.
     */
    // @formatter:on
    public static ByteArray packQ8_0Tiles(ByteArray w, int n, int k) {
        return ByteArray.fromArray(packQ8_0TileBytes(w, n, k));
    }

    /** The bytes of {@link #packQ8_0Tiles}, on the heap. */
    public static byte[] packQ8_0TileBytes(ByteArray w, int n, int k) {
        return PackedTilePacker.packQ8_0(w, n, k);
    }

    // @formatter:off
    /**
     * {@link #gemmInt8Q8_0} over weights packed by {@link #packQ8_0Tiles}: a round's activation
     * tile and weight tile arrive by 16-byte {@code cp.async} copies into two slots,
     * double-buffered (one weight copy per lane), so no lane holds prefetched weights in registers
     * and no weight quant goes through 16-bit global loads. The round's 256 weight scales are read
     * when the round is issued, a contiguous 512 bytes. The next round's copies are in flight while
     * this round's MMAs run. Same fragments, MMAs, scaling and summation order as {@link
     * #gemmInt8Q8_0}, so the results are equal bit for bit.
     *
     * <p>36 KB of shared memory ({@link #PACKED_RING} slots), static. Same parameters, requirements
     * and worker as {@link #gemmInt8Q8_0}, with {@code packed} in place of the Q8_0 weights.
     */
    // @formatter:on
    public static void gemmInt8Q8_0Packed(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray packed,
            FloatArray out,
            FloatArray gate,
            int m,
            int n,
            int k,
            int epilogue,
            FloatArray partial,
            int splits,
            IntArray rowsHolder,
            int ldo,
            int colOffset) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / Q8_WARPS_N;
        int warpN = warpId - warpM * Q8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int colTile = ctx.groupIdy / splits;
        int split = ctx.groupIdy - colTile * splits;
        int blockCol = I8_BN * colTile;
        // A chunk's real rows are rowsHolder[1]; a block wholly past them has nothing to do.
        if (blockRow < m && blockRow < rowsHolder.get(1) && blockCol < n) {
            int kBlocks = k / Q8_BLOCK;
            int rounds = k / I8_BK;
            int roundsPerSplit = (rounds + splits - 1) / splits;
            int roundBegin = split * roundsPerSplit;
            int roundEnd = Math.min(rounds, roundBegin + roundsPerSplit);
            int count = roundEnd - roundBegin;
            int tileBase = colTile * rounds * PACKED_TILE_BYTES;

            int[] aTile = ctx.allocateIntLocalArray(PACKED_RING * 2 * TILE_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(PACKED_RING * 2 * TILE_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(PACKED_RING * 2 * I8_BM);
            float[] sW = ctx.allocateFloatLocalArray(PACKED_RING * 2 * I8_BN);

            float[] acc = new float[32];
            for (int i = 0; i < 32; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int r0 = blockRow + warpM * I8_WM + rowInWarp;
            int c0 = blockCol + warpN * Q8_WN + colInWarp;

            // Prologue: round 0 into slot 0.
            if (count > 0) {
                int tile = tileBase + roundBegin * PACKED_TILE_BYTES;
                stageActivationsQ8_0(
                        ctx, aTile, sA, a8, dA, roundBegin, 0, blockRow, k, kBlocks, tid);
                ctx.asyncCopyToLocal16(bTile, tid << 2, packed, tile + (tid << 4));
                if (tid < 256) {
                    sW[tid] =
                            packed.getHalfFloat(tile + PACKED_QUANT_BYTES + (tid << 1))
                                    .getFloat32();
                }
            }
            ctx.asyncCopyCommit();

            int cur = 0;
            for (int i = 0; i < count; i++) {
                // Round i's copies have landed, and every warp is past round i - 1.
                ctx.asyncCopyWaitGroup(0);
                ctx.localBarrier();

                // Round i + 1 goes into the other slot, which round i - 1 used.
                int ahead = i + 1;
                int aheadSlot = 1 - cur;
                if (ahead < count) {
                    int round = roundBegin + ahead;
                    stageActivationsQ8_0(
                            ctx, aTile, sA, a8, dA, round, aheadSlot, blockRow, k, kBlocks, tid);
                    int tile = tileBase + round * PACKED_TILE_BYTES;
                    ctx.asyncCopyToLocal16(
                            bTile,
                            aheadSlot * 2 * TILE_WORDS + (tid << 2),
                            packed,
                            tile + (tid << 4));
                    if (tid < 256) {
                        sW[aheadSlot * 2 * I8_BN + tid] =
                                packed.getHalfFloat(tile + PACKED_QUANT_BYTES + (tid << 1))
                                        .getFloat32();
                    }
                }
                ctx.asyncCopyCommit();

                int aBuf = cur * 2 * TILE_WORDS;
                int sBuf = cur * 2 * I8_BM;
                for (int b = 0; b < 2; b++) {
                    int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                    byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                    byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                    int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                    float dA0 = sA[sRow];
                    float dA1 = sA[sRow + 8];
                    float dA2 = sA[sRow + 16];
                    float dA3 = sA[sRow + 24];
                    int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 1024;
                    int sCol = sBuf + b * I8_BN + warpN * Q8_WN + colInWarp;
                    for (int jj = 0; jj < 4; jj += 2) {
                        byte[] fb0 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256);
                        byte[] fb1 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256 + 256);
                        int[] d0 = ctx.mmaInt8(a0, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] d1 = ctx.mmaInt8(a1, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e0 = ctx.mmaInt8(a0, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e1 = ctx.mmaInt8(a1, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        float dW0 = sW[sCol + (jj << 3)];
                        float dW1 = sW[sCol + (jj << 3) + 1];
                        float eW0 = sW[sCol + (jj << 3) + 8];
                        float eW1 = sW[sCol + (jj << 3) + 9];
                        acc[jj * 8 + 0] += (float) d0[0] * dA0 * dW0;
                        acc[jj * 8 + 1] += (float) d0[1] * dA0 * dW1;
                        acc[jj * 8 + 2] += (float) d0[2] * dA1 * dW0;
                        acc[jj * 8 + 3] += (float) d0[3] * dA1 * dW1;
                        acc[jj * 8 + 4] += (float) d1[0] * dA2 * dW0;
                        acc[jj * 8 + 5] += (float) d1[1] * dA2 * dW1;
                        acc[jj * 8 + 6] += (float) d1[2] * dA3 * dW0;
                        acc[jj * 8 + 7] += (float) d1[3] * dA3 * dW1;
                        acc[jj * 8 + 8] += (float) e0[0] * dA0 * eW0;
                        acc[jj * 8 + 9] += (float) e0[1] * dA0 * eW1;
                        acc[jj * 8 + 10] += (float) e0[2] * dA1 * eW0;
                        acc[jj * 8 + 11] += (float) e0[3] * dA1 * eW1;
                        acc[jj * 8 + 12] += (float) e1[0] * dA2 * eW0;
                        acc[jj * 8 + 13] += (float) e1[1] * dA2 * eW1;
                        acc[jj * 8 + 14] += (float) e1[2] * dA3 * eW0;
                        acc[jj * 8 + 15] += (float) e1[3] * dA3 * eW1;
                    }
                }
                cur = 1 - cur;
            }

            int splitBase = split * m * n;
            for (int j = 0; j < 4; j++) {
                int col = c0 + (j << 3);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 0],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 1],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 8,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 2],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 8,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 3],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 16,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 4],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 16,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 5],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 24,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 6],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0 + 24,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 7],
                        epilogue,
                        splits);
            }
        }
    }

    /**
     * Slots of {@link #gemmInt8Q8_0Packed}'s ring: two, 18 KB each, so that the 36 KB stay within
     * the 48 KB of static shared memory a block can declare.
     */
    public static final int PACKED_RING = 2;

    // @formatter:off
    /**
     * {@link #gemmInt8Q8_0} over Q4_0 weights where they lie: the same tiles, rounds, fragments and
     * epilogues, each 16-bit pair of staged quants decoded from a pair of packed bytes — the low
     * nibbles for the first sixteen quants of a block, the high for the rest — and recentred to
     * signed bytes, so the products are exact. Same requirements and worker.
     */
    // @formatter:on
    public static void gemmInt8Q4_0(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray w,
            FloatArray out,
            FloatArray gate,
            int m,
            int n,
            int k,
            int epilogue,
            FloatArray partial,
            int splits,
            IntArray rowsHolder,
            int ldo,
            int colOffset) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / Q8_WARPS_N;
        int warpN = warpId - warpM * Q8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int colTile = ctx.groupIdy / splits;
        int split = ctx.groupIdy - colTile * splits;
        int blockCol = I8_BN * colTile;
        // A chunk's real rows are rowsHolder[1]; a block wholly past them has nothing to do.
        if (blockRow < m && blockRow < rowsHolder.get(1) && blockCol < n) {
            int kBlocks = k / Q8_BLOCK;
            int rounds = k / I8_BK;
            // This block's share of the rounds; with one split, all of them.
            int roundsPerSplit = (rounds + splits - 1) / splits;
            int roundBegin = split * roundsPerSplit;
            int roundEnd = Math.min(rounds, roundBegin + roundsPerSplit);

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
            float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

            float[] acc = new float[32];
            for (int i = 0; i < 32; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int r0 = blockRow + warpM * I8_WM + rowInWarp;
            int c0 = blockCol + warpN * Q8_WN + colInWarp;

            int rowBytes = kBlocks * Q4_0_BLOCK_BYTES;
            // This lane's four weight words of a round: lanes of a warp on consecutive kRows of
            // two column pairs, so each load is part of a run of 32 contiguous bytes. Word 0 is
            // column pair colPair of the round's first block, word 1 the pair 32 further on (64
            // columns, 512 words), words 2 and 3 the same in the second block.
            int kRow = tid & 15;
            int colPair = (tid >> 4) & 31;
            int slot = ((colPair >> 2) << 6) + (kRow << 2) + (colPair & 3);
            // A Q4_0 block's quants k and k + 1 sit in the low nibbles of bytes k, k + 1 for k < 16
            // and in the high nibbles of bytes k - 16, k - 15 above.
            int nibbleShift = (kRow >> 3) << 2;
            int laneOffset = (blockCol + (colPair << 1)) * rowBytes + 2 + ((kRow & 7) << 1);
            int halfStride = 64 * rowBytes;
            if (roundBegin < roundEnd) {
                stageActivationsQ8_0(
                        ctx, aTile, sA, a8, dA, roundBegin, 0, blockRow, k, kBlocks, tid);
                stageWeightsQ4_0(
                        bTile,
                        sW,
                        w,
                        roundBegin,
                        0,
                        blockCol,
                        rowBytes,
                        laneOffset,
                        halfStride,
                        slot,
                        tid,
                        nibbleShift);
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(0);
                ctx.localBarrier();
            }

            for (int round = roundBegin; round < roundEnd; round++) {
                int buf = (round - roundBegin) & 1;
                int bufNext = 1 - buf;
                boolean hasNext = round + 1 < roundEnd;
                // The next round's weight words, loaded now and stored after this round's MMAs,
                // so the loads are in flight while the tensor cores work.
                int next = (round + 1) * 2 * Q4_0_BLOCK_BYTES + laneOffset;
                int b1 = Q4_0_BLOCK_BYTES;
                int l0 = 0, h0 = 0, l1 = 0, h1 = 0, l2 = 0, h2 = 0, l3 = 0, h3 = 0;
                float scaleNext = 0.0f;
                if (hasNext) {
                    stageActivationsQ8_0(
                            ctx, aTile, sA, a8, dA, round + 1, bufNext, blockRow, k, kBlocks, tid);
                    ctx.asyncCopyCommit();
                    l0 = q4Pair(w, next, nibbleShift);
                    h0 = q4Pair(w, next + rowBytes, nibbleShift);
                    l1 = q4Pair(w, next + halfStride, nibbleShift);
                    h1 = q4Pair(w, next + halfStride + rowBytes, nibbleShift);
                    l2 = q4Pair(w, next + b1, nibbleShift);
                    h2 = q4Pair(w, next + b1 + rowBytes, nibbleShift);
                    l3 = q4Pair(w, next + b1 + halfStride, nibbleShift);
                    h3 = q4Pair(w, next + b1 + halfStride + rowBytes, nibbleShift);
                    if (tid >= 256) {
                        scaleNext =
                                w.getHalfFloat(
                                                (blockCol + (tid & 127)) * rowBytes
                                                        + ((round + 1) * 2 + ((tid >> 7) & 1))
                                                                * Q4_0_BLOCK_BYTES)
                                        .getFloat32();
                    }
                }
                int aBuf = buf * 2 * TILE_WORDS;
                int sBuf = buf * 2 * I8_BM;
                for (int b = 0; b < 2; b++) {
                    int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                    byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                    byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                    int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                    float dA0 = sA[sRow];
                    float dA1 = sA[sRow + 8];
                    float dA2 = sA[sRow + 16];
                    float dA3 = sA[sRow + 24];
                    int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 1024;
                    int sCol = sBuf + b * I8_BN + warpN * Q8_WN + colInWarp;
                    // Two column steps at a time: their four MMAs issue before either's results
                    // are scaled, so the scaling of one waits behind the other's MMAs rather than
                    // stalling on its own.
                    for (int jj = 0; jj < 4; jj += 2) {
                        byte[] fb0 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256);
                        byte[] fb1 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256 + 256);
                        int[] d0 = ctx.mmaInt8(a0, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] d1 = ctx.mmaInt8(a1, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e0 = ctx.mmaInt8(a0, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e1 = ctx.mmaInt8(a1, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        float dW0 = sW[sCol + (jj << 3)];
                        float dW1 = sW[sCol + (jj << 3) + 1];
                        float eW0 = sW[sCol + (jj << 3) + 8];
                        float eW1 = sW[sCol + (jj << 3) + 9];
                        acc[jj * 8 + 0] += (float) d0[0] * dA0 * dW0;
                        acc[jj * 8 + 1] += (float) d0[1] * dA0 * dW1;
                        acc[jj * 8 + 2] += (float) d0[2] * dA1 * dW0;
                        acc[jj * 8 + 3] += (float) d0[3] * dA1 * dW1;
                        acc[jj * 8 + 4] += (float) d1[0] * dA2 * dW0;
                        acc[jj * 8 + 5] += (float) d1[1] * dA2 * dW1;
                        acc[jj * 8 + 6] += (float) d1[2] * dA3 * dW0;
                        acc[jj * 8 + 7] += (float) d1[3] * dA3 * dW1;
                        acc[jj * 8 + 8] += (float) e0[0] * dA0 * eW0;
                        acc[jj * 8 + 9] += (float) e0[1] * dA0 * eW1;
                        acc[jj * 8 + 10] += (float) e0[2] * dA1 * eW0;
                        acc[jj * 8 + 11] += (float) e0[3] * dA1 * eW1;
                        acc[jj * 8 + 12] += (float) e1[0] * dA2 * eW0;
                        acc[jj * 8 + 13] += (float) e1[1] * dA2 * eW1;
                        acc[jj * 8 + 14] += (float) e1[2] * dA3 * eW0;
                        acc[jj * 8 + 15] += (float) e1[3] * dA3 * eW1;
                    }
                }
                if (hasNext) {
                    int dst = bufNext * 2 * TILE_WORDS + slot;
                    bTile[dst] = l0 | (h0 << 16);
                    bTile[dst + 512] = l1 | (h1 << 16);
                    bTile[dst + TILE_WORDS] = l2 | (h2 << 16);
                    bTile[dst + TILE_WORDS + 512] = l3 | (h3 << 16);
                    if (tid >= 256) {
                        sW[bufNext * 2 * I8_BN + tid - 256] = scaleNext;
                    }
                    ctx.asyncCopyWaitGroup(0);
                }
                ctx.localBarrier();
            }

            int splitBase = split * m * n;
            for (int j = 0; j < 4; j++) {
                int col = c0 + (j << 3);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 0],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 1],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 2],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 3],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 4],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 5],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 6],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 7],
                        epilogue,
                        splits);
            }
        }
    }

    /**
     * Bytes of one packed Q4_0 tile of {@link #packQ4_0Tiles}: 4 KB of nibbles, 512 bytes of
     * scales.
     */
    public static final int PACKED_Q4_TILE_BYTES = 4608;

    public static final int PACKED_Q4_QUANT_BYTES = 4096;

    /**
     * The int8 weight word of a 16-bit group of four Q4_0 nibbles, low nibble first: each nibble
     * spread to its own byte and recentred by eight, byte-wise and without carries ({@code (v |
     * 0x80) - 8} keeps each byte at or above 0x78, and the XOR restores the sign bit).
     */
    private static int q4Word(int group) {
        int x =
                (group & 0xF)
                        | ((group & 0xF0) << 4)
                        | ((group & 0xF00) << 8)
                        | ((group & 0xF000) << 12);
        return ((x | 0x80808080) - 0x08080808) ^ 0x80808080;
    }

    /** Stages one packed Q4_0 round, at byte {@code tile}, into weight buffer {@code buf}. */
    private static void stagePackedQ4_0(
            int[] bTile, float[] sW, ByteArray packed, int tile, int buf, int slot, int tid) {
        long nibbles = packed.getLong(tile + (tid << 3));
        int dst = buf * 2 * TILE_WORDS + slot;
        bTile[dst] = q4Word((int) nibbles);
        bTile[dst + 512] = q4Word((int) (nibbles >>> 16));
        bTile[dst + TILE_WORDS] = q4Word((int) (nibbles >>> 32));
        bTile[dst + TILE_WORDS + 512] = q4Word((int) (nibbles >>> 48));
        if (tid >= 256) {
            sW[buf * 2 * I8_BN + tid - 256] =
                    packed.getHalfFloat(tile + PACKED_Q4_QUANT_BYTES + ((tid - 256) << 1))
                            .getFloat32();
        }
    }

    // @formatter:off
    /**
     * Repacks Q4_0 weights ({@code n} rows of {@code k}) for {@link #gemmInt8Q4_0Packed}: one
     * 4608-byte tile per 128 columns and 64-k round, column tile major, the same bytes as the 256
     * Q4_0 blocks it holds. Thread {@code t} of the GEMM finds at byte {@code 8t} the sixteen
     * nibbles of the four weight words it stages (gemmInt8Q4_0's words 0 to 3 of slot {@code t}),
     * four nibbles a word in the word's byte order, low nibble first; then the 256 block scales as
     * FP16, block-major (scale {@code j} is column {@code j & 127} of the round's block {@code j >>
     * 7}). Requires {@code n % 128 == 0} and {@code k % 64 == 0}.
     */
    // @formatter:on
    public static ByteArray packQ4_0Tiles(ByteArray w, int n, int k) {
        return ByteArray.fromArray(packQ4_0TileBytes(w, n, k));
    }

    /** The bytes of {@link #packQ4_0Tiles}, on the heap. */
    public static byte[] packQ4_0TileBytes(ByteArray w, int n, int k) {
        return PackedTilePacker.packQ4_0(w, n, k);
    }

    // @formatter:off
    /**
     * {@link #gemmInt8Q4_0} over weights packed by {@link #packQ4_0Tiles}: each lane's next-round
     * weights are one aligned 8-byte load rather than eight 2-byte ones, expanded to the same int8
     * words after this round's MMAs. Same tiles, fragments, arithmetic and summation order, so the
     * results are equal to {@link #gemmInt8Q4_0}'s bit for bit. Same parameters, requirements and
     * worker, with {@code packed} in place of the Q4_0 weights.
     */
    // @formatter:on
    public static void gemmInt8Q4_0Packed(
            KernelContext ctx,
            ByteArray a8,
            FloatArray dA,
            ByteArray packed,
            FloatArray out,
            FloatArray gate,
            int m,
            int n,
            int k,
            int epilogue,
            FloatArray partial,
            int splits,
            IntArray rowsHolder,
            int ldo,
            int colOffset) {
        int tid = ctx.localIdx;
        int warpId = tid >> 5;
        int lane = tid & 31;
        int warpM = warpId / Q8_WARPS_N;
        int warpN = warpId - warpM * Q8_WARPS_N;
        int blockRow = I8_BM * ctx.groupIdx;
        int colTile = ctx.groupIdy / splits;
        int split = ctx.groupIdy - colTile * splits;
        int blockCol = I8_BN * colTile;
        // A chunk's real rows are rowsHolder[1]; a block wholly past them has nothing to do.
        if (blockRow < m && blockRow < rowsHolder.get(1) && blockCol < n) {
            int kBlocks = k / Q8_BLOCK;
            int rounds = k / I8_BK;
            // This block's share of the rounds; with one split, all of them.
            int roundsPerSplit = (rounds + splits - 1) / splits;
            int roundBegin = split * roundsPerSplit;
            int roundEnd = Math.min(rounds, roundBegin + roundsPerSplit);

            int[] aTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            int[] bTile = ctx.allocateIntLocalArray(2 * 2 * TILE_WORDS);
            float[] sA = ctx.allocateFloatLocalArray(2 * 2 * I8_BM);
            float[] sW = ctx.allocateFloatLocalArray(2 * 2 * I8_BN);

            float[] acc = new float[32];
            for (int i = 0; i < 32; i++) {
                acc[i] = 0.0f;
            }
            int rowInWarp = lane >> 2;
            int colInWarp = (lane & 3) << 1;
            int r0 = blockRow + warpM * I8_WM + rowInWarp;
            int c0 = blockCol + warpN * Q8_WN + colInWarp;

            // This lane's four weight words of a round, as in gemmInt8Q4_0; the packed tile holds
            // their sixteen nibbles as this lane's eight bytes.
            int kRow = tid & 15;
            int colPair = (tid >> 4) & 31;
            int slot = ((colPair >> 2) << 6) + (kRow << 2) + (colPair & 3);
            int tileBase = colTile * rounds * PACKED_Q4_TILE_BYTES;
            if (roundBegin < roundEnd) {
                stageActivationsQ8_0(
                        ctx, aTile, sA, a8, dA, roundBegin, 0, blockRow, k, kBlocks, tid);
                stagePackedQ4_0(
                        bTile,
                        sW,
                        packed,
                        tileBase + roundBegin * PACKED_Q4_TILE_BYTES,
                        0,
                        slot,
                        tid);
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(0);
                ctx.localBarrier();
            }

            for (int round = roundBegin; round < roundEnd; round++) {
                int buf = (round - roundBegin) & 1;
                int bufNext = 1 - buf;
                boolean hasNext = round + 1 < roundEnd;
                // The next round's weight words, loaded now and stored after this round's MMAs,
                // so the loads are in flight while the tensor cores work.
                int nextTile = tileBase + (round + 1) * PACKED_Q4_TILE_BYTES;
                long nibbles = 0L;
                float scaleNext = 0.0f;
                if (hasNext) {
                    stageActivationsQ8_0(
                            ctx, aTile, sA, a8, dA, round + 1, bufNext, blockRow, k, kBlocks, tid);
                    ctx.asyncCopyCommit();
                    nibbles = packed.getLong(nextTile + (tid << 3));
                    if (tid >= 256) {
                        scaleNext =
                                packed.getHalfFloat(
                                                nextTile
                                                        + PACKED_Q4_QUANT_BYTES
                                                        + ((tid - 256) << 1))
                                        .getFloat32();
                    }
                }
                int aBuf = buf * 2 * TILE_WORDS;
                int sBuf = buf * 2 * I8_BM;
                for (int b = 0; b < 2; b++) {
                    int aOff = ((aBuf + b * TILE_WORDS) << 2) + warpM * 1024;
                    byte[] a0 = ctx.mmaLoadAInt8(aTile, 32, aOff);
                    byte[] a1 = ctx.mmaLoadAInt8(aTile, 32, aOff + 512);
                    int sRow = sBuf + b * I8_BM + warpM * I8_WM + rowInWarp;
                    float dA0 = sA[sRow];
                    float dA1 = sA[sRow + 8];
                    float dA2 = sA[sRow + 16];
                    float dA3 = sA[sRow + 24];
                    int bOff = ((aBuf + b * TILE_WORDS) << 2) + warpN * 1024;
                    int sCol = sBuf + b * I8_BN + warpN * Q8_WN + colInWarp;
                    // Two column steps at a time: their four MMAs issue before either's results
                    // are scaled, so the scaling of one waits behind the other's MMAs rather than
                    // stalling on its own.
                    for (int jj = 0; jj < 4; jj += 2) {
                        byte[] fb0 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256);
                        byte[] fb1 = ctx.mmaLoadBInt8(bTile, 32, bOff + jj * 256 + 256);
                        int[] d0 = ctx.mmaInt8(a0, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] d1 = ctx.mmaInt8(a1, fb0, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e0 = ctx.mmaInt8(a0, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        int[] e1 = ctx.mmaInt8(a1, fb1, ctx.mmaFragmentInt(0), MMAShape.M16N8K32);
                        float dW0 = sW[sCol + (jj << 3)];
                        float dW1 = sW[sCol + (jj << 3) + 1];
                        float eW0 = sW[sCol + (jj << 3) + 8];
                        float eW1 = sW[sCol + (jj << 3) + 9];
                        acc[jj * 8 + 0] += (float) d0[0] * dA0 * dW0;
                        acc[jj * 8 + 1] += (float) d0[1] * dA0 * dW1;
                        acc[jj * 8 + 2] += (float) d0[2] * dA1 * dW0;
                        acc[jj * 8 + 3] += (float) d0[3] * dA1 * dW1;
                        acc[jj * 8 + 4] += (float) d1[0] * dA2 * dW0;
                        acc[jj * 8 + 5] += (float) d1[1] * dA2 * dW1;
                        acc[jj * 8 + 6] += (float) d1[2] * dA3 * dW0;
                        acc[jj * 8 + 7] += (float) d1[3] * dA3 * dW1;
                        acc[jj * 8 + 8] += (float) e0[0] * dA0 * eW0;
                        acc[jj * 8 + 9] += (float) e0[1] * dA0 * eW1;
                        acc[jj * 8 + 10] += (float) e0[2] * dA1 * eW0;
                        acc[jj * 8 + 11] += (float) e0[3] * dA1 * eW1;
                        acc[jj * 8 + 12] += (float) e1[0] * dA2 * eW0;
                        acc[jj * 8 + 13] += (float) e1[1] * dA2 * eW1;
                        acc[jj * 8 + 14] += (float) e1[2] * dA3 * eW0;
                        acc[jj * 8 + 15] += (float) e1[3] * dA3 * eW1;
                    }
                }
                if (hasNext) {
                    int dst = bufNext * 2 * TILE_WORDS + slot;
                    bTile[dst] = q4Word((int) nibbles);
                    bTile[dst + 512] = q4Word((int) (nibbles >>> 16));
                    bTile[dst + TILE_WORDS] = q4Word((int) (nibbles >>> 32));
                    bTile[dst + TILE_WORDS + 512] = q4Word((int) (nibbles >>> 48));
                    if (tid >= 256) {
                        sW[bufNext * 2 * I8_BN + tid - 256] = scaleNext;
                    }
                    ctx.asyncCopyWaitGroup(0);
                }
                ctx.localBarrier();
            }

            int splitBase = split * m * n;
            for (int j = 0; j < 4; j++) {
                int col = c0 + (j << 3);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 0],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        r0,
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 1],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 2],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 8),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 3],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 4],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 16),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 5],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 6],
                        epilogue,
                        splits);
                storeQ8_0(
                        out,
                        gate,
                        partial,
                        splitBase,
                        (r0 + 24),
                        col + 1,
                        n,
                        ldo,
                        colOffset,
                        acc[j * 8 + 7],
                        epilogue,
                        splits);
            }
        }
    }

    /**
     * One output of {@link #gemmInt8Q8_0}, by its epilogue, at row {@code row * ldo + colOffset +
     * col} of {@code out}; with several splits, this split's partial sum instead, for {@link
     * #reduceSplitsQ8_0} to finish, which is always laid out {@code n} wide.
     */
    private static void storeQ8_0(
            FloatArray out,
            FloatArray gate,
            FloatArray partial,
            int splitBase,
            int row,
            int col,
            int n,
            int ldo,
            int colOffset,
            float product,
            int epilogue,
            int splits) {
        int index = row * ldo + colOffset + col;
        if (splits > 1) {
            partial.set(splitBase + row * n + col, product);
        } else if (epilogue == EPILOGUE_RESIDUAL) {
            out.set(index, out.get(index) + product);
        } else if (epilogue == EPILOGUE_SWIGLU) {
            float g = gate.get(index);
            out.set(index, (g / (1.0f + TornadoMath.exp(-g))) * product);
        } else {
            out.set(index, product);
        }
    }

    /** The 16 bits at byte {@code offset} of {@code w}, zero-extended. */
    private static int halfBits(ByteArray w, int offset) {
        return w.getHalfFloat(offset).getHalfFloatValue() & 0xFFFF;
    }

    // @formatter:off
    /**
     * Finishes a split {@link #gemmInt8Q8_0}: {@code out[i] (+)= partial[0][i] + ... +
     * partial[splits - 1][i]}, the splits added in order so the result does not depend on which
     * finished first. {@code epilogue} is {@link #EPILOGUE_STORE} or {@link #EPILOGUE_RESIDUAL}; a
     * SwiGLU projection is never split. Worker: {@code total} lanes, any local size.
     */
    // @formatter:on
    public static void reduceSplitsQ8_0(
            KernelContext ctx,
            FloatArray partial,
            FloatArray out,
            int total,
            int splits,
            int epilogue,
            IntArray rowsHolder,
            int n) {
        int i = ctx.globalIdx;
        if (i < total && i < rowsHolder.get(1) * n) {
            float sum = partial.get(i);
            for (int s = 1; s < splits; s++) {
                sum += partial.get(s * total + i);
            }
            out.set(i, epilogue == EPILOGUE_RESIDUAL ? out.get(i) + sum : sum);
        }
    }

    /**
     * Stages round {@code round}'s activation tile of {@link #gemmInt8Q8_0} by {@code cp.async}, as
     * {@link #stageRound} does but over 512 lanes, and its activation scales (the first 256 lanes).
     */
    private static void stageActivationsQ8_0(
            KernelContext ctx,
            int[] aTile,
            float[] sA,
            ByteArray a8,
            FloatArray dA,
            int round,
            int buf,
            int blockRow,
            int k,
            int kBlocks,
            int tid) {
        // One 16-byte copy per lane: a row's two 32-byte blocks of the round are four 16-byte
        // chunks, 512 in all. Four 4-byte copies each cost a sixth of the GEMM's time.
        int kb0 = round * 2;
        int b = tid >> 8;
        int row = (tid >> 1) & 127;
        int half = tid & 1;
        ctx.asyncCopyToLocal16(
                aTile,
                buf * 2 * TILE_WORDS + b * TILE_WORDS + (row << 3) + (half << 2),
                a8,
                (blockRow + row) * k + (kb0 + b) * Q8_BLOCK + (half << 4));
        if (tid < 256) {
            sA[buf * 2 * I8_BM + tid] =
                    dA.get((blockRow + (tid & 127)) * kBlocks + kb0 + (tid >> 7));
        }
    }

    /**
     * Stages round {@code round}'s weight tile of {@link #gemmInt8Q8_0} from the Q8_0 blocks by
     * plain loads, this lane's four words as the main loop's prefetch takes them, and the weight
     * scales (the last 256 lanes).
     */
    private static void stageWeightsQ8_0(
            int[] bTile,
            float[] sW,
            ByteArray w,
            int round,
            int buf,
            int blockCol,
            int rowBytes,
            int laneOffset,
            int halfStride,
            int slot,
            int tid) {
        int base = round * 2 * Q8_0_BLOCK_BYTES + laneOffset;
        int dst = buf * 2 * TILE_WORDS + slot;
        for (int s = 0; s < 4; s++) {
            int offset = base + (s >> 1) * Q8_0_BLOCK_BYTES + (s & 1) * halfStride;
            bTile[dst + (s >> 1) * TILE_WORDS + (s & 1) * 512] =
                    halfBits(w, offset) | (halfBits(w, offset + rowBytes) << 16);
        }
        if (tid >= 256) {
            sW[buf * 2 * I8_BN + tid - 256] =
                    w.getHalfFloat(
                                    (blockCol + (tid & 127)) * rowBytes
                                            + (round * 2 + ((tid >> 7) & 1)) * Q8_0_BLOCK_BYTES)
                            .getFloat32();
        }
    }

    /** {@link #stageWeightsQ8_0} over Q4_0 blocks: each pair decoded by {@link #q4Pair}. */
    private static void stageWeightsQ4_0(
            int[] bTile,
            float[] sW,
            ByteArray w,
            int round,
            int buf,
            int blockCol,
            int rowBytes,
            int laneOffset,
            int halfStride,
            int slot,
            int tid,
            int nibbleShift) {
        int base = round * 2 * Q4_0_BLOCK_BYTES + laneOffset;
        int dst = buf * 2 * TILE_WORDS + slot;
        for (int s = 0; s < 4; s++) {
            int offset = base + (s >> 1) * Q4_0_BLOCK_BYTES + (s & 1) * halfStride;
            bTile[dst + (s >> 1) * TILE_WORDS + (s & 1) * 512] =
                    q4Pair(w, offset, nibbleShift)
                            | (q4Pair(w, offset + rowBytes, nibbleShift) << 16);
        }
        if (tid >= 256) {
            sW[buf * 2 * I8_BN + tid - 256] =
                    w.getHalfFloat(
                                    (blockCol + (tid & 127)) * rowBytes
                                            + (round * 2 + ((tid >> 7) & 1)) * Q4_0_BLOCK_BYTES)
                            .getFloat32();
        }
    }

    /**
     * Two consecutive Q4_0 quants as the 16-bit pair of signed bytes the weight tile holds: the
     * nibbles at {@code shift} of the two bytes at {@code offset}, recentred by eight.
     */
    private static int q4Pair(ByteArray w, int offset, int shift) {
        int v = halfBits(w, offset);
        int q0 = ((v >> shift) & 0xF) - 8;
        int q1 = ((v >> (8 + shift)) & 0xF) - 8;
        return (q0 & 0xFF) | ((q1 & 0xFF) << 8);
    }

    private static final int Q4_0_BLOCK_BYTES = 18;

    /**
     * Issues round {@code round}'s copies into buffer {@code buf} (not committed) and stages its
     * scales. Word {@code w = tid + 256 s} of the activation tile: block {@code w >> 10}, row
     * {@code (w >> 3) & 127}, quad {@code w & 7}; of the weight tile: block {@code w >> 10}, word
     * {@code w & 1023} of that decoded tile.
     */
    private static void stageRound(
            KernelContext ctx,
            int[] aTile,
            int[] bTile,
            float[] sA,
            float[] sW,
            ByteArray a8,
            FloatArray dA,
            ByteArray w8,
            FloatArray dW,
            int round,
            int buf,
            int blockRow,
            int blockCol,
            int k,
            int kBlocks,
            int wTileBase,
            int tid) {
        int kb0 = round * 2;
        int dst = buf * 2 * TILE_WORDS;
        for (int s = 0; s < 8; s++) {
            int w = tid + (s << 8);
            int b = w >> 10;
            int row = (w >> 3) & 127;
            int quad = w & 7;
            ctx.asyncCopyToLocal(
                    aTile, dst + w, a8, (blockRow + row) * k + (kb0 + b) * Q8_BLOCK + (quad << 2));
            ctx.asyncCopyToLocal(
                    bTile,
                    dst + w,
                    w8,
                    ((wTileBase + (kb0 + b) * TILE_WORDS) << 2) + ((w & 1023) << 2));
        }
        int sBase = buf * 2 * I8_BM;
        // 256 activation scales (2 blocks x 128 rows) and 256 weight scales, one each per lane.
        int b = tid >> 7;
        int r = tid & 127;
        sA[sBase + tid] = dA.get((blockRow + r) * kBlocks + kb0 + b);
        sW[sBase + tid] = dW.get((blockCol + r) * kBlocks + kb0 + b);
    }

    // @formatter:off
    /**
     * Per-32-block int8 quantization of an FP16 chunk for the int8 GEMMs, a lane per element: the
     * arithmetic of {@link Int8GemmKernels#quantizeActivationsQ8Warp} on the widened halves. The 32
     * lanes of a block find its {@code amax} by a shuffle-max, each quantizes its own element, and
     * lane zero writes the scale. Lanes at or past {@code total} read zero and store nothing.
     * Worker: {@code total} lanes rounded up to the local size, local a multiple of 32.
     */
    // @formatter:on
    public static void quantizeActivationsQ8WarpFP16(
            KernelContext ctx, HalfFloatArray x, ByteArray q8, FloatArray scales, int total) {
        int lane = ctx.globalIdx;
        int sub = ctx.localIdx & 31;
        float v0 = 0.0f;
        if (lane < total) {
            v0 = x.get(lane).getFloat32();
        }
        float amax = TornadoMath.abs(v0);
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 16));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 8));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 4));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 2));
        amax = TornadoMath.max(amax, ctx.simdShuffleDown(amax, 1));
        amax = ctx.simdBroadcastFirst(amax);
        float d = amax / 127.0f;
        int qv = 0;
        if (d > 0.0f) {
            float v;
            if (amax >= Int8GemmKernels.RECIPROCAL_FINITE_AMAX) {
                v = v0 * (127.0f / amax);
            } else {
                v = (v0 / amax) * 127.0f;
            }
            qv = (int) TornadoMath.floor(TornadoMath.abs(v) + 0.5f);
            if (qv > 127) {
                qv = 127;
            }
            if (v < 0.0f) {
                qv = -qv;
            }
        }
        if (lane < total) {
            q8.set(lane, (byte) qv);
            if (sub == 0) {
                scales.set(lane >> 5, d);
            }
        }
    }
}
