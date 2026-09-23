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
public final class Qwen35Int8Kernels {

    private Qwen35Int8Kernels() {}

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
}
