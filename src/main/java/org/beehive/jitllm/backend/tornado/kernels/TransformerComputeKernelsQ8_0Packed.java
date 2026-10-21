package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Single-token matrix-vector products over Q8_0 weights packed by {@link
 * Int8GemmKernels#packQ8_0Tiles}, against the activation quantized per 32-block (the quants of
 * {@link TransformerComputeKernelsQ8_0DP4A}, four to an int, and one scale per block).
 *
 * <p>A workgroup computes eight consecutive output rows, its {@link #WARPS} warps splitting the
 * 64-k rounds between them (warp {@code w} takes rounds {@code w, w + WARPS, ...}) so that enough
 * warps are in flight to keep the memory system busy; their sums meet in shared memory. In the
 * packed layout, eight rows of one 32-block are 64 consecutive words (256 bytes), so each step of
 * the warp is one coalesced read: lane {@code L} takes words {@code 2L} and {@code 2L + 1}, which
 * hold quants {@code k, k + 1} ({@code k = 2 * (L >> 1)} in the block) of four rows, {@code 4 * (L
 * & 1)} to {@code 4 * (L & 1) + 3} of the eight. Each lane scales its integer products by the
 * weight and activation block scales and accumulates in FP32; the lanes of a row are summed by
 * shuffles at the end.
 *
 * <p>Worker: a workgroup per eight rows, {@code d / 8 * LOCAL} lanes with local {@link #LOCAL}
 * ({@code d / 8 * FUSED_LOCAL} and {@link #FUSED_LOCAL} for the fused gate/up). Requires {@code d %
 * 128 == 0} and {@code n % 64 == 0}, as the packing does.
 */
// @formatter:on
public final class TransformerComputeKernelsQ8_0Packed {

    /** Warps of a matrix-vector workgroup, all on the same eight rows. */
    public static final int WARPS = 16;

    /** Threads of a matrix-vector workgroup. */
    public static final int LOCAL = WARPS * 32;

    /**
     * Warps of a fused gate/up workgroup. Each reads both weights and two rounds per step, so eight
     * warps keep as many loads in flight as sixteen do on one weight.
     */
    public static final int FUSED_WARPS = 8;

    /** Threads of a fused gate/up workgroup. */
    public static final int FUSED_LOCAL = FUSED_WARPS * 32;

    private static final int TILE_BYTES = Int8GemmKernels.PACKED_TILE_BYTES;

    private static final int QUANT_BYTES = Int8GemmKernels.PACKED_QUANT_BYTES;

    private TransformerComputeKernelsQ8_0Packed() {}

    /** Signed byte {@code i} of {@code v}. */
    private static int signedByte(int v, int i) {
        return (v << (24 - (i << 3))) >> 24;
    }

    /**
     * This lane's FP32 partial sums of its four rows, {@code rowBase + 4 * (lane & 1)} on, over the
     * rounds {@code firstRound, firstRound + WARPS, ...} of {@code n}, into {@code acc[0..3]}.
     */
    private static void accumulate(
            ByteArray w,
            IntArray xQuants,
            FloatArray xScales,
            int n,
            int rowBase,
            int lane,
            int firstRound,
            float[] acc) {
        int rounds = n >> 6;
        int colTile = rowBase >> 7;
        int half = (rowBase >> 6) & 1;
        int group = (rowBase >> 3) & 7;
        int wordInBlock = (half << 9) + (group << 6) + (lane << 1);
        int scaleColumn = (rowBase & 127) + ((lane & 1) << 2);
        int kRow = lane >> 1;
        int shift = (kRow & 1) << 1;
        int tileBase = colTile * rounds * TILE_BYTES;
        for (int round = firstRound; round < rounds; round += WARPS) {
            int tile = tileBase + round * TILE_BYTES;
            for (int b = 0; b < 2; b++) {
                int block = round * 2 + b;
                // The lane's two words are adjacent and 8-byte aligned: one 64-bit load.
                long pair = w.getLong(tile + (((b << 10) + wordInBlock) << 2));
                int w0 = (int) pair;
                int w1 = (int) (pair >>> 32);
                int xw = xQuants.get((block << 3) + (kRow >> 1));
                int x0 = signedByte(xw, shift);
                int x1 = signedByte(xw, shift + 1);
                int d0 = signedByte(w0, 0) * x0 + signedByte(w0, 1) * x1;
                int d1 = signedByte(w0, 2) * x0 + signedByte(w0, 3) * x1;
                int d2 = signedByte(w1, 0) * x0 + signedByte(w1, 1) * x1;
                int d3 = signedByte(w1, 2) * x0 + signedByte(w1, 3) * x1;
                float xScale = xScales.get(block);
                int scales = tile + QUANT_BYTES + (((b << 7) + scaleColumn) << 1);
                acc[0] += d0 * (w.getHalfFloat(scales).getFloat32() * xScale);
                acc[1] += d1 * (w.getHalfFloat(scales + 2).getFloat32() * xScale);
                acc[2] += d2 * (w.getHalfFloat(scales + 4).getFloat32() * xScale);
                acc[3] += d3 * (w.getHalfFloat(scales + 6).getFloat32() * xScale);
            }
        }
    }

    /** {@link #accumulate} against an FP32 activation rather than a quantized one. */
    private static void accumulateF32(
            ByteArray w, FloatArray x, int n, int rowBase, int lane, int firstRound, float[] acc) {
        int rounds = n >> 6;
        int colTile = rowBase >> 7;
        int half = (rowBase >> 6) & 1;
        int group = (rowBase >> 3) & 7;
        int wordInBlock = (half << 9) + (group << 6) + (lane << 1);
        int scaleColumn = (rowBase & 127) + ((lane & 1) << 2);
        int kInBlock = (lane >> 1) << 1;
        int tileBase = colTile * rounds * TILE_BYTES;
        for (int round = firstRound; round < rounds; round += WARPS) {
            int tile = tileBase + round * TILE_BYTES;
            for (int b = 0; b < 2; b++) {
                int block = round * 2 + b;
                // The lane's two words are adjacent and 8-byte aligned: one 64-bit load.
                long pair = w.getLong(tile + (((b << 10) + wordInBlock) << 2));
                int w0 = (int) pair;
                int w1 = (int) (pair >>> 32);
                int k = (block << 5) + kInBlock;
                float x0 = x.get(k);
                float x1 = x.get(k + 1);
                int scales = tile + QUANT_BYTES + (((b << 7) + scaleColumn) << 1);
                acc[0] +=
                        w.getHalfFloat(scales).getFloat32()
                                * (signedByte(w0, 0) * x0 + signedByte(w0, 1) * x1);
                acc[1] +=
                        w.getHalfFloat(scales + 2).getFloat32()
                                * (signedByte(w0, 2) * x0 + signedByte(w0, 3) * x1);
                acc[2] +=
                        w.getHalfFloat(scales + 4).getFloat32()
                                * (signedByte(w1, 0) * x0 + signedByte(w1, 1) * x1);
                acc[3] +=
                        w.getHalfFloat(scales + 6).getFloat32()
                                * (signedByte(w1, 2) * x0 + signedByte(w1, 3) * x1);
            }
        }
    }

    /**
     * One 32-block of one weight into {@code acc[o..o + 3]}: block {@code b} of the round whose
     * tile starts at byte {@code tile}, the lane's two words in one 64-bit load.
     */
    private static void accumulateBlock(
            ByteArray w,
            IntArray xQuants,
            FloatArray xScales,
            int tile,
            int b,
            int block,
            int wordInBlock,
            int scaleColumn,
            int kRow,
            int shift,
            float[] acc,
            int o) {
        long pair = w.getLong(tile + (((b << 10) + wordInBlock) << 2));
        int w0 = (int) pair;
        int w1 = (int) (pair >>> 32);
        int xw = xQuants.get((block << 3) + (kRow >> 1));
        int x0 = signedByte(xw, shift);
        int x1 = signedByte(xw, shift + 1);
        float xScale = xScales.get(block);
        int scales = tile + QUANT_BYTES + (((b << 7) + scaleColumn) << 1);
        acc[o] +=
                (signedByte(w0, 0) * x0 + signedByte(w0, 1) * x1)
                        * (w.getHalfFloat(scales).getFloat32() * xScale);
        acc[o + 1] +=
                (signedByte(w0, 2) * x0 + signedByte(w0, 3) * x1)
                        * (w.getHalfFloat(scales + 2).getFloat32() * xScale);
        acc[o + 2] +=
                (signedByte(w1, 0) * x0 + signedByte(w1, 1) * x1)
                        * (w.getHalfFloat(scales + 4).getFloat32() * xScale);
        acc[o + 3] +=
                (signedByte(w1, 2) * x0 + signedByte(w1, 3) * x1)
                        * (w.getHalfFloat(scales + 6).getFloat32() * xScale);
    }

    /** Sums a value over the lanes of this warp with the same parity (shuffles by even offsets). */
    private static float reduceParity(KernelContext context, float v) {
        float s = v;
        s += context.simdShuffleDown(s, 16);
        s += context.simdShuffleDown(s, 8);
        s += context.simdShuffleDown(s, 4);
        s += context.simdShuffleDown(s, 2);
        return s;
    }

    /** {@code output[row] = w[row]·x} (or {@code +=} with {@code residual} 1). */
    public static void matrixVectorQ8_0Packed(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int residual) {
        int rowBase = context.groupIdx << 3;
        int lane = context.localIdx & 31;
        if (rowBase < d) {
            float[] sums = context.allocateFloatLocalArray(WARPS * 8);
            float[] acc = new float[4];
            for (int i = 0; i < 4; i++) {
                acc[i] = 0.0f;
            }
            accumulate(w, xQuants, xScales, n, rowBase, lane, context.localIdx >> 5, acc);
            int warp = context.localIdx >> 5;
            float r0 = reduceParity(context, acc[0]);
            float r1 = reduceParity(context, acc[1]);
            float r2 = reduceParity(context, acc[2]);
            float r3 = reduceParity(context, acc[3]);
            if (lane < 2) {
                int slot = (warp << 3) + (lane << 2);
                sums[slot] = r0;
                sums[slot + 1] = r1;
                sums[slot + 2] = r2;
                sums[slot + 3] = r3;
            }
            context.localBarrier();
            int row = context.localIdx;
            if (row < 8) {
                float total = 0.0f;
                for (int wi = 0; wi < WARPS; wi++) {
                    total += sums[(wi << 3) + row];
                }
                if (residual != 0) {
                    output.set(rowBase + row, output.get(rowBase + row) + total);
                } else {
                    output.set(rowBase + row, total);
                }
            }
        }
    }

    /**
     * {@code output[row] = w[row]·x} (or {@code +=} with {@code residual} 1), {@code x} in FP32.
     */
    public static void matrixVectorQ8_0PackedF32(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int residual) {
        int rowBase = context.groupIdx << 3;
        int lane = context.localIdx & 31;
        if (rowBase < d) {
            float[] sums = context.allocateFloatLocalArray(WARPS * 8);
            float[] acc = new float[4];
            for (int i = 0; i < 4; i++) {
                acc[i] = 0.0f;
            }
            accumulateF32(w, x, n, rowBase, lane, context.localIdx >> 5, acc);
            int warp = context.localIdx >> 5;
            float r0 = reduceParity(context, acc[0]);
            float r1 = reduceParity(context, acc[1]);
            float r2 = reduceParity(context, acc[2]);
            float r3 = reduceParity(context, acc[3]);
            if (lane < 2) {
                int slot = (warp << 3) + (lane << 2);
                sums[slot] = r0;
                sums[slot + 1] = r1;
                sums[slot + 2] = r2;
                sums[slot + 3] = r3;
            }
            context.localBarrier();
            int row = context.localIdx;
            if (row < 8) {
                float total = 0.0f;
                for (int wi = 0; wi < WARPS; wi++) {
                    total += sums[(wi << 3) + row];
                }
                if (residual != 0) {
                    output.set(rowBase + row, output.get(rowBase + row) + total);
                } else {
                    output.set(rowBase + row, total);
                }
            }
        }
    }

    /**
     * {@code hb[row] = silu(w1[row]·x) * (w3[row]·x)}, both against the same quantized activation.
     */
    public static void fusedFFNGateUpSiLUQ8_0Packed(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d) {
        int rowBase = context.groupIdx << 3;
        int lane = context.localIdx & 31;
        if (rowBase < d) {
            float[] gateSums = context.allocateFloatLocalArray(FUSED_WARPS * 8);
            float[] upSums = context.allocateFloatLocalArray(FUSED_WARPS * 8);
            float[] gate = new float[8];
            float[] up = new float[8];
            for (int i = 0; i < 8; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            // Both weights and two rounds per step, so that eight loads are in flight per lane
            // (inline: as a helper this exceeds TornadoVM's inlining limit). Rounds go to [0..3]
            // and
            // [4..7] by parity of the step, summed at the end.
            int rounds = n >> 6;
            int wordInBlock =
                    (((rowBase >> 6) & 1) << 9) + (((rowBase >> 3) & 7) << 6) + (lane << 1);
            int scaleColumn = (rowBase & 127) + ((lane & 1) << 2);
            int kRow = lane >> 1;
            int shift = (kRow & 1) << 1;
            int tileBase = (rowBase >> 7) * rounds * TILE_BYTES;
            int round = context.localIdx >> 5;
            for (; round + FUSED_WARPS < rounds; round += 2 * FUSED_WARPS) {
                int t0 = tileBase + round * TILE_BYTES;
                int t1 = t0 + FUSED_WARPS * TILE_BYTES;
                for (int b = 0; b < 2; b++) {
                    accumulateBlock(
                            w1,
                            xQuants,
                            xScales,
                            t0,
                            b,
                            (round << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            gate,
                            0);
                    accumulateBlock(
                            w3,
                            xQuants,
                            xScales,
                            t0,
                            b,
                            (round << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            up,
                            0);
                    accumulateBlock(
                            w1,
                            xQuants,
                            xScales,
                            t1,
                            b,
                            ((round + FUSED_WARPS) << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            gate,
                            4);
                    accumulateBlock(
                            w3,
                            xQuants,
                            xScales,
                            t1,
                            b,
                            ((round + FUSED_WARPS) << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            up,
                            4);
                }
            }
            for (; round < rounds; round += FUSED_WARPS) {
                int t0 = tileBase + round * TILE_BYTES;
                for (int b = 0; b < 2; b++) {
                    accumulateBlock(
                            w1,
                            xQuants,
                            xScales,
                            t0,
                            b,
                            (round << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            gate,
                            0);
                    accumulateBlock(
                            w3,
                            xQuants,
                            xScales,
                            t0,
                            b,
                            (round << 1) + b,
                            wordInBlock,
                            scaleColumn,
                            kRow,
                            shift,
                            up,
                            0);
                }
            }
            int warp = context.localIdx >> 5;
            float g0 = reduceParity(context, gate[0] + gate[4]);
            float g1 = reduceParity(context, gate[1] + gate[5]);
            float g2 = reduceParity(context, gate[2] + gate[6]);
            float g3 = reduceParity(context, gate[3] + gate[7]);
            float u0 = reduceParity(context, up[0] + up[4]);
            float u1 = reduceParity(context, up[1] + up[5]);
            float u2 = reduceParity(context, up[2] + up[6]);
            float u3 = reduceParity(context, up[3] + up[7]);
            if (lane < 2) {
                int slot = (warp << 3) + (lane << 2);
                gateSums[slot] = g0;
                gateSums[slot + 1] = g1;
                gateSums[slot + 2] = g2;
                gateSums[slot + 3] = g3;
                upSums[slot] = u0;
                upSums[slot + 1] = u1;
                upSums[slot + 2] = u2;
                upSums[slot + 3] = u3;
            }
            context.localBarrier();
            int row = context.localIdx;
            if (row < 8) {
                float g = 0.0f;
                float u = 0.0f;
                for (int wi = 0; wi < FUSED_WARPS; wi++) {
                    g += gateSums[(wi << 3) + row];
                    u += upSums[(wi << 3) + row];
                }
                hb.set(rowBase + row, g / (1.0f + TornadoMath.exp(-g)) * u);
            }
        }
    }
}
