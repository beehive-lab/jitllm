package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

// @formatter:off
/**
 * Single-token matrix-vector products over Q4_0 weights packed by {@link
 * Int8GemmKernels#packQ4_0Tiles}, against the activation quantized per 32-block or in FP32.
 *
 * <p>A workgroup computes sixteen output rows: columns {@code 8g .. 8g + 7} and {@code 64 + 8g ..
 * 64 + 8g + 7} of a 128-column tile, whose nibbles of a round are 512 contiguous bytes (the GEMM
 * lanes {@code 64g .. 64g + 63}). Lane {@code L} of a warp reads two of those lanes' eight bytes,
 * one aligned 64-bit load each: column pair {@code L >> 3} of the group, quants {@code 4 (L & 7)}
 * to {@code 4 (L & 7) + 3} of both blocks of the round, in both column halves. Its nibbles expand
 * to signed bytes, four per row, multiplied by DP4A against one activation word; each product is
 * scaled by its weight and activation block scales and summed in FP32. The {@link #WARPS} warps
 * split the rounds between them and meet in shared memory.
 *
 * <p>Worker: {@code d / 16 * LOCAL} lanes, local {@link #LOCAL}. Requires {@code d % 128 == 0} and
 * {@code n % 64 == 0}, as the packing does.
 */
// @formatter:on
public final class TransformerComputeKernelsQ4_0Packed {

    /**
     * Warps of a matrix-vector workgroup, all on the same sixteen rows (1 to {@link #MAX_WARPS}).
     * The kernels read it from the launch's local size.
     */
    public static final int WARPS = 8;

    /** Threads of a matrix-vector workgroup. */
    public static final int LOCAL = WARPS * 32;

    /** Warps of a fused gate/up workgroup. */
    public static final int FUSED_WARPS = 8;

    /** Threads of a fused gate/up workgroup. */
    public static final int FUSED_LOCAL = FUSED_WARPS * 32;

    /** The most warps a workgroup may have: the size of the shared partial sums. */
    private static final int MAX_WARPS = 16;

    private static final int TILE_BYTES = Int8GemmKernels.PACKED_Q4_TILE_BYTES;

    private static final int QUANT_BYTES = Int8GemmKernels.PACKED_Q4_QUANT_BYTES;

    private TransformerComputeKernelsQ4_0Packed() {}

    /** The int8 word of a 16-bit group of four nibbles (see Int8GemmKernels#q4Word). */
    private static int q4Word(int group) {
        int x =
                (group & 0xF)
                        | ((group & 0xF0) << 4)
                        | ((group & 0xF00) << 8)
                        | ((group & 0xF000) << 12);
        return ((x | 0x80808080) - 0x08080808) ^ 0x80808080;
    }

    /**
     * The four weights of a 16-bit nibble group, each its nibble minus eight, against four
     * activations.
     */
    private static float nibbleDot(int group, float x0, float x1, float x2, float x3) {
        return ((group & 0xF) - 8) * x0
                + (((group >> 4) & 0xF) - 8) * x1
                + (((group >> 8) & 0xF) - 8) * x2
                + (((group >> 12) & 0xF) - 8) * x3;
    }

    /**
     * This lane's sums of its four rows (column pair {@code L >> 3}, both halves: acc[0], acc[1]
     * the pair in the low half, acc[2], acc[3] in the high half) over the rounds {@code firstRound,
     * firstRound + warps, ...}, the activation quantized. Two rounds a step, both loaded before
     * either is used, so each lane keeps four 64-bit loads in flight.
     */
    private static void accumulate(
            ByteArray w,
            IntArray xQuants,
            FloatArray xScales,
            int n,
            int wg,
            int lane,
            int firstRound,
            int warps,
            float[] acc) {
        int rounds = n >> 6;
        int colTile = wg >> 3;
        int group = wg & 7;
        int pair = lane >> 3;
        int quad = lane & 7;
        int column = (group << 3) + (pair << 1);
        int laneBytes = (group << 9) + (lane << 4);
        int tileBase = colTile * rounds * TILE_BYTES;
        for (int round = firstRound; round < rounds; round += warps << 1) {
            int second = round + warps;
            boolean hasSecond = second < rounds;
            int tile = tileBase + round * TILE_BYTES;
            int tile2 = tileBase + (hasSecond ? second : round) * TILE_BYTES;
            long n0 = w.getLong(tile + laneBytes);
            long n1 = w.getLong(tile + laneBytes + 8);
            long m0 = w.getLong(tile2 + laneBytes);
            long m1 = w.getLong(tile2 + laneBytes + 8);
            roundDot(w, xQuants, xScales, tile, round, n0, n1, quad, column, acc);
            if (hasSecond) {
                roundDot(w, xQuants, xScales, tile2, second, m0, m1, quad, column, acc);
            }
        }
    }

    /**
     * {@link #accumulate} over two weights of the same shape at once (gate and up): each step loads
     * both weights' words of a round before using either.
     */
    private static void accumulatePair(
            ByteArray w1,
            ByteArray w3,
            IntArray xQuants,
            FloatArray xScales,
            int n,
            int wg,
            int lane,
            int firstRound,
            int warps,
            float[] gate,
            float[] up) {
        int rounds = n >> 6;
        int colTile = wg >> 3;
        int group = wg & 7;
        int pair = lane >> 3;
        int quad = lane & 7;
        int column = (group << 3) + (pair << 1);
        int laneBytes = (group << 9) + (lane << 4);
        int tileBase = colTile * rounds * TILE_BYTES;
        for (int round = firstRound; round < rounds; round += warps) {
            int tile = tileBase + round * TILE_BYTES;
            long g0 = w1.getLong(tile + laneBytes);
            long g1 = w1.getLong(tile + laneBytes + 8);
            long u0 = w3.getLong(tile + laneBytes);
            long u1 = w3.getLong(tile + laneBytes + 8);
            roundDot(w1, xQuants, xScales, tile, round, g0, g1, quad, column, gate);
            roundDot(w3, xQuants, xScales, tile, round, u0, u1, quad, column, up);
        }
    }

    /** One round's products of this lane: its two loaded nibble words {@code n0}, {@code n1}. */
    private static void roundDot(
            ByteArray w,
            IntArray xQuants,
            FloatArray xScales,
            int tile,
            int round,
            long n0,
            long n1,
            int quad,
            int column,
            float[] acc) {
        // Block 0's two nibble groups are the low 32 bits, block 1's the high: a 64-bit value is
        // only ever shifted by the constant 32 (a variable 64-bit shift miscompiles here).
        int lo0 = (int) n0;
        int hi0 = (int) (n0 >>> 32);
        int lo1 = (int) n1;
        int hi1 = (int) (n1 >>> 32);
        for (int b = 0; b < 2; b++) {
            int g0 = b == 0 ? lo0 : hi0;
            int g1 = b == 0 ? lo1 : hi1;
            int block = (round << 1) + b;
            int x = xQuants.get((block << 3) + quad);
            float xScale = xScales.get(block);
            int scales = tile + QUANT_BYTES + (((b << 7) + column) << 1);
            // Each row's four nibbles gathered while they are a small 16-bit group, then expanded:
            // no step masks or shifts a word with its top bit set (TornadoVM miscompiles such
            // masks here). Group g holds (c, k), (c, k + 1), (c + 1, k), (c + 1, k + 1).
            int g0h = g0 >> 16;
            int g1h = g1 >> 16;
            int even = q4Word((g0 & 0xFF) | ((g1 & 0xFF) << 8));
            int odd = q4Word(((g0 >> 8) & 0xFF) | (((g1 >> 8) & 0xFF) << 8));
            acc[0] +=
                    QuantizationUtils.dp4a_packed(even, x, 0)
                            * (w.getHalfFloat(scales).getFloat32() * xScale);
            acc[1] +=
                    QuantizationUtils.dp4a_packed(odd, x, 0)
                            * (w.getHalfFloat(scales + 2).getFloat32() * xScale);
            even = q4Word((g0h & 0xFF) | ((g1h & 0xFF) << 8));
            odd = q4Word(((g0h >> 8) & 0xFF) | (((g1h >> 8) & 0xFF) << 8));
            acc[2] +=
                    QuantizationUtils.dp4a_packed(even, x, 0)
                            * (w.getHalfFloat(scales + 128).getFloat32() * xScale);
            acc[3] +=
                    QuantizationUtils.dp4a_packed(odd, x, 0)
                            * (w.getHalfFloat(scales + 130).getFloat32() * xScale);
        }
    }

    /** {@link #accumulate} against an FP32 activation. */
    private static void accumulateF32(
            ByteArray w,
            FloatArray x,
            int n,
            int wg,
            int lane,
            int firstRound,
            int warps,
            float[] acc) {
        int rounds = n >> 6;
        int colTile = wg >> 3;
        int group = wg & 7;
        int pair = lane >> 3;
        int quad = lane & 7;
        int column = (group << 3) + (pair << 1);
        int laneBytes = (group << 9) + (lane << 4);
        int tileBase = colTile * rounds * TILE_BYTES;
        for (int round = firstRound; round < rounds; round += warps) {
            int tile = tileBase + round * TILE_BYTES;
            long n0 = w.getLong(tile + laneBytes);
            long n1 = w.getLong(tile + laneBytes + 8);
            // Block 0's two nibble groups are the low 32 bits, block 1's the high: a 64-bit value
            // is
            // only ever shifted by the constant 32 (a variable 64-bit shift miscompiles here).
            int lo0 = (int) n0;
            int hi0 = (int) (n0 >>> 32);
            int lo1 = (int) n1;
            int hi1 = (int) (n1 >>> 32);
            for (int b = 0; b < 2; b++) {
                int g0 = b == 0 ? lo0 : hi0;
                int g1 = b == 0 ? lo1 : hi1;
                int k = (((round << 1) + b) << 5) + (quad << 2);
                float x0 = x.get(k);
                float x1 = x.get(k + 1);
                float x2 = x.get(k + 2);
                float x3 = x.get(k + 3);
                int scales = tile + QUANT_BYTES + (((b << 7) + column) << 1);
                // Rows gathered as nibble groups, as above, and each weight read as its nibble
                // minus
                // eight: no byte is sign-extended (the shifts that would do it miscompile here).
                int g0h = g0 >> 16;
                int g1h = g1 >> 16;
                int even = (g0 & 0xFF) | ((g1 & 0xFF) << 8);
                int odd = ((g0 >> 8) & 0xFF) | (((g1 >> 8) & 0xFF) << 8);
                acc[0] += w.getHalfFloat(scales).getFloat32() * nibbleDot(even, x0, x1, x2, x3);
                acc[1] += w.getHalfFloat(scales + 2).getFloat32() * nibbleDot(odd, x0, x1, x2, x3);
                even = (g0h & 0xFF) | ((g1h & 0xFF) << 8);
                odd = ((g0h >> 8) & 0xFF) | (((g1h >> 8) & 0xFF) << 8);
                acc[2] +=
                        w.getHalfFloat(scales + 128).getFloat32() * nibbleDot(even, x0, x1, x2, x3);
                acc[3] +=
                        w.getHalfFloat(scales + 130).getFloat32() * nibbleDot(odd, x0, x1, x2, x3);
            }
        }
    }

    /** Sums over the eight lanes of this warp that share {@code lane >> 3}. */
    private static float reduceOctet(KernelContext context, float v) {
        float s = v;
        s += context.simdShuffleDown(s, 4);
        s += context.simdShuffleDown(s, 2);
        s += context.simdShuffleDown(s, 1);
        return s;
    }

    /**
     * Output row of slot {@code r} (0..15) of workgroup {@code wg}: pair {@code r >> 2}, then
     * low/high half, then even/odd.
     */
    private static int rowOf(int wg, int r) {
        return ((wg >> 3) << 7)
                + (((r >> 1) & 1) << 6)
                + ((wg & 7) << 3)
                + ((r >> 2) << 1)
                + (r & 1);
    }

    /** {@code output[row] = w[row]·x} (or {@code +=} with {@code residual} 1). */
    public static void matrixVectorQ4_0Packed(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int residual) {
        int wg = context.groupIdx;
        int lane = context.localIdx & 31;
        if ((wg << 4) < d) {
            float[] sums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] acc = new float[4];
            for (int i = 0; i < 4; i++) {
                acc[i] = 0.0f;
            }
            accumulate(
                    w,
                    xQuants,
                    xScales,
                    n,
                    wg,
                    lane,
                    context.localIdx >> 5,
                    context.localGroupSizeX >> 5,
                    acc);
            float r0 = reduceOctet(context, acc[0]);
            float r1 = reduceOctet(context, acc[1]);
            float r2 = reduceOctet(context, acc[2]);
            float r3 = reduceOctet(context, acc[3]);
            if ((lane & 7) == 0) {
                int slot = ((context.localIdx >> 5) << 4) + ((lane >> 3) << 2);
                sums[slot] = r0;
                sums[slot + 1] = r1;
                sums[slot + 2] = r2;
                sums[slot + 3] = r3;
            }
            context.localBarrier();
            int r = context.localIdx;
            if (r < 16) {
                float total = 0.0f;
                for (int wi = 0; wi < (context.localGroupSizeX >> 5); wi++) {
                    total += sums[(wi << 4) + r];
                }
                int row = rowOf(wg, r);
                if (residual != 0) {
                    output.set(row, output.get(row) + total);
                } else {
                    output.set(row, total);
                }
            }
        }
    }

    /**
     * {@code output[row] = w[row]·x} (or {@code +=} with {@code residual} 1), {@code x} in FP32.
     */
    public static void matrixVectorQ4_0PackedF32(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int residual) {
        int wg = context.groupIdx;
        int lane = context.localIdx & 31;
        if ((wg << 4) < d) {
            float[] sums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] acc = new float[4];
            for (int i = 0; i < 4; i++) {
                acc[i] = 0.0f;
            }
            accumulateF32(
                    w, x, n, wg, lane, context.localIdx >> 5, context.localGroupSizeX >> 5, acc);
            float r0 = reduceOctet(context, acc[0]);
            float r1 = reduceOctet(context, acc[1]);
            float r2 = reduceOctet(context, acc[2]);
            float r3 = reduceOctet(context, acc[3]);
            if ((lane & 7) == 0) {
                int slot = ((context.localIdx >> 5) << 4) + ((lane >> 3) << 2);
                sums[slot] = r0;
                sums[slot + 1] = r1;
                sums[slot + 2] = r2;
                sums[slot + 3] = r3;
            }
            context.localBarrier();
            int r = context.localIdx;
            if (r < 16) {
                float total = 0.0f;
                for (int wi = 0; wi < (context.localGroupSizeX >> 5); wi++) {
                    total += sums[(wi << 4) + r];
                }
                int row = rowOf(wg, r);
                if (residual != 0) {
                    output.set(row, output.get(row) + total);
                } else {
                    output.set(row, total);
                }
            }
        }
    }

    /**
     * {@code hb[row] = silu(w1[row]·x) * (w3[row]·x)}, both against the same quantized activation.
     */
    public static void fusedFFNGateUpSiLUQ4_0Packed(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d) {
        int wg = context.groupIdx;
        int lane = context.localIdx & 31;
        if ((wg << 4) < d) {
            float[] gateSums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] upSums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] gate = new float[4];
            float[] up = new float[4];
            for (int i = 0; i < 4; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            int firstRound = context.localIdx >> 5;
            accumulatePair(
                    w1,
                    w3,
                    xQuants,
                    xScales,
                    n,
                    wg,
                    lane,
                    firstRound,
                    context.localGroupSizeX >> 5,
                    gate,
                    up);
            float g0 = reduceOctet(context, gate[0]);
            float g1 = reduceOctet(context, gate[1]);
            float g2 = reduceOctet(context, gate[2]);
            float g3 = reduceOctet(context, gate[3]);
            float u0 = reduceOctet(context, up[0]);
            float u1 = reduceOctet(context, up[1]);
            float u2 = reduceOctet(context, up[2]);
            float u3 = reduceOctet(context, up[3]);
            if ((lane & 7) == 0) {
                int slot = ((context.localIdx >> 5) << 4) + ((lane >> 3) << 2);
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
            int r = context.localIdx;
            if (r < 16) {
                float g = 0.0f;
                float u = 0.0f;
                for (int wi = 0; wi < (context.localGroupSizeX >> 5); wi++) {
                    g += gateSums[(wi << 4) + r];
                    u += upSums[(wi << 4) + r];
                }
                hb.set(rowOf(wg, r), g / (1.0f + TornadoMath.exp(-g)) * u);
            }
        }
    }

    /**
     * {@code hb[row] = gelu(w1[row]·x) * (w3[row]·x)} (Gemma's GeGLU), both against the same
     * quantized activation.
     */
    public static void fusedFFNGateUpGeGLUQ4_0Packed(
            KernelContext context,
            IntArray xQuants,
            FloatArray xScales,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d) {
        int wg = context.groupIdx;
        int lane = context.localIdx & 31;
        if ((wg << 4) < d) {
            float[] gateSums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] upSums = context.allocateFloatLocalArray(MAX_WARPS * 16);
            float[] gate = new float[4];
            float[] up = new float[4];
            for (int i = 0; i < 4; i++) {
                gate[i] = 0.0f;
                up[i] = 0.0f;
            }
            int firstRound = context.localIdx >> 5;
            accumulatePair(
                    w1,
                    w3,
                    xQuants,
                    xScales,
                    n,
                    wg,
                    lane,
                    firstRound,
                    context.localGroupSizeX >> 5,
                    gate,
                    up);
            float g0 = reduceOctet(context, gate[0]);
            float g1 = reduceOctet(context, gate[1]);
            float g2 = reduceOctet(context, gate[2]);
            float g3 = reduceOctet(context, gate[3]);
            float u0 = reduceOctet(context, up[0]);
            float u1 = reduceOctet(context, up[1]);
            float u2 = reduceOctet(context, up[2]);
            float u3 = reduceOctet(context, up[3]);
            if ((lane & 7) == 0) {
                int slot = ((context.localIdx >> 5) << 4) + ((lane >> 3) << 2);
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
            int r = context.localIdx;
            if (r < 16) {
                float g = 0.0f;
                float u = 0.0f;
                for (int wi = 0; wi < (context.localGroupSizeX >> 5); wi++) {
                    g += gateSums[(wi << 4) + r];
                    u += upSums[(wi << 4) + r];
                }
                hb.set(rowOf(wg, r), TransformerComputeKernelsLayered.geluActivation(g) * u);
            }
        }
    }
}
