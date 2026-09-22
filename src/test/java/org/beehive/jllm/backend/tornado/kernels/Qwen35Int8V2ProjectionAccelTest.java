package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * Research: the second int8 Q4_0 projection design ({@code Qwen35Int8Kernels}) — quantize + decode
 * + block-scaled int8 GEMM — checked and screened against the production FP16 pair.
 *
 * <p>Implementation correctness is separated from the activation-quantization error: (1a) the
 * quantizer equals the host quantizer bit for bit; (1b) the decoder's int8 words and FP32 scales
 * equal the host's decode into the same layout; (1c) the GEMM equals the host evaluation of the
 * same quantized operands (exact int64 block dots, then the kernel's FP32 expression with its
 * FMUL+FFMA contraction) bit for bit; (2) both the int8 path and the production FP16 path are
 * compared with an FP64 reference from the original FP32 activations, which reports the
 * quantization tradeoff without hiding it. Needs the TornadoVM extension the kernels document.
 */
// @formatter:on
public class Qwen35Int8V2ProjectionAccelTest {

    private static final int BLOCK_BYTES = 18;

    private static byte[] randomWeights(int n, int k, long seed) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * BLOCK_BYTES;
            int bits = 0x2000 | rng.nextInt(0x400) | (rng.nextBoolean() ? 0x8000 : 0);
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static FloatArray activations(int m, int k, long seed) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            float v = (float) rng.nextGaussian();
            if (rng.nextInt(512) == 0) {
                v *= 20.0f;
            }
            a.set(i, v);
        }
        return a;
    }

    private static float decodedWeight(byte[] raw, int k, int col, int e) {
        return weightScale(raw, k, col, e / 32) * nibble(raw, k, col, e);
    }

    private static int nibble(byte[] raw, int k, int col, int e) {
        int base = (col * (k / 32) + e / 32) * BLOCK_BYTES;
        int packed = raw[base + 2 + (e % 32 & 15)] & 0xFF;
        return ((e % 32) >= 16 ? (packed >> 4) & 0xF : packed & 0xF) - 8;
    }

    private static float weightScale(byte[] raw, int k, int col, int block) {
        int base = (col * (k / 32) + block) * BLOCK_BYTES;
        return new HalfFloat((short) ((raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8)))
                .getFloat32();
    }

    private static void hostQuantize(FloatArray a, int m, int k, byte[] q8, float[] scales) {
        for (int row = 0; row < m; row++) {
            for (int b = 0; b < k / 32; b++) {
                float amax = 0;
                for (int i = 0; i < 32; i++) {
                    amax = Math.max(amax, Math.abs(a.get(row * k + b * 32 + i)));
                }
                float d = amax / 127.0f;
                scales[row * (k / 32) + b] = d;
                for (int i = 0; i < 32; i++) {
                    float x = a.get(row * k + b * 32 + i);
                    int qv = 0;
                    if (d > 0) {
                        float v =
                                amax >= Qwen35Int8Kernels.RECIPROCAL_FINITE_AMAX
                                        ? x * (127.0f / amax)
                                        : (x / amax) * 127.0f;
                        qv = Math.min(127, (int) Math.floor(Math.abs(v) + 0.5f));
                        if (v < 0) {
                            qv = -qv;
                        }
                    }
                    q8[row * k + b * 32 + i] = (byte) qv;
                }
            }
        }
    }

    /** The decoded word layout on the host: byte {@code 4 word + t}. */
    private static byte hostDecodedByte(byte[] raw, int k, int wordIndex, int t) {
        int kBlocks = k / 32;
        int idx = wordIndex & 1023;
        int tile = wordIndex >> 10;
        int colBlock = tile / kBlocks;
        int kBlock = tile - colBlock * kBlocks;
        int sub = idx >> 6;
        int kRow = (idx >> 2) & 15;
        int pair = idx & 3;
        int col = (colBlock << 7) + (sub << 3) + (pair << 1) + (t >> 1);
        int kk = kBlock * 32 + (kRow << 1) + (t & 1);
        return (byte) nibble(raw, k, col, kk);
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid lanes(int count) {
        WorkerGrid g = new WorkerGrid1D(count);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private record Result(
            FloatArray out, ByteArray q8, FloatArray dA, ByteArray w8, FloatArray dW) {}

    private static Result runInt8(int m, int n, int k, FloatArray a, ByteArray w) throws Exception {
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        ByteArray w8 = new ByteArray(n * k);
        FloatArray dW = new FloatArray(n * k / 32);
        dW.init(Float.NaN);
        FloatArray out = new FloatArray(m * n);
        out.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("i8")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, a, w, q8, dA, w8, dW, out)
                        .task(
                                "q",
                                Qwen35Int8Kernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                a,
                                q8,
                                dA,
                                k)
                        .task(
                                "d",
                                Qwen35Int8Kernels::decodeQ4_0ToInt8Tiled,
                                new KernelContext(),
                                w,
                                w8,
                                dW,
                                n,
                                k)
                        .task(
                                "g",
                                Qwen35Int8Kernels::gemmInt8BlockScaled,
                                new KernelContext(),
                                q8,
                                dA,
                                w8,
                                dW,
                                out,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out, q8, dA, w8, dW);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("i8.q", lanes(m * k));
        s.addWorkerGrid("i8.d", lanes(n * k / 4));
        s.addWorkerGrid("i8.g", gemmGrid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        return new Result(out, q8, dA, w8, dW);
    }

    /**
     * The quantizer on crafted blocks, checked against the arithmetic's definition rather than a
     * copy of its code: {@code d = amax / 127}, {@code q = round-half-away-from-zero(x * (127 /
     * amax))}, a zero block gives {@code q = 0, d = 0}. Blocks: all zero; one nonzero element; all
     * negative; exact ties (amax 127 so the inverse is exactly one, values n + 0.5); a single
     * outlier that crushes the rest to zero; tiny magnitudes (1e-30); large magnitudes (1e30);
     * mixed signs with the maximum at both ends. The device bytes and scales must also equal the
     * host quantizer's, and a negative control (a block with one element perturbed) must differ.
     */
    @Test
    public void theQuantizerHandlesTheEdgeBlocks() throws Exception {
        assumeTrue("no tensor-core-capable device", TensorCoreSupport.isTensorCoreCapableBackend());
        int m = 8, k = 64; // 512 lanes: two 256-lane workgroups
        FloatArray a = new FloatArray(m * k);
        a.init(0.0f);
        float[][] blocks = new float[m * 2][32];
        // block 1: one nonzero
        blocks[1][5] = -3.25f;
        // block 2: all negative, descending
        for (int i = 0; i < 32; i++) blocks[2][i] = -(i + 1) * 0.5f;
        // block 3: ties, amax = 127 so inv == 1 exactly
        blocks[3][0] = 127.0f;
        for (int i = 1; i < 32; i++) blocks[3][i] = (i % 2 == 0 ? 1 : -1) * (i + 0.5f);
        // block 4: outlier crushes the rest
        for (int i = 0; i < 32; i++) blocks[4][i] = 0.001f * (i - 16);
        blocks[4][7] = 500.0f;
        // block 5: tiny
        for (int i = 0; i < 32; i++) blocks[5][i] = (i - 15.5f) * 1e-30f;
        // block 6: large
        for (int i = 0; i < 32; i++) blocks[6][i] = (i - 15.5f) * 1e30f;
        // block 7: max at both ends, mixed
        for (int i = 0; i < 32; i++) blocks[7][i] = (float) Math.sin(i * 0.7) * 4.0f;
        blocks[7][0] = 9.0f;
        blocks[7][31] = -9.0f;
        // block 8: amax 1e-37, where 127 / amax overflows FP32 (the old form saturated to -1)
        for (int i = 0; i < 32; i++) blocks[8][i] = (i - 15.5f) / 15.5f * 1e-37f;
        // block 9: subnormal inputs (below FLT_MIN 1.18e-38), scale subnormal
        for (int i = 0; i < 32; i++) blocks[9][i] = (i - 15.5f) / 15.5f * 5e-39f;
        // block 10: amax 3e-44, d = amax / 127 rounds to a subnormal of a few ulps
        for (int i = 0; i < 32; i++) blocks[10][i] = (i - 15.5f) / 15.5f * 3e-44f;
        // block 11: amax 5e-45, d rounds to zero: defined as a zero block
        for (int i = 0; i < 32; i++) blocks[11][i] = (i % 3 == 0 ? 1 : 0) * 5e-45f;
        // block 12: mixed zeros and a few small nonzeros around the overflow edge
        blocks[12][3] = 2e-37f;
        blocks[12][17] = -3.9e-37f;
        blocks[12][31] = 1e-40f;
        // block 13: near FLT_MAX
        for (int i = 0; i < 32; i++) blocks[13][i] = (i - 15.5f) / 15.5f * 3e38f;
        // remaining blocks: gaussian-ish
        Random rng = new Random(99);
        for (int b = 14; b < m * 2; b++)
            for (int i = 0; i < 32; i++) blocks[b][i] = (float) rng.nextGaussian();
        for (int b = 0; b < m * 2; b++)
            for (int i = 0; i < 32; i++) a.set(b * 32 + i, blocks[b][i]);

        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        q8.init((byte) 77);
        dA.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("qz")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, q8, dA)
                        .task(
                                "q",
                                Qwen35Int8Kernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                a,
                                q8,
                                dA,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, q8, dA);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("qz.q", lanes(m * k));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        byte[] hq = new byte[m * k];
        float[] hd = new float[m * k / 32];
        hostQuantize(a, m, k, hq, hd);
        int mismatches = 0;
        for (int b = 0; b < m * 2; b++) {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(blocks[b][i]));
            float d = dA.get(b);
            assertEquals("block " + b + " scale", amax / 127.0f, d, 0.0f);
            for (int i = 0; i < 32; i++) {
                int q = q8.get(b * 32 + i);
                if (q != hq[b * 32 + i]) mismatches++;
                assertTrue("block " + b + " |q| <= 127: " + q, Math.abs(q) <= 127);
                float x = blocks[b][i];
                if (amax == 0) {
                    assertEquals("zero block quantizes to zero", 0, q);
                    continue;
                }
                if (d == 0.0f) {
                    assertEquals(
                            "block " + b + " with an unrepresentable scale is a zero block", 0, q);
                    continue;
                }
                // Definition: q = round-half-away(x * 127 / amax), evaluated in double, clamped.
                double v = (double) x * 127.0 / (double) amax;
                int expected = Math.min(127, (int) Math.floor(Math.abs(v) + 0.5));
                if (v < 0) expected = -expected;
                // The device evaluates in FP32; a value on .5 in double may sit one ulp either
                // side in FP32, so allow the rounding neighbour only.
                assertTrue(
                        "block "
                                + b
                                + " element "
                                + i
                                + " (x="
                                + x
                                + ") q="
                                + q
                                + " expected "
                                + expected,
                        Math.abs(q - expected) <= 1);
                // Reconstruction error within half a step (relative form so subnormal scales
                // count).
                assertTrue(
                        "block "
                                + b
                                + " element "
                                + i
                                + " reconstruction x="
                                + x
                                + " q*d="
                                + (q * d),
                        Math.abs(x - q * d) <= d * 0.5f + Math.abs(x) * 1e-6f + 1e-45f);
            }
        }
        // The ties: exactly n + 0.5 with inverse one rounds away from zero.
        assertEquals("tie 1.5 -> -2 (i=1, negative)", -2, q8.get(3 * 32 + 1));
        assertEquals("tie 2.5 -> 3 (i=2, positive)", 3, q8.get(3 * 32 + 2));
        assertEquals("amax itself -> 127", 127, q8.get(3 * 32));
        assertEquals("outlier -> 127, neighbours -> 0", 127, q8.get(4 * 32 + 7));
        assertEquals("crushed neighbour", 0, q8.get(4 * 32 + 8));
        assertEquals("host quantizer mismatches", 0, mismatches);
        // The overflow edge: every element in bounds and signed correctly, none saturated.
        for (int b : new int[] {8, 9, 10, 12, 13}) {
            for (int i = 0; i < 32; i++) {
                int q = q8.get(b * 32 + i);
                float x = blocks[b][i];
                assertTrue(
                        "block " + b + " sign at " + i,
                        x == 0 ? q == 0 : (x > 0) == (q > 0) || q == 0);
            }
        }
        assertEquals("1e-37 block: amax element -> 127", 127, q8.get(8 * 32 + 31));
        assertEquals("1e-37 block: -amax element -> -127", -127, q8.get(8 * 32 + 0));
        assertEquals("5e-45 block is a zero block", 0.0f, dA.get(11), 0.0f);
        assertEquals("mixed block: 1e-40 against amax 3.9e-37 -> 0", 0, q8.get(12 * 32 + 31));
        assertEquals("mixed block: -3.9e-37 -> -127", -127, q8.get(12 * 32 + 17));
        assertTrue("FLT_MAX block scale finite", Float.isFinite(dA.get(13)));
        // Negative control: perturb one element of a block on the host and require a difference.
        FloatArray a2 = new FloatArray(m * k);
        for (int i = 0; i < m * k; i++) a2.set(i, a.get(i));
        a2.set(2 * 32 + 3, a2.get(2 * 32 + 3) * 0.5f);
        byte[] hq2 = new byte[m * k];
        hostQuantize(a2, m, k, hq2, hd);
        assertTrue("negative control", hq2[2 * 32 + 3] != q8.get(2 * 32 + 3));
        System.out.println(
                "[int8v2] quantizer edge blocks: 16 blocks, 512 elements, all as defined (zero, ties,"
                        + " outlier, 1e-37 reciprocal overflow, subnormal input, subnormal scale, zero"
                        + " scale, mixed, FLT_MAX)");
    }

    /** The residual and SwiGLU epilogues, from the same quantized operands as the store. */
    private static FloatArray[] runInt8Epilogues(
            int m, int n, int k, Result r, FloatArray residualIn, FloatArray gate)
            throws Exception {
        FloatArray res = new FloatArray(m * n);
        for (int i = 0; i < m * n; i++) {
            res.set(i, residualIn.get(i));
        }
        FloatArray hb = new FloatArray(m * n);
        hb.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("i8e")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                r.q8(),
                                r.dA(),
                                r.w8(),
                                r.dW(),
                                res,
                                gate,
                                hb)
                        .task(
                                "r",
                                Qwen35Int8Kernels::gemmInt8BlockScaledResidual,
                                new KernelContext(),
                                r.q8(),
                                r.dA(),
                                r.w8(),
                                r.dW(),
                                res,
                                m,
                                n,
                                k)
                        .task(
                                "s",
                                Qwen35Int8Kernels::gemmInt8BlockScaledSwiGLU,
                                new KernelContext(),
                                r.q8(),
                                r.dA(),
                                r.w8(),
                                r.dW(),
                                gate,
                                hb,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, res, hb);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("i8e.r", gemmGrid(m, n));
        s.addWorkerGrid("i8e.s", gemmGrid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        return new FloatArray[] {res, hb};
    }

    @Test
    public void theEpiloguesMatchTheStoreAndTheSeparateOps() throws Exception {
        assumeTrue("no tensor-core-capable device", TensorCoreSupport.isTensorCoreCapableBackend());
        int m = 256, n = 256, k = 320;
        FloatArray a = activations(m, k, 11L);
        ByteArray w = toDevice(randomWeights(n, k, 12L));
        Result r = runInt8(m, n, k, a, w);
        FloatArray residualIn = activations(m, n, 13L);
        FloatArray gate = activations(m, n, 14L);
        FloatArray[] e = runInt8Epilogues(m, n, k, r, residualIn, gate);
        int bad = 0;
        int badS = 0;
        for (int i = 0; i < m * n; i++) {
            float prod = r.out().get(i);
            float expectedRes = residualIn.get(i) + prod;
            if (Float.floatToRawIntBits(expectedRes) != Float.floatToRawIntBits(e[0].get(i))) {
                bad++;
            }
            float g = gate.get(i);
            float expectedHb = (g / (1.0f + (float) Math.exp(-g))) * prod;
            // exp differs between host and device; compare to 2 ulp of the product scale.
            if (Math.abs(expectedHb - e[1].get(i)) > 4e-6f * Math.max(1.0f, Math.abs(expectedHb))) {
                badS++;
            }
        }
        System.out.printf(
                "[int8v2] epilogues: residual raw-bit mismatches %d/%d, swiglu beyond tolerance %d/%d%n",
                bad, m * n, badS, m * n);
        assertEquals("residual epilogue differs from store + add", 0, bad);
        assertEquals("swiglu epilogue differs from store + host swiglu", 0, badS);
        // Negative control: a shifted residual is not the same.
        int shifted = 0;
        for (int i = 1; i < m * n; i++) {
            if (Float.floatToRawIntBits(residualIn.get(i - 1) + r.out().get(i))
                    != Float.floatToRawIntBits(e[0].get(i))) {
                shifted++;
            }
        }
        assertTrue("negative control did not fail", shifted > m * n / 2);
    }

    private static FloatArray runControl(
            int m, int n, int k, FloatArray a, ByteArray w, HalfFloatArray scratch)
            throws Exception {
        HalfFloatArray a16 = new HalfFloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a16.set(i, new HalfFloat(a.get(i)));
        }
        FloatArray out = new FloatArray(m * n);
        out.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("ctl")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a16, w, scratch, out)
                        .task(
                                "d",
                                Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                new KernelContext(),
                                w,
                                scratch,
                                n,
                                k)
                        .task(
                                "g",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a16,
                                scratch,
                                out,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("ctl.d", lanes(n * k / 2));
        s.addWorkerGrid("ctl.g", gemmGrid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        return out;
    }

    private static void check(String what, int m, int n, int k, long seed) throws Exception {
        FloatArray a = activations(m, k, seed);
        byte[] raw = randomWeights(n, k, seed + 1);
        ByteArray w = toDevice(raw);
        Result r = runInt8(m, n, k, a, w);

        // (1a) quantizer vs host quantizer.
        byte[] hq = new byte[m * k];
        float[] hs = new float[m * k / 32];
        hostQuantize(a, m, k, hq, hs);
        int qDiff = 0;
        for (int i = 0; i < m * k; i++) {
            if (r.q8.get(i) != hq[i]) {
                qDiff++;
            }
        }
        int sDiff = 0;
        for (int i = 0; i < m * k / 32; i++) {
            if (Float.floatToRawIntBits(r.dA.get(i)) != Float.floatToRawIntBits(hs[i])) {
                sDiff++;
            }
        }
        assertEquals(what + ": quantized values differing from the host quantizer", 0, qDiff);
        assertEquals(what + ": scales differing from the host quantizer", 0, sDiff);

        // (1b) decoder vs host decode into the same layout.
        int wDiff = 0;
        for (int i = 0; i < n * k; i++) {
            if (r.w8.get(i) != hostDecodedByte(raw, k, i >> 2, i & 3)) {
                wDiff++;
            }
        }
        assertEquals(what + ": decoded int8 weights differing from the host layout", 0, wDiff);
        int dwDiff = 0;
        for (int col = 0; col < n; col++) {
            for (int b = 0; b < k / 32; b++) {
                if (Float.floatToRawIntBits(r.dW.get(col * (k / 32) + b))
                        != Float.floatToRawIntBits(weightScale(raw, k, col, b))) {
                    dwDiff++;
                }
            }
        }
        assertEquals(what + ": weight scales differing", 0, dwDiff);

        // (1c) GEMM vs the host evaluation of the same operands; (2) both paths vs FP64.
        int exact = 0;
        double errI8 = 0, errCtl = 0, ref2 = 0, maxI8 = 0, maxCtl = 0;
        HalfFloatArray scratch = new HalfFloatArray(n * k);
        FloatArray ctl = runControl(m, n, k, a, w, scratch);
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                float acc = 0.0f;
                double ref = 0;
                for (int b = 0; b < k / 32; b++) {
                    long dot = 0;
                    for (int e = b * 32; e < b * 32 + 32; e++) {
                        dot += (long) hq[row * k + e] * nibble(raw, k, col, e);
                        ref += (double) a.get(row * k + e) * decodedWeight(raw, k, col, e);
                    }
                    acc =
                            Math.fma(
                                    (float) dot * hs[row * (k / 32) + b],
                                    weightScale(raw, k, col, b),
                                    acc);
                }
                float got = r.out.get(row * n + col);
                assertTrue(
                        what + ": int8 output not finite at " + row + "," + col,
                        Float.isFinite(got));
                if (Float.floatToRawIntBits(got) == Float.floatToRawIntBits(acc)) {
                    exact++;
                }
                double e1 = got - ref;
                double e2 = ctl.get(row * n + col) - ref;
                errI8 += e1 * e1;
                errCtl += e2 * e2;
                ref2 += ref * ref;
                maxI8 = Math.max(maxI8, Math.abs(e1));
                maxCtl = Math.max(maxCtl, Math.abs(e2));
            }
        }
        System.out.printf(
                Locale.ROOT,
                "[int8v2] %s: GEMM bit-equal to host %d/%d | vs FP64 from FP32 inputs: int8 path relL2 %.3e maxAbs %.3e; FP16 path relL2 %.3e maxAbs %.3e%n",
                what,
                exact,
                m * n,
                Math.sqrt(errI8 / ref2),
                maxI8,
                Math.sqrt(errCtl / ref2),
                maxCtl);
        assertEquals(what + ": int8 outputs not bit-equal to the host evaluation", m * n, exact);
    }

    @Test
    public void thePathIsExactAgainstTheHostOnSmallShapes() throws Exception {
        check("128x256x64", 128, 256, 64, 1L);
        check("256x128x320", 256, 128, 320, 2L);
    }

    @Test
    public void thePathIsExactAgainstTheHostOnAProductionSlice() throws Exception {
        check("256x512x5120", 256, 512, 5120, 3L);
        // The down projection's reduction length: 272 rounds of 64 k, the longest in the model.
        check("128x128x17408", 128, 128, 17408, 4L);
    }

    /** Complete paths at production shapes; opt in with JLLM_KERNEL_SCREEN=true. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        int[][] shapes = {{17408, 5120}, {5120, 17408}, {10240, 5120}};
        String[] names = {"gate/up", "ffn_down", "ssm_qkv"};
        HalfFloatArray scratch = new HalfFloatArray(17408 * 5120);
        ByteArray w8 = new ByteArray(17408 * 5120);
        FloatArray dW = new FloatArray(17408 * 5120 / 32);
        for (int m : new int[] {512, 2048}) {
            for (int si = 0; si < shapes.length; si++) {
                int n = shapes[si][0];
                int k = shapes[si][1];
                ByteArray w = toDevice(randomWeights(n, k, 7L + n));
                FloatArray a = activations(m, k, 9L + m);
                HalfFloatArray a16 = new HalfFloatArray(m * k);
                for (int i = 0; i < m * k; i++) {
                    a16.set(i, new HalfFloat(a.get(i)));
                }
                ByteArray q8 = new ByteArray(m * k);
                FloatArray dA = new FloatArray(m * k / 32);
                FloatArray o1 = new FloatArray(m * n);
                FloatArray o2 = new FloatArray(m * n);
                TaskGraph ctl =
                        new TaskGraph("c")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a16, w, scratch)
                                .task(
                                        "d",
                                        Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        Qwen35MMAKernels::gemmMMATiledB,
                                        new KernelContext(),
                                        a16,
                                        scratch,
                                        o1,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, o1);
                TaskGraph cand =
                        new TaskGraph("i")
                                .transferToDevice(
                                        DataTransferMode.FIRST_EXECUTION, a, w, q8, dA, w8, dW)
                                .task(
                                        "q",
                                        Qwen35Int8Kernels::quantizeActivationsQ8Warp,
                                        new KernelContext(),
                                        a,
                                        q8,
                                        dA,
                                        k)
                                .task(
                                        "d",
                                        Qwen35Int8Kernels::decodeQ4_0ToInt8Tiled,
                                        new KernelContext(),
                                        w,
                                        w8,
                                        dW,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        Qwen35Int8Kernels::gemmInt8BlockScaled,
                                        new KernelContext(),
                                        q8,
                                        dA,
                                        w8,
                                        dW,
                                        o2,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, o2);
                TaskGraph onlyQ =
                        new TaskGraph("oq")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, q8, dA)
                                .task(
                                        "q",
                                        Qwen35Int8Kernels::quantizeActivationsQ8Warp,
                                        new KernelContext(),
                                        a,
                                        q8,
                                        dA,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, q8);
                TaskGraph onlyD =
                        new TaskGraph("od")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, w8, dW)
                                .task(
                                        "d",
                                        Qwen35Int8Kernels::decodeQ4_0ToInt8Tiled,
                                        new KernelContext(),
                                        w,
                                        w8,
                                        dW,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, w8);
                TaskGraph onlyG =
                        new TaskGraph("og")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, q8, dA, w8, dW)
                                .task(
                                        "g",
                                        Qwen35Int8Kernels::gemmInt8BlockScaled,
                                        new KernelContext(),
                                        q8,
                                        dA,
                                        w8,
                                        dW,
                                        o2,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, o2);
                GridScheduler sc = new GridScheduler();
                sc.addWorkerGrid("c.d", lanes(n * k / 2));
                sc.addWorkerGrid("c.g", gemmGrid(m, n));
                GridScheduler si8 = new GridScheduler();
                si8.addWorkerGrid("i.q", lanes(m * k));
                si8.addWorkerGrid("i.d", lanes(n * k / 4));
                si8.addWorkerGrid("i.g", gemmGrid(m, n));
                GridScheduler sq = new GridScheduler();
                sq.addWorkerGrid("oq.q", lanes(m * k));
                GridScheduler sd = new GridScheduler();
                sd.addWorkerGrid("od.d", lanes(n * k / 4));
                GridScheduler sg = new GridScheduler();
                sg.addWorkerGrid("og.g", gemmGrid(m, n));
                try (TornadoExecutionPlan p1 = new TornadoExecutionPlan(ctl.snapshot());
                        TornadoExecutionPlan p2 = new TornadoExecutionPlan(cand.snapshot());
                        TornadoExecutionPlan pq = new TornadoExecutionPlan(onlyQ.snapshot());
                        TornadoExecutionPlan pd = new TornadoExecutionPlan(onlyD.snapshot());
                        TornadoExecutionPlan pg = new TornadoExecutionPlan(onlyG.snapshot())) {
                    p1.withGridScheduler(sc).withProfiler(ProfilerMode.SILENT);
                    p2.withGridScheduler(si8).withProfiler(ProfilerMode.SILENT);
                    pq.withGridScheduler(sq).withProfiler(ProfilerMode.SILENT);
                    pd.withGridScheduler(sd).withProfiler(ProfilerMode.SILENT);
                    pg.withGridScheduler(sg).withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        p1.execute();
                        p2.execute();
                        pq.execute();
                        pd.execute();
                        pg.execute();
                    }
                    int samples = 15;
                    long[] t1 = new long[samples];
                    long[] t2 = new long[samples];
                    long[] tq = new long[samples];
                    long[] td = new long[samples];
                    long[] tg = new long[samples];
                    for (int i = 0; i < samples; i++) {
                        if ((i & 1) == 0) {
                            t1[i] = kernelNs(p1.execute());
                            t2[i] = kernelNs(p2.execute());
                        } else {
                            t2[i] = kernelNs(p2.execute());
                            t1[i] = kernelNs(p1.execute());
                        }
                        tq[i] = kernelNs(pq.execute());
                        td[i] = kernelNs(pd.execute());
                        tg[i] = kernelNs(pg.execute());
                    }
                    String tag = " " + names[si] + " m=" + m;
                    report("fp16 pair (dq+gemm)" + tag, t1);
                    report("int8 path (q+dec+gemm)" + tag, t2);
                    report("  int8 quantize" + tag, tq);
                    report("  int8 decode" + tag, td);
                    report("  int8 gemm" + tag, tg);
                }
            }
        }
    }

    private static long kernelNs(TornadoExecutionResult result) {
        return result.getProfilerResult().getDeviceKernelTime();
    }

    private static void report(String label, long[] ns) {
        long[] sorted = ns.clone();
        Arrays.sort(sorted);
        StringBuilder samples = new StringBuilder();
        for (long v : ns) {
            samples.append(String.format(Locale.ROOT, "%.1f;", v / 1e3));
        }
        System.out.printf(
                Locale.ROOT,
                "[screen] %-34s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
