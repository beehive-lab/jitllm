package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
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
 * Experiment: the Q4_1 and Q5_K projections on the tiled pair — {@code
 * dequantizeQ4_1ToFP16TiledPairs} / {@code dequantizeQ5_KToFP16TiledPairs} with {@code
 * gemmMMATiledB} — against their retained row-major decoders with {@code gemmMMA}.
 *
 * <p>The Q5_K address mapping is proved on the host first: over every lane of several shapes the
 * two half positions a lane writes are a bijection onto the matrix's halves, the tiled layout's
 * inverse returns them to the same row 32 elements apart with the low element in an even sub-block,
 * and the largest address is {@code n * k - 1}. Then on the device, for each format: every half of
 * the tiled candidate bit-equal to the row-major decoder's element at the position the inverse
 * names, over structured nibbles, scales across the finite range and the production shapes, into a
 * NaN-poisoned destination; a candidate with the high half misplaced differs; and the complete
 * pair's FP32 output is raw-bit equal to the retained pair's at the production shapes. An opt-in
 * screen times the complete pairs, control against candidate, alternating.
 */
// @formatter:on
public class Qwen35Q41Q5KTiledPairsDecoderAccelTest {

    private static final int Q4_1_BLOCK_BYTES = 20;
    private static final int Q5_K_SUPER_BYTES = 176;
    private static final int SCRATCH_ELEMENTS = 17408 * 5120;

    enum Format {
        Q4_1,
        Q5_K
    }

    /** The Q5_K candidate's own lane-to-address mapping, restated on the host. */
    static int[] q5kHalves(int k, int lane) {
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tile = ((lane >>> 12) << 2) + ((lane >>> 11) & 1);
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        return new int[] {
            lowHalf, lowHalf + 4096, tile / kSteps, tile % kSteps, kk, sub, pairInSub, parity
        };
    }

    @Test
    public void theQ5KMappingCoversEveryHalfOnceAndPairsTheRightElements() {
        for (int[] shape : new int[][] {{128, 256}, {256, 512}, {384, 1024}, {128, 6144}}) {
            int n = shape[0];
            int k = shape[1];
            boolean[] seen = new boolean[n * k];
            for (int lane = 0; lane < n * k / 2; lane++) {
                int[] c = q5kHalves(k, lane);
                for (int which = 0; which < 2; which++) {
                    int h = c[which];
                    assertTrue("address in range: " + h, h >= 0 && h < n * k);
                    assertTrue("address written once: " + h, !seen[h]);
                    seen[h] = true;
                }
                int[] lo = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, c[0]);
                int[] hi = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, c[1]);
                assertEquals("same row", lo[0], hi[0]);
                assertEquals("elements 32 apart", lo[1] + 32, hi[1]);
                assertEquals("low element in an even sub-block", 0, (lo[1] >> 5) & 1);
                int row = c[2] * 128 + (c[5] << 3) + (c[6] << 1) + c[7];
                int element = c[3] * 16 + c[4];
                assertEquals("kernel row", lo[0], row);
                assertEquals("kernel element", lo[1], element);
            }
            for (int h = 0; h < n * k; h++) {
                assertTrue("half " + h + " never written", seen[h]);
            }
        }
    }

    // ---- weights -------------------------------------------------------------------------------

    private static byte[] q41Weights(int n, int k, long seed, boolean structured, boolean varied) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * Q4_1_BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int c = 0; c < n; c++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int b = c * blocksPerRow + blk;
                int base = b * Q4_1_BLOCK_BYTES;
                int dBits;
                int mBits;
                if (varied) {
                    // Exponents keeping scale * 15 + minimum finite: 1..24 and 1..27.
                    dBits =
                            ((1 + rng.nextInt(24)) << 10)
                                    | rng.nextInt(1024)
                                    | (rng.nextBoolean() ? 0x8000 : 0);
                    mBits =
                            ((1 + rng.nextInt(27)) << 10)
                                    | rng.nextInt(1024)
                                    | (rng.nextBoolean() ? 0x8000 : 0);
                } else {
                    dBits = 0x2C00 | (b & 0xFF) | (((b & 1) == 1) ? 0x8000 : 0);
                    mBits = 0x1C00 | ((b * 7) & 0xFF) | (((b & 2) == 2) ? 0x8000 : 0);
                }
                raw[base] = (byte) dBits;
                raw[base + 1] = (byte) (dBits >> 8);
                raw[base + 2] = (byte) mBits;
                raw[base + 3] = (byte) (mBits >> 8);
                if (structured) {
                    for (int t = 0; t < 16; t++) {
                        int lo = (t + c + blk) & 0xF;
                        int hi = (15 - t + c) & 0xF;
                        raw[base + 4 + t] = (byte) ((hi << 4) | lo);
                    }
                }
            }
        }
        return raw;
    }

    private static byte[] q5kWeights(int n, int k, long seed, boolean structured, boolean varied) {
        int superPerRow = k / 256;
        byte[] raw = new byte[n * superPerRow * Q5_K_SUPER_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int c = 0; c < n; c++) {
            for (int sb = 0; sb < superPerRow; sb++) {
                int base = (c * superPerRow + sb) * Q5_K_SUPER_BYTES;
                int b = c * superPerRow + sb;
                int dBits;
                int dminBits;
                if (varied) {
                    // d * 63 * 31 and dmin * 63 stay finite: exponents 1..17 and 1..22.
                    dBits =
                            ((1 + rng.nextInt(17)) << 10)
                                    | rng.nextInt(1024)
                                    | (rng.nextBoolean() ? 0x8000 : 0);
                    dminBits =
                            ((1 + rng.nextInt(22)) << 10)
                                    | rng.nextInt(1024)
                                    | (rng.nextBoolean() ? 0x8000 : 0);
                } else {
                    dBits = 0x2C00 | (b & 0xFF) | (((b & 1) == 1) ? 0x8000 : 0);
                    dminBits = 0x1C00 | ((b * 7) & 0xFF) | (((b & 2) == 2) ? 0x8000 : 0);
                }
                raw[base] = (byte) dBits;
                raw[base + 1] = (byte) (dBits >> 8);
                raw[base + 2] = (byte) dminBits;
                raw[base + 3] = (byte) (dminBits >> 8);
                if (structured) {
                    for (int q = 0; q < 4; q++) {
                        for (int p = 0; p < 32; p++) {
                            int lo = (p + q + c) & 0xF;
                            int hi = (15 - p + c + q) & 0xF;
                            raw[base + 48 + q * 32 + p] = (byte) ((hi << 4) | lo);
                        }
                    }
                    for (int p = 0; p < 32; p++) {
                        int bits = 0;
                        for (int s = 0; s < 8; s++) {
                            if (((p + s + c) & 1) == 1) {
                                bits |= 1 << s;
                            }
                        }
                        raw[base + 16 + p] = (byte) bits;
                    }
                }
            }
        }
        return raw;
    }

    private static byte[] weights(
            Format f, int n, int k, long seed, boolean structured, boolean varied) {
        return f == Format.Q4_1
                ? q41Weights(n, k, seed, structured, varied)
                : q5kWeights(n, k, seed, structured, varied);
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static WorkerGrid lanes(long count) {
        WorkerGrid g = new WorkerGrid1D((int) count);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    // ---- decoder bit equality ------------------------------------------------------------------

    /**
     * Row-major control and tiled candidate over the same weights; every candidate half compared
     * with the control element the tiled inverse names. Returns the number of halves that differ.
     */
    private static int decodeMismatches(
            String what, Format f, int n, int k, byte[] raw, boolean broken) throws Exception {
        ByteArray w = toDevice(raw);
        HalfFloatArray control = new HalfFloatArray(n * k);
        HalfFloatArray candidate = new HalfFloatArray(n * k);
        control.init(new HalfFloat(Float.NaN));
        candidate.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("dq")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, control, candidate);
        if (f == Format.Q4_1) {
            graph.task(
                            "control",
                            Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
                            new KernelContext(),
                            w,
                            control,
                            n,
                            k)
                    .task(
                            "candidate",
                            broken
                                    ? Qwen35Q41Q5KTiledPairsDecoderAccelTest
                                            ::q41PairsWithTheHighHalfMisplaced
                                    : Qwen35MMAKernels::dequantizeQ4_1ToFP16TiledPairs,
                            new KernelContext(),
                            w,
                            candidate,
                            n,
                            k);
        } else {
            graph.task(
                            "control",
                            Qwen35ReferenceKernels::dequantizeQ5_KToFP16,
                            new KernelContext(),
                            w,
                            control,
                            n,
                            k)
                    .task(
                            "candidate",
                            broken
                                    ? Qwen35Q41Q5KTiledPairsDecoderAccelTest
                                            ::q5kPairsWithTheHighHalfMisplaced
                                    : Qwen35MMAKernels::dequantizeQ5_KToFP16TiledPairs,
                            new KernelContext(),
                            w,
                            candidate,
                            n,
                            k);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("dq.control", lanes((long) n * k));
        s.addWorkerGrid("dq.candidate", lanes((long) n * k / 2));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        String first = null;
        for (int h = 0; h < n * k; h++) {
            int[] e = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, h);
            short c = control.get(e[0] * k + e[1]).getHalfFloatValue();
            short d = candidate.get(h).getHalfFloatValue();
            assertTrue(what + ": control not finite at " + h, (c & 0x7C00) != 0x7C00);
            if (!broken) {
                assertTrue(
                        what
                                + ": candidate not finite at "
                                + h
                                + ": 0x"
                                + Integer.toHexString(d & 0xFFFF),
                        (d & 0x7C00) != 0x7C00);
            }
            if (c != d) {
                if (first == null) {
                    first =
                            "half "
                                    + h
                                    + " (row "
                                    + e[0]
                                    + " element "
                                    + e[1]
                                    + "): control 0x"
                                    + Integer.toHexString(c & 0xFFFF)
                                    + " candidate 0x"
                                    + Integer.toHexString(d & 0xFFFF);
                }
                mismatches++;
            }
        }
        if (!broken) {
            assertEquals(
                    what + ": " + mismatches + " halves differ, first at " + first, 0, mismatches);
        }
        return mismatches;
    }

    @Test
    public void everyQ41NibbleAndBothSignsDecodeToTheRetainedBits() throws Exception {
        decodeMismatches(
                "q4_1 nibbles 256x512",
                Format.Q4_1,
                256,
                512,
                q41Weights(256, 512, 1L, true, false),
                false);
        decodeMismatches(
                "q4_1 nibbles 128x32",
                Format.Q4_1,
                128,
                32,
                q41Weights(128, 32, 2L, true, false),
                false);
        decodeMismatches(
                "q4_1 varied 384x1024",
                Format.Q4_1,
                384,
                1024,
                q41Weights(384, 1024, 3L, false, true),
                false);
    }

    @Test
    public void everyQ5KNibbleHighBitAndBothSignsDecodeToTheRetainedBits() throws Exception {
        decodeMismatches(
                "q5_k nibbles 256x512",
                Format.Q5_K,
                256,
                512,
                q5kWeights(256, 512, 1L, true, false),
                false);
        decodeMismatches(
                "q5_k nibbles 128x256",
                Format.Q5_K,
                128,
                256,
                q5kWeights(128, 256, 2L, true, false),
                false);
        decodeMismatches(
                "q5_k varied 384x1024",
                Format.Q5_K,
                384,
                1024,
                q5kWeights(384, 1024, 3L, false, true),
                false);
    }

    @Test
    public void theProductionShapesDecodeToTheRetainedBits() throws Exception {
        decodeMismatches(
                "q4_1 ffn_down",
                Format.Q4_1,
                5120,
                17408,
                q41Weights(5120, 17408, 11L, false, false),
                false);
        decodeMismatches(
                "q5_k ssm_out",
                Format.Q5_K,
                5120,
                6144,
                q5kWeights(5120, 6144, 12L, false, false),
                false);
    }

    @Test
    public void theBrokenCandidatesDiffer() throws Exception {
        assertTrue(
                "misplaced q4_1 high half agreed",
                decodeMismatches(
                                "q4_1 misplaced",
                                Format.Q4_1,
                                256,
                                512,
                                q41Weights(256, 512, 1L, true, false),
                                true)
                        > 0);
        assertTrue(
                "misplaced q5_k high half agreed",
                decodeMismatches(
                                "q5_k misplaced",
                                Format.Q5_K,
                                256,
                                512,
                                q5kWeights(256, 512, 1L, true, false),
                                true)
                        > 0);
    }

    // ---- complete pair -------------------------------------------------------------------------

    private static TaskGraph pair(
            String name,
            Format f,
            boolean tiled,
            boolean everyExecution,
            HalfFloatArray a,
            ByteArray w,
            HalfFloatArray scratch,
            FloatArray out,
            int m,
            int n,
            int k) {
        int in =
                everyExecution
                        ? DataTransferMode.EVERY_EXECUTION
                        : DataTransferMode.FIRST_EXECUTION;
        int back =
                everyExecution ? DataTransferMode.EVERY_EXECUTION : DataTransferMode.UNDER_DEMAND;
        TaskGraph g = new TaskGraph(name).transferToDevice(in, a, w, scratch, out);
        if (tiled) {
            g.task(
                    "d",
                    f == Format.Q4_1
                            ? Qwen35MMAKernels::dequantizeQ4_1ToFP16TiledPairs
                            : Qwen35MMAKernels::dequantizeQ5_KToFP16TiledPairs,
                    new KernelContext(),
                    w,
                    scratch,
                    n,
                    k);
            g.task(
                    "g",
                    Qwen35MMAKernels::gemmMMATiledB,
                    new KernelContext(),
                    a,
                    scratch,
                    out,
                    m,
                    n,
                    k);
        } else {
            g.task(
                    "d",
                    f == Format.Q4_1
                            ? Qwen35ReferenceKernels::dequantizeQ4_1ToFP16
                            : Qwen35ReferenceKernels::dequantizeQ5_KToFP16,
                    new KernelContext(),
                    w,
                    scratch,
                    n,
                    k);
            g.task(
                    "g",
                    TransformerBatchPrefillKernels::gemmMMA,
                    new KernelContext(),
                    a,
                    scratch,
                    out,
                    m,
                    n,
                    k);
        }
        return g.transferToHost(back, out);
    }

    private static GridScheduler pairGrid(String name, boolean tiled, int m, int n, int k) {
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid(name + ".d", lanes(tiled ? (long) n * k / 2 : (long) n * k));
        s.addWorkerGrid(name + ".g", gemmGrid(m, n));
        return s;
    }

    private static HalfFloatArray activations(int m, int k, long seed) {
        HalfFloatArray a = new HalfFloatArray(m * k);
        Random rng = new Random(seed);
        for (int i = 0; i < m * k; i++) {
            a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
        }
        return a;
    }

    private static void assertPairParity(
            Format f, int m, int n, int k, byte[] raw, HalfFloatArray scratch) throws Exception {
        ByteArray w = toDevice(raw);
        HalfFloatArray a = activations(m, k, 5L + m);
        FloatArray o1 = new FloatArray(m * n);
        FloatArray o2 = new FloatArray(m * n);
        o1.init(Float.NaN);
        o2.init(Float.NaN);
        TaskGraph pc = pair("pc", f, false, true, a, w, scratch, o1, m, n, k);
        TaskGraph pk = pair("pk", f, true, true, a, w, scratch, o2, m, n, k);
        try (TornadoExecutionPlan p1 = new TornadoExecutionPlan(pc.snapshot());
                TornadoExecutionPlan p2 = new TornadoExecutionPlan(pk.snapshot())) {
            p1.withGridScheduler(pairGrid("pc", false, m, n, k)).execute();
            p2.withGridScheduler(pairGrid("pk", true, m, n, k)).execute();
        }
        int mismatches = 0;
        String first = null;
        for (int i = 0; i < m * n; i++) {
            int c = Float.floatToRawIntBits(o1.get(i));
            int d = Float.floatToRawIntBits(o2.get(i));
            assertTrue(f + " m=" + m + ": control not finite at " + i, !Float.isNaN(o1.get(i)));
            if (c != d) {
                if (first == null) {
                    first = "output " + i + ": control " + o1.get(i) + " candidate " + o2.get(i);
                }
                mismatches++;
            }
        }
        assertEquals(
                f + " m=" + m + ": " + mismatches + " outputs differ, first at " + first,
                0,
                mismatches);
    }

    @Test
    public void theCompletePairsAgreeRawBitAtTheProductionShapes() throws Exception {
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        byte[] q41 = q41Weights(5120, 17408, 21L, false, false);
        byte[] q5k = q5kWeights(5120, 6144, 22L, false, false);
        for (int m : new int[] {128, 512}) {
            assertPairParity(Format.Q4_1, m, 5120, 17408, q41, scratch);
            assertPairParity(Format.Q5_K, m, 5120, 6144, q5k, scratch);
        }
    }

    /**
     * Complete pair, control versus candidate, alternating; opt in with JLLM_KERNEL_SCREEN=true.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        for (Format f : Format.values()) {
            int n = 5120;
            int k = f == Format.Q4_1 ? 17408 : 6144;
            ByteArray w = toDevice(weights(f, n, k, 7L, false, false));
            for (int m : new int[] {128, 256, 512, 1024, 2048}) {
                HalfFloatArray a = activations(m, k, 9L + m);
                FloatArray o1 = new FloatArray(m * n);
                FloatArray o2 = new FloatArray(m * n);
                HalfFloatArray dOnly = new HalfFloatArray(SCRATCH_ELEMENTS);
                TaskGraph dc =
                        new TaskGraph("dc")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, dOnly);
                TaskGraph dk =
                        new TaskGraph("dk")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, dOnly);
                if (f == Format.Q4_1) {
                    dc.task(
                            "d",
                            Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
                            new KernelContext(),
                            w,
                            dOnly,
                            n,
                            k);
                    dk.task(
                            "d",
                            Qwen35MMAKernels::dequantizeQ4_1ToFP16TiledPairs,
                            new KernelContext(),
                            w,
                            dOnly,
                            n,
                            k);
                } else {
                    dc.task(
                            "d",
                            Qwen35ReferenceKernels::dequantizeQ5_KToFP16,
                            new KernelContext(),
                            w,
                            dOnly,
                            n,
                            k);
                    dk.task(
                            "d",
                            Qwen35MMAKernels::dequantizeQ5_KToFP16TiledPairs,
                            new KernelContext(),
                            w,
                            dOnly,
                            n,
                            k);
                }
                dc.transferToHost(DataTransferMode.UNDER_DEMAND, dOnly);
                dk.transferToHost(DataTransferMode.UNDER_DEMAND, dOnly);
                GridScheduler sdc = new GridScheduler();
                sdc.addWorkerGrid("dc.d", lanes((long) n * k));
                GridScheduler sdk = new GridScheduler();
                sdk.addWorkerGrid("dk.d", lanes((long) n * k / 2));
                TaskGraph pc = pair("pc", f, false, false, a, w, scratch, o1, m, n, k);
                TaskGraph pk = pair("pk", f, true, false, a, w, scratch, o2, m, n, k);
                try (TornadoExecutionPlan p1 = new TornadoExecutionPlan(dc.snapshot());
                        TornadoExecutionPlan p2 = new TornadoExecutionPlan(dk.snapshot());
                        TornadoExecutionPlan p3 = new TornadoExecutionPlan(pc.snapshot());
                        TornadoExecutionPlan p4 = new TornadoExecutionPlan(pk.snapshot())) {
                    p1.withGridScheduler(sdc).withProfiler(ProfilerMode.SILENT);
                    p2.withGridScheduler(sdk).withProfiler(ProfilerMode.SILENT);
                    p3.withGridScheduler(pairGrid("pc", false, m, n, k))
                            .withProfiler(ProfilerMode.SILENT);
                    p4.withGridScheduler(pairGrid("pk", true, m, n, k))
                            .withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        p1.execute();
                        p2.execute();
                        p3.execute();
                        p4.execute();
                    }
                    int samples = 15;
                    long[] t1 = new long[samples];
                    long[] t2 = new long[samples];
                    long[] t3 = new long[samples];
                    long[] t4 = new long[samples];
                    for (int i = 0; i < samples; i++) {
                        if ((i & 1) == 0) {
                            t1[i] = kernelNs(p1.execute());
                            t2[i] = kernelNs(p2.execute());
                            t3[i] = kernelNs(p3.execute());
                            t4[i] = kernelNs(p4.execute());
                        } else {
                            t4[i] = kernelNs(p4.execute());
                            t3[i] = kernelNs(p3.execute());
                            t2[i] = kernelNs(p2.execute());
                            t1[i] = kernelNs(p1.execute());
                        }
                    }
                    String tag = " " + f + " m=" + m;
                    report("decoder control " + tag, t1);
                    report("decoder tiled   " + tag, t2);
                    report("pair control    " + tag, t3);
                    report("pair tiled      " + tag, t4);
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
                "[screen] %-36s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }

    // ---- negative controls ---------------------------------------------------------------------

    /** The Q4_1 candidate with the high half one tile too far. */
    public static void q41PairsWithTheHighHalfMisplaced(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tile = (lane >>> 11) << 1;
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * 128 + (sub << 3) + (pairInSub << 1) + parity;
        int element = kStep * 16 + kk;
        int base = (row * (k / 32) + (element >> 5)) * 20;
        float scale = w.getHalfFloat(base).getFloat32();
        float minimum = w.getHalfFloat(base + 2).getFloat32();
        int packed = w.get(base + 4 + (element & 15)) & 0xFF;
        int low = packed & 0xF;
        int high = (packed & 0xF0) >>> 4;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        out.set(lowHalf, new HalfFloat(scale * low + minimum));
        out.set((lowHalf + 4096) % (n * k), new HalfFloat(scale * high + minimum));
    }

    /** The Q5_K candidate with the high half one tile away instead of two. */
    public static void q5kPairsWithTheHighHalfMisplaced(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tile = ((lane >>> 12) << 2) + ((lane >>> 11) & 1);
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * 128 + (sub << 3) + (pairInSub << 1) + parity;
        int element = kStep * 16 + kk;
        int superBlock = element >> 8;
        int inSuper = element & 255;
        int subBlock = inSuper >> 5;
        int posInSub = inSuper & 31;
        int base = (row * (k / 256) + superBlock) * 176;
        float d = w.getHalfFloat(base).getFloat32();
        float dmin = w.getHalfFloat(base + 2).getFloat32();
        int psl = scaleAndMinHost(w, base + 4, subBlock);
        int psh = scaleAndMinHost(w, base + 4, subBlock + 1);
        int pairIndex = subBlock >> 1;
        int qsByte = w.get(base + 48 + pairIndex * 32 + posInSub) & 0xFF;
        int qhByte = w.get(base + 16 + posInSub) & 0xFF;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        out.set(
                lowHalf,
                new HalfFloat(
                        d * (psl >> 8) * ((qsByte & 0xF) + ((qhByte >> (pairIndex * 2)) & 1) * 16)
                                - dmin * (psl & 0xFF)));
        out.set(
                lowHalf + 2048,
                new HalfFloat(
                        d
                                        * (psh >> 8)
                                        * (((qsByte & 0xF0) >>> 4)
                                                + ((qhByte >> (pairIndex * 2 + 1)) & 1) * 16)
                                - dmin * (psh & 0xFF)));
    }

    private static int scaleAndMinHost(ByteArray w, int scalesBase, int subBlock) {
        if (subBlock < 4) {
            return ((w.get(scalesBase + subBlock) & 63) << 8)
                    | (w.get(scalesBase + subBlock + 4) & 63);
        }
        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
        int sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
        int m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }
}
