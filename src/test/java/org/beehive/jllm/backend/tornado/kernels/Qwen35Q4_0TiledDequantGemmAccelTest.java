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
 * Experiment: a whole {@code Q4_0} matrix decoded straight into the order the tiled GEMM stages its
 * B tile, consumed by a GEMM whose B staging is a contiguous four-byte copy, against the row-major
 * decode plus the general GEMM, and against the direct quantized projection.
 *
 * <p>Four checks. The layout: the host maps every element through the documented forward index and
 * finds its decoded half there, bit for bit, and the documented inverse returns every half position
 * to its element. The decoded bits: identical to the row-major decoder's, element for element. The
 * projection: raw-bit equal to the direct kernel over NaN-poisoned outputs at the production
 * shapes; and the same loader fed the row-major scratch (a producer whose layout the loader does
 * not agree with) fails, so agreement is the layout's doing. And an opt-in timing screen of decode
 * and GEMM separately and together.
 */
// @formatter:on
public class Qwen35Q4_0TiledDequantGemmAccelTest {

    private static final int BLOCK_BYTES = 18;

    /** One reusable FP16 scratch for the largest production matrix (17408 x 5120 halves). */
    private static final int SCRATCH_ELEMENTS = 17408 * 5120;

    private static byte[] randomWeights(int n, int k, long seed) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * BLOCK_BYTES;
            int bits = 0x2C00 | (b & 0xFF);
            if ((b & 1) == 1) {
                bits |= 0x8000;
            }
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static byte[] everyNibbleWeights(int n, int k) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        for (int c = 0; c < n; c++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int base = (c * blocksPerRow + blk) * BLOCK_BYTES;
                int bits = 0x2C00 | ((c + blk) & 0xFF);
                if (((c + blk) & 1) == 1) {
                    bits |= 0x8000;
                }
                raw[base] = (byte) (bits & 0xFF);
                raw[base + 1] = (byte) (bits >> 8);
                for (int b = 0; b < 16; b++) {
                    int lo = (b + c + blk) & 0xF;
                    int hi = (15 - b + c) & 0xF;
                    raw[base + 2 + b] = (byte) ((hi << 4) | lo);
                }
            }
        }
        return raw;
    }

    private static short expectedHalf(byte[] raw, int k, int row, int element) {
        int blocksPerRow = k / 32;
        int block = element / 32;
        int within = element % 32;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        int scaleBits = (raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8);
        float scale = new HalfFloat((short) scaleBits).getFloat32();
        int packed = raw[base + 2 + (within & 15)] & 0xFF;
        int q = within >= 16 ? (packed >> 4) & 0xF : packed & 0xF;
        return new HalfFloat(scale * (q - 8)).getHalfFloatValue();
    }

    /** The documented forward index: half position of {@code (row, element)}. */
    static int tiledIndex(int k, int row, int element) {
        int kSteps = k / 16;
        int tile = (row / 128) * kSteps + element / 16;
        int r = row % 128;
        int kk = element % 16;
        int idx = (r / 8) * 64 + kk * 4 + (r % 8) / 2;
        return tile * 2048 + idx * 2 + (row & 1);
    }

    /** The documented inverse: {@code {row, element}} of half position {@code h}. */
    static int[] tiledElement(int k, int h) {
        int kSteps = k / 16;
        int idx = (h >>> 1) & 1023;
        int tile = h >>> 11;
        int row = (tile / kSteps) * 128 + ((idx >>> 6) << 3) + ((idx & 3) << 1) + (h & 1);
        int kk = (tile % kSteps) * 16 + ((idx & 63) >>> 2);
        return new int[] {row, kk};
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static HalfFloatArray activations(int m, int k, long seed) {
        Random rng = new Random(seed);
        HalfFloatArray a = new HalfFloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
        }
        return a;
    }

    private static WorkerGrid dequantGrid(int n, int k) {
        WorkerGrid g = new WorkerGrid1D(n * k);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid prefetchGrid(int m, int n) {
        WorkerGrid g =
                new WorkerGrid1D(
                        (m / Qwen35MMAKernels.BM)
                                * (n / Qwen35MMAKernels.BN)
                                * Qwen35MMAKernels.LOCAL);
        g.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
        return g;
    }

    /** The index and its inverse are a bijection on every half position of the matrix. */
    @Test
    public void theIndexAndItsInverseAgree() {
        for (int[] shape : new int[][] {{128, 32}, {256, 512}, {384, 1024}}) {
            int n = shape[0];
            int k = shape[1];
            boolean[] seen = new boolean[n * k];
            for (int row = 0; row < n; row++) {
                for (int e = 0; e < k; e++) {
                    int h = tiledIndex(k, row, e);
                    assertTrue("index in range", h >= 0 && h < n * k);
                    assertTrue("index unique", !seen[h]);
                    seen[h] = true;
                    int[] back = tiledElement(k, h);
                    assertEquals("row of " + h, row, back[0]);
                    assertEquals("element of " + h, e, back[1]);
                }
            }
        }
    }

    private static HalfFloatArray decodeTiled(int n, int k, byte[] raw) throws Exception {
        ByteArray w = toDevice(raw);
        HalfFloatArray out = new HalfFloatArray(n * k);
        out.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("dqt")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, out)
                        .task(
                                "dequant",
                                Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                new KernelContext(),
                                w,
                                out,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("dqt.dequant", dequantGrid(n, k));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        return out;
    }

    private static void assertLayoutAndBits(String what, int n, int k, byte[] raw)
            throws Exception {
        HalfFloatArray out = decodeTiled(n, k, raw);
        int mismatches = 0;
        String first = null;
        for (int row = 0; row < n; row++) {
            for (int e = 0; e < k; e++) {
                short got = out.get(tiledIndex(k, row, e)).getHalfFloatValue();
                short want = expectedHalf(raw, k, row, e);
                if (got != want) {
                    if (first == null) {
                        first =
                                "row "
                                        + row
                                        + " element "
                                        + e
                                        + " at half "
                                        + tiledIndex(k, row, e)
                                        + ": got 0x"
                                        + Integer.toHexString(got & 0xFFFF)
                                        + " want 0x"
                                        + Integer.toHexString(want & 0xFFFF);
                    }
                    mismatches++;
                }
            }
        }
        assertEquals(what + ": " + mismatches + " halves differ, first at " + first, 0, mismatches);
    }

    /** Every element found at its documented position with the decode expression's bits. */
    @Test
    public void theDecodedHalvesSitAtTheDocumentedIndexWithTheDecodeExpressionsBits()
            throws Exception {
        assertLayoutAndBits("nibbles", 256, 512, everyNibbleWeights(256, 512));
        assertLayoutAndBits("random", 384, 1024, randomWeights(384, 1024, 3L));
    }

    /** The tiled decoder and the row-major decoder produce the same bits for every element. */
    @Test
    public void theTiledDecoderMatchesTheRowMajorDecoderElementForElement() throws Exception {
        int n = 384;
        int k = 1024;
        byte[] raw = randomWeights(n, k, 5L);
        ByteArray w = toDevice(raw);
        HalfFloatArray rowMajor = new HalfFloatArray(n * k);
        HalfFloatArray tiled = new HalfFloatArray(n * k);
        rowMajor.init(new HalfFloat(Float.NaN));
        tiled.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("both")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, rowMajor, tiled)
                        .task(
                                "rm",
                                Qwen35ReferenceKernels::dequantizeQ4_0ToFP16,
                                new KernelContext(),
                                w,
                                rowMajor,
                                n,
                                k)
                        .task(
                                "t",
                                Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                new KernelContext(),
                                w,
                                tiled,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, rowMajor, tiled);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("both.rm", dequantGrid(n, k));
        s.addWorkerGrid("both.t", dequantGrid(n, k));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        for (int row = 0; row < n; row++) {
            for (int e = 0; e < k; e++) {
                short a = rowMajor.get(row * k + e).getHalfFloatValue();
                short b = tiled.get(tiledIndex(k, row, e)).getHalfFloatValue();
                assertTrue("row-major finite at " + row + "," + e, (a & 0x7C00) != 0x7C00);
                if (a != b) {
                    mismatches++;
                }
            }
        }
        assertEquals(mismatches + " elements differ between the decoders", 0, mismatches);
    }

    /**
     * Control: the direct quantized kernel. Candidate: tiled decode + tiled-B GEMM. When {@code
     * mismatchedProducer} is set, the candidate GEMM instead reads the row-major decode, which it
     * must not agree with.
     */
    private static int projectionMismatches(
            String what,
            int m,
            int n,
            int k,
            byte[] raw,
            HalfFloatArray scratch,
            long seed,
            boolean mismatchedProducer)
            throws Exception {
        HalfFloatArray a = activations(m, k, seed);
        ByteArray w = toDevice(raw);
        FloatArray control = new FloatArray(m * n);
        FloatArray candidate = new FloatArray(m * n);
        control.init(Float.NaN);
        candidate.init(Float.NaN);
        scratch.init(new HalfFloat(Float.NaN));

        TaskGraph graph =
                new TaskGraph("proj")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, a, w, control, candidate, scratch)
                        .task(
                                "control",
                                Qwen35MMAKernels::projectionMMAQ4_0Prefetch,
                                new KernelContext(),
                                a,
                                w,
                                control,
                                m,
                                n,
                                k)
                        .task(
                                "dequant",
                                mismatchedProducer
                                        ? Qwen35ReferenceKernels::dequantizeQ4_0ToFP16
                                        : Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                new KernelContext(),
                                w,
                                scratch,
                                n,
                                k)
                        .task(
                                "gemm",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a,
                                scratch,
                                candidate,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("proj.control", prefetchGrid(m, n));
        s.addWorkerGrid("proj.dequant", dequantGrid(n, k));
        s.addWorkerGrid("proj.gemm", gemmGrid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        String first = null;
        for (int i = 0; i < m * n; i++) {
            float o = control.get(i);
            float c = candidate.get(i);
            assertTrue(
                    what + ": control not finite at " + (i / n) + "," + (i % n), Float.isFinite(o));
            assertTrue(
                    what + ": candidate not finite at " + (i / n) + "," + (i % n),
                    Float.isFinite(c));
            if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(c)) {
                if (first == null) {
                    first =
                            "row "
                                    + (i / n)
                                    + " col "
                                    + (i % n)
                                    + ": control "
                                    + o
                                    + " candidate "
                                    + c;
                }
                mismatches++;
            }
        }
        if (!mismatchedProducer) {
            assertEquals(
                    what + ": " + mismatches + " outputs differ, first at " + first, 0, mismatches);
        }
        return mismatches;
    }

    /** The tiled pair against the direct kernel at the production shapes and the target widths. */
    @Test
    public void theTiledPairAgreesBitForBitWithTheDirectKernel() throws Exception {
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        projectionMismatches(
                "nibbles m=128", 128, 256, 512, everyNibbleWeights(256, 512), scratch, 1L, false);
        int[][] shapes = {{17408, 5120}, {5120, 17408}};
        for (int[] shape : shapes) {
            int n = shape[0];
            int k = shape[1];
            byte[] raw = randomWeights(n, k, 100L + n);
            for (int m : new int[] {128, 512, 1024}) {
                projectionMismatches(
                        "n=" + n + " k=" + k + " m=" + m, m, n, k, raw, scratch, n + m, false);
            }
        }
    }

    /** The same loader over the row-major scratch disagrees: the agreement is the layout's. */
    @Test
    public void theLoaderFedTheRowMajorLayoutDisagrees() throws Exception {
        HalfFloatArray scratch = new HalfFloatArray(256 * 512);
        int mismatches =
                projectionMismatches(
                        "row-major producer",
                        128,
                        256,
                        512,
                        randomWeights(256, 512, 11L),
                        scratch,
                        2L,
                        true);
        assertTrue(
                "row-major producer under the tiled loader agreed on every output", mismatches > 0);
    }

    /**
     * Timing screen at the target widths: the direct kernel, the row-major decode alone, the
     * row-major pair, the tiled decode alone and the tiled pair, each a full plan execution (device
     * kernel time from the profiler and synchronized wall time), alternating after a warm-up.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        int[][] shapes = {{17408, 5120}, {5120, 17408}};
        for (int m : new int[] {512, 1024}) {
            for (int[] shape : shapes) {
                int n = shape[0];
                int k = shape[1];
                ByteArray w = toDevice(randomWeights(n, k, 7L + n));
                HalfFloatArray a = activations(m, k, 9L + m);
                FloatArray outA = new FloatArray(m * n);
                FloatArray outB = new FloatArray(m * n);
                FloatArray outC = new FloatArray(m * n);

                TaskGraph control =
                        new TaskGraph("ctl")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                                .task(
                                        "p",
                                        Qwen35MMAKernels::projectionMMAQ4_0Prefetch,
                                        new KernelContext(),
                                        a,
                                        w,
                                        outA,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
                TaskGraph rowDequant =
                        new TaskGraph("rdq")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, scratch);
                TaskGraph rowPair =
                        new TaskGraph("rpr")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        TransformerBatchPrefillKernels::gemmMMA,
                                        new KernelContext(),
                                        a,
                                        scratch,
                                        outB,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
                TaskGraph tiledDequant =
                        new TaskGraph("tdq")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, scratch);
                TaskGraph tiledPair =
                        new TaskGraph("tpr")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        Qwen35MMAKernels::gemmMMATiledB,
                                        new KernelContext(),
                                        a,
                                        scratch,
                                        outC,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, outC);
                GridScheduler sc = new GridScheduler();
                sc.addWorkerGrid("ctl.p", prefetchGrid(m, n));
                GridScheduler srd = new GridScheduler();
                srd.addWorkerGrid("rdq.d", dequantGrid(n, k));
                GridScheduler srp = new GridScheduler();
                srp.addWorkerGrid("rpr.d", dequantGrid(n, k));
                srp.addWorkerGrid("rpr.g", gemmGrid(m, n));
                GridScheduler std = new GridScheduler();
                std.addWorkerGrid("tdq.d", dequantGrid(n, k));
                GridScheduler stp = new GridScheduler();
                stp.addWorkerGrid("tpr.d", dequantGrid(n, k));
                stp.addWorkerGrid("tpr.g", gemmGrid(m, n));
                try (TornadoExecutionPlan pc = new TornadoExecutionPlan(control.snapshot());
                        TornadoExecutionPlan prd = new TornadoExecutionPlan(rowDequant.snapshot());
                        TornadoExecutionPlan prp = new TornadoExecutionPlan(rowPair.snapshot());
                        TornadoExecutionPlan ptd =
                                new TornadoExecutionPlan(tiledDequant.snapshot());
                        TornadoExecutionPlan ptp = new TornadoExecutionPlan(tiledPair.snapshot())) {
                    pc.withGridScheduler(sc).withProfiler(ProfilerMode.SILENT);
                    prd.withGridScheduler(srd).withProfiler(ProfilerMode.SILENT);
                    prp.withGridScheduler(srp).withProfiler(ProfilerMode.SILENT);
                    ptd.withGridScheduler(std).withProfiler(ProfilerMode.SILENT);
                    ptp.withGridScheduler(stp).withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        pc.execute();
                        prd.execute();
                        prp.execute();
                        ptd.execute();
                        ptp.execute();
                    }
                    int samples = 15;
                    long[] kc = new long[samples];
                    long[] krd = new long[samples];
                    long[] krp = new long[samples];
                    long[] ktd = new long[samples];
                    long[] ktp = new long[samples];
                    long[] wc = new long[samples];
                    long[] wrp = new long[samples];
                    long[] wtp = new long[samples];
                    for (int i = 0; i < samples; i++) {
                        if ((i & 1) == 0) {
                            kc[i] = timed(pc, wc, i);
                            krd[i] = kernelNs(prd.execute());
                            krp[i] = timed(prp, wrp, i);
                            ktd[i] = kernelNs(ptd.execute());
                            ktp[i] = timed(ptp, wtp, i);
                        } else {
                            ktp[i] = timed(ptp, wtp, i);
                            ktd[i] = kernelNs(ptd.execute());
                            krp[i] = timed(prp, wrp, i);
                            krd[i] = kernelNs(prd.execute());
                            kc[i] = timed(pc, wc, i);
                        }
                    }
                    String tag = " n=" + n + " k=" + k + " m=" + m;
                    report("direct     kernel" + tag, kc);
                    report("direct     wall  " + tag, wc);
                    report("row dq     kernel" + tag, krd);
                    report("row pair   kernel" + tag, krp);
                    report("row pair   wall  " + tag, wrp);
                    report("tiled dq   kernel" + tag, ktd);
                    report("tiled pair kernel" + tag, ktp);
                    report("tiled pair wall  " + tag, wtp);
                }
            }
        }
    }

    private static long timed(TornadoExecutionPlan plan, long[] wall, int i) {
        long t0 = System.nanoTime();
        TornadoExecutionResult r = plan.execute();
        wall[i] = System.nanoTime() - t0;
        return kernelNs(r);
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
                "[screen] %-42s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
