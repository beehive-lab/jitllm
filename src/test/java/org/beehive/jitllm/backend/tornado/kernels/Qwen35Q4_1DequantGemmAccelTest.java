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
 * Experiment: a whole {@code Q4_1} matrix dequantized to FP16 scratch on the device, then the
 * existing tiled tensor-core GEMM, against the direct quantized projection.
 *
 * <p>Three checks. The decoded halves against the decode expression evaluated on the host over
 * every nibble in both halves of the block, varied finite scales and minima of both signs, and
 * block and row boundaries — bit for bit, recording whether the device contracted the multiply-add
 * into one rounding or two. The projection through the two-kernel path against {@code
 * projectionMMAQ4_1}, raw-bit equal over NaN-poisoned outputs. And an opt-in timing screen in which
 * every candidate sample runs the dequantization and the GEMM on the production shape.
 */
// @formatter:on
public class Qwen35Q4_1DequantGemmAccelTest {

    /** One reusable FP16 scratch, the size the production state allocates (17408 x 5120 halves). */
    private static final int SCRATCH_ELEMENTS = 17408 * 5120;

    private static final int BLOCK_BYTES = 20;

    /** Production ffn_down on the Q4_1 blocks: 17408 inputs (544 blocks) to 5120 outputs. */
    private static final int PROD_N = 5120;

    private static final int PROD_K = 17408;

    /**
     * Q4_1 blocks: a finite scale of either sign (about ±0.06) and a non-zero minimum of either
     * sign (about ±0.004), both varied per block; nibbles random, or structured so every value
     * appears in both halves.
     */
    private static byte[] weights(int n, int k, long seed, boolean structured) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int c = 0; c < n; c++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int b = c * blocksPerRow + blk;
                int base = b * BLOCK_BYTES;
                int dBits = 0x2C00 | (b & 0xFF);
                if ((b & 1) == 1) {
                    dBits |= 0x8000;
                }
                int mBits = 0x1C00 | ((b * 7) & 0xFF);
                if ((b & 2) == 2) {
                    mBits |= 0x8000;
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

    /**
     * The kernel's decode expression on the host: two roundings, or one with the FMA contraction.
     */
    private static short expectedHalf(byte[] raw, int n, int k, int row, int element, boolean fma) {
        int blocksPerRow = k / 32;
        int block = element / 32;
        int within = element % 32;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale =
                new HalfFloat((short) ((raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8)))
                        .getFloat32();
        float minimum =
                new HalfFloat((short) ((raw[base + 2] & 0xFF) | ((raw[base + 3] & 0xFF) << 8)))
                        .getFloat32();
        int packed = raw[base + 4 + (within & 15)] & 0xFF;
        int q = within >= 16 ? (packed >> 4) & 0xF : packed & 0xF;
        float value = fma ? Math.fma(scale, (float) q, minimum) : scale * q + minimum;
        return new HalfFloat(value).getHalfFloatValue();
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

    private static void assertDecodedBits(String what, int n, int k, byte[] raw) throws Exception {
        ByteArray w = toDevice(raw);
        HalfFloatArray out = new HalfFloatArray(n * k);
        out.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("dq")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, out)
                        .task(
                                "dequant",
                                Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
                                new KernelContext(),
                                w,
                                out,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("dq.dequant", dequantGrid(n, k));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        int contracted = 0;
        int separate = 0;
        String first = null;
        for (int row = 0; row < n; row++) {
            for (int e = 0; e < k; e++) {
                short got = out.get(row * k + e).getHalfFloatValue();
                short wantSeparate = expectedHalf(raw, n, k, row, e, false);
                short wantFma = expectedHalf(raw, n, k, row, e, true);
                if (got == wantSeparate) {
                    separate++;
                }
                if (got == wantFma) {
                    contracted++;
                }
                short want = wantSeparate;
                if (got != wantSeparate && got != wantFma) {
                    if (first == null) {
                        first =
                                "row "
                                        + row
                                        + " element "
                                        + e
                                        + " (block "
                                        + (e / 32)
                                        + ", within "
                                        + (e % 32)
                                        + "): got 0x"
                                        + Integer.toHexString(got & 0xFFFF)
                                        + " want 0x"
                                        + Integer.toHexString(want & 0xFFFF);
                    }
                    mismatches++;
                }
            }
        }
        assertEquals(what + ": " + mismatches + " halves differ, first at " + first, 0, mismatches);
        // Which contraction the device used, for the record: every half matched at least one
        // form; where the two forms differ, the count says which the compiler emitted.
        System.out.printf(
                Locale.ROOT,
                "[decode] %s: %d halves; %d match the two-rounding form, %d the fused form%n",
                what,
                n * k,
                separate,
                contracted);
    }

    private static void assertProjectionParity(
            String what, int m, int n, int k, byte[] raw, HalfFloatArray scratch, long seed)
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
                                Qwen35MMAKernels::projectionMMAQ4_1,
                                new KernelContext(),
                                a,
                                w,
                                control,
                                m,
                                n,
                                k)
                        .task(
                                "dequant",
                                Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
                                new KernelContext(),
                                w,
                                scratch,
                                n,
                                k)
                        .task(
                                "gemm",
                                TransformerBatchPrefillKernels::gemmMMA,
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
        assertEquals(
                what + ": " + mismatches + " outputs differ, first at " + first, 0, mismatches);
    }

    /** Every nibble in both halves, both signs of scale and minimum, block and row boundaries. */
    @Test
    public void theDecodedHalvesMatchTheDecodeExpression() throws Exception {
        assertDecodedBits("structured", 256, 512, weights(256, 512, 1L, true));
        assertDecodedBits("random", 384, 1024, weights(384, 1024, 3L, false));
    }

    /** The two-kernel projection against the direct kernel on the production shape. */
    @Test
    public void theTwoKernelProjectionAgreesBitForBit() throws Exception {
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        assertProjectionParity(
                "structured m=128", 128, 256, 512, weights(256, 512, 5L, true), scratch, 1L);
        byte[] raw = weights(PROD_N, PROD_K, 100L, false);
        for (int m : new int[] {128, 256, 512, 1024}) {
            assertProjectionParity("ffn_down m=" + m, m, PROD_N, PROD_K, raw, scratch, 7L + m);
        }
    }

    /**
     * Timing screen: control versus dequantization-only versus dequantization + GEMM, each a full
     * plan execution (device kernel time from the profiler and synchronized wall time), alternating
     * after a warm-up. Every candidate sample re-runs the dequantization.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        System.out.printf(
                Locale.ROOT,
                "[screen] scratch %d halves = %.1f MiB%n",
                SCRATCH_ELEMENTS,
                SCRATCH_ELEMENTS * 2.0 / (1 << 20));
        int[][] shapes = {{PROD_N, PROD_K}};
        for (int m : new int[] {128, 256, 512, 1024}) {
            for (int[] shape : shapes) {
                int n = shape[0];
                int k = shape[1];
                ByteArray w = toDevice(weights(n, k, 7L + n, false));
                HalfFloatArray a = activations(m, k, 9L + m);
                FloatArray outA = new FloatArray(m * n);
                FloatArray outB = new FloatArray(m * n);

                TaskGraph control =
                        new TaskGraph("ctl")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                                .task(
                                        "p",
                                        Qwen35MMAKernels::projectionMMAQ4_1,
                                        new KernelContext(),
                                        a,
                                        w,
                                        outA,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
                TaskGraph dequantOnly =
                        new TaskGraph("dqo")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, scratch);
                TaskGraph combined =
                        new TaskGraph("cmb")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_1ToFP16,
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
                GridScheduler sc = new GridScheduler();
                sc.addWorkerGrid("ctl.p", prefetchGrid(m, n));
                GridScheduler sd = new GridScheduler();
                sd.addWorkerGrid("dqo.d", dequantGrid(n, k));
                GridScheduler sm = new GridScheduler();
                sm.addWorkerGrid("cmb.d", dequantGrid(n, k));
                sm.addWorkerGrid("cmb.g", gemmGrid(m, n));
                try (TornadoExecutionPlan pc = new TornadoExecutionPlan(control.snapshot());
                        TornadoExecutionPlan pd = new TornadoExecutionPlan(dequantOnly.snapshot());
                        TornadoExecutionPlan pm = new TornadoExecutionPlan(combined.snapshot())) {
                    pc.withGridScheduler(sc).withProfiler(ProfilerMode.SILENT);
                    pd.withGridScheduler(sd).withProfiler(ProfilerMode.SILENT);
                    pm.withGridScheduler(sm).withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        pc.execute();
                        pd.execute();
                        pm.execute();
                    }
                    int samples = 15;
                    long[] kc = new long[samples];
                    long[] kd = new long[samples];
                    long[] km = new long[samples];
                    long[] wc = new long[samples];
                    long[] wm = new long[samples];
                    for (int i = 0; i < samples; i++) {
                        if ((i & 1) == 0) {
                            kc[i] = timed(pc, wc, i);
                            kd[i] = kernelNs(pd.execute());
                            km[i] = timed(pm, wm, i);
                        } else {
                            km[i] = timed(pm, wm, i);
                            kd[i] = kernelNs(pd.execute());
                            kc[i] = timed(pc, wc, i);
                        }
                    }
                    String tag = " n=" + n + " k=" + k + " m=" + m;
                    report("control  kernel" + tag, kc);
                    report("control  wall  " + tag, wc);
                    report("dequant  kernel" + tag, kd);
                    report("dq+gemm  kernel" + tag, km);
                    report("dq+gemm  wall  " + tag, wm);
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
                "[screen] %-40s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
