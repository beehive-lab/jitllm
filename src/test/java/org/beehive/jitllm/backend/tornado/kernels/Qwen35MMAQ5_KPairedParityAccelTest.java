package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The paired-load {@code Q5_K} tensor-core projection against the byte-load kernel it was copied
 * from, on the device, over identical input bytes — and, independently, against the host tensor's
 * own dot product, since two kernels sharing a decode defect would agree with each other.
 *
 * <p>Bit parity: the candidate reads the same {@code qs} and {@code qh} bytes through 16-bit words
 * and feeds the same decode, so every output must carry the same raw bits. Outputs are NaN-poisoned
 * so an unwritten column fails as a non-finite value.
 *
 * <p>Fixtures: super-blocks with a finite {@code d} of either sign and a non-zero {@code dmin},
 * random six-bit scale bytes (so both {@code scaleAndMin} branches decode non-trivial scales and
 * minima), and packed planes that put every one of the 32 five-bit values at every position of
 * every sub-block — the structured plane walks the values, the random plane covers the rest.
 */
// @formatter:on
public class Qwen35MMAQ5_KPairedParityAccelTest {

    private static final int SUPER_BLOCK_BYTES = 176;
    private static final int QK_K = 256;

    /** Production ssm_out: 6144 inputs (24 super-blocks) to 5120 outputs. */
    private static final int PROD_N = 5120;

    private static final int PROD_K = 6144;

    private static byte[] weights(int n, int k, long seed, boolean structured) {
        int superPerRow = k / QK_K;
        byte[] raw = new byte[n * superPerRow * SUPER_BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int c = 0; c < n; c++) {
            for (int sb = 0; sb < superPerRow; sb++) {
                int base = (c * superPerRow + sb) * SUPER_BLOCK_BYTES;
                int b = c * superPerRow + sb;
                // d: about ±0.06, varied per block; dmin: about ±0.004, never zero.
                int dBits = 0x2C00 | (b & 0xFF);
                if ((b & 1) == 1) {
                    dBits |= 0x8000;
                }
                int dminBits = 0x1C00 | ((b * 7) & 0xFF);
                if ((b & 2) == 2) {
                    dminBits |= 0x8000;
                }
                raw[base] = (byte) dBits;
                raw[base + 1] = (byte) (dBits >> 8);
                raw[base + 2] = (byte) dminBits;
                raw[base + 3] = (byte) (dminBits >> 8);
                if (structured) {
                    // qs byte at position p of pair q: low nibble (p + q + c), high nibble
                    // (15 - p + c); qh bit s of position p: set when (p + s + c) is odd, so every
                    // sub-block sees both high-bit values at every position.
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

    private static HalfFloatArray activations(int m, int k, long seed) {
        Random rng = new Random(seed);
        HalfFloatArray a = new HalfFloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
        }
        return a;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static WorkerGrid grid(int m, int n) {
        WorkerGrid worker =
                new WorkerGrid1D(
                        (m / Qwen35MMAKernels.BM)
                                * (n / Qwen35MMAKernels.BN)
                                * Qwen35MMAKernels.LOCAL);
        worker.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
        return worker;
    }

    /** Both kernels over the same bytes; raw-bit equality; then the candidate against the host. */
    private static void assertParity(String what, int m, int n, int k, byte[] raw, long seed)
            throws Exception {
        HalfFloatArray a = activations(m, k, seed);
        ByteArray w = toDevice(raw);
        FloatArray original = new FloatArray(m * n);
        FloatArray candidate = new FloatArray(m * n);
        original.init(Float.NaN);
        candidate.init(Float.NaN);

        TaskGraph graph =
                new TaskGraph("q5k")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, a, w, original, candidate)
                        .task(
                                "original",
                                Qwen35ReferenceKernels::projectionMMAQ5_K,
                                new KernelContext(),
                                a,
                                w,
                                original,
                                m,
                                n,
                                k)
                        .task(
                                "paired",
                                Qwen35MMAKernels::projectionMMAQ5_KPaired,
                                new KernelContext(),
                                a,
                                w,
                                candidate,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, original, candidate);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("q5k.original", grid(m, n));
        scheduler.addWorkerGrid("q5k.paired", grid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int mismatches = 0;
        String first = null;
        for (int i = 0; i < m * n; i++) {
            float o = original.get(i);
            float c = candidate.get(i);
            assertTrue(what + ": original not finite at " + where(i, n), Float.isFinite(o));
            assertTrue(what + ": candidate not finite at " + where(i, n), Float.isFinite(c));
            if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(c)) {
                if (first == null) {
                    first = where(i, n) + ": original " + o + " candidate " + c;
                }
                mismatches++;
            }
        }
        assertEquals(
                what + ": " + mismatches + " outputs differ, first at " + first, 0, mismatches);

        // Independent host reference for the candidate, with the tolerance the host-reference
        // projection test uses: sixteen-bit multiplicands against FP32 ones.
        float[] expected = new float[m * n];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor host = new Q5_KFloatTensor(n * k, segment);
            for (int r = 0; r < m; r++) {
                for (int col = 0; col < n; col++) {
                    float sum = 0f;
                    for (int i = 0; i < k; i++) {
                        sum += host.getFloat(col * k + i) * a.get(r * k + i).getFloat32();
                    }
                    expected[r * n + col] = sum;
                }
            }
        }
        double largest = 0;
        double worst = 0;
        int worstAt = 0;
        for (int i = 0; i < m * n; i++) {
            largest = Math.max(largest, Math.abs(expected[i]));
            double err = Math.abs(expected[i] - candidate.get(i));
            if (err > worst) {
                worst = err;
                worstAt = i;
            }
        }
        assertTrue(
                what
                        + ": candidate vs host worst error "
                        + worst
                        + " at "
                        + where(worstAt, n)
                        + " (expected "
                        + expected[worstAt]
                        + ", got "
                        + candidate.get(worstAt)
                        + "), largest "
                        + largest,
                worst < 0.01 * largest);
    }

    private static String where(int i, int n) {
        return "row " + (i / n) + " col " + (i % n);
    }

    /** Every five-bit value at every position of every sub-block, both scale branches. */
    @Test
    public void everyValueAtEveryPositionAgreesBitForBit() throws Exception {
        assertParity("structured", 32, 256, 512, weights(256, 512, 1L, true), 11L);
    }

    /** Random planes over several super-blocks per row, one to four row tiles. */
    @Test
    public void randomPlanesAcrossSuperBlocksAgreeBitForBit() throws Exception {
        byte[] raw = weights(512, 1024, 2L, false);
        for (int m : new int[] {16, 32, 64}) {
            assertParity("random m=" + m, m, 512, 1024, raw, 20L + m);
        }
    }

    /** The production ssm_out shape, at the production widths. */
    @Test
    public void theProductionShapeAgreesBitForBit() throws Exception {
        byte[] raw = weights(PROD_N, PROD_K, 3L, false);
        for (int m : new int[] {32, 256}) {
            assertParity("ssm_out m=" + m, m, PROD_N, PROD_K, raw, 30L + m);
        }
    }

    /**
     * Kernel time of both forms on the production shape at each width, from the TornadoVM profiler,
     * alternating order after a warm-up. Opt-in; prints distributions only.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        ByteArray w = toDevice(weights(PROD_N, PROD_K, 5L, false));
        for (int m : new int[] {32, 64, 128, 256}) {
            HalfFloatArray a = activations(m, PROD_K, 40L + m);
            FloatArray outA = new FloatArray(m * PROD_N);
            FloatArray outB = new FloatArray(m * PROD_N);
            TaskGraph ga =
                    new TaskGraph("orig")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                            .task(
                                    "p",
                                    Qwen35ReferenceKernels::projectionMMAQ5_K,
                                    new KernelContext(),
                                    a,
                                    w,
                                    outA,
                                    m,
                                    PROD_N,
                                    PROD_K)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
            TaskGraph gb =
                    new TaskGraph("pair")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                            .task(
                                    "p",
                                    Qwen35MMAKernels::projectionMMAQ5_KPaired,
                                    new KernelContext(),
                                    a,
                                    w,
                                    outB,
                                    m,
                                    PROD_N,
                                    PROD_K)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("orig.p", grid(m, PROD_N));
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("pair.p", grid(m, PROD_N));
            try (TornadoExecutionPlan planA = new TornadoExecutionPlan(ga.snapshot());
                    TornadoExecutionPlan planB = new TornadoExecutionPlan(gb.snapshot())) {
                planA.withGridScheduler(sa).withProfiler(ProfilerMode.SILENT);
                planB.withGridScheduler(sb).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 5; i++) {
                    planA.execute();
                    planB.execute();
                }
                int samples = 25;
                long[] tA = new long[samples];
                long[] tB = new long[samples];
                for (int i = 0; i < samples; i++) {
                    if ((i & 1) == 0) {
                        tA[i] = kernelNs(planA.execute());
                        tB[i] = kernelNs(planB.execute());
                    } else {
                        tB[i] = kernelNs(planB.execute());
                        tA[i] = kernelNs(planA.execute());
                    }
                }
                report("original ssm_out m=" + m, tA);
                report("paired   ssm_out m=" + m, tB);
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
                "[screen] %-26s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
