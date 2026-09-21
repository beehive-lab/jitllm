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
 * The Q4_0 attention key/value projection shape, N = 1024 outputs over K = 5120, on the two
 * existing paths: the direct kernel {@code projectionMMAQ4_0Prefetch} (the current dispatch, the
 * pair being excluded below 5120 outputs) against the paired-nibble decoder + {@code
 * gemmMMATiledB}. Raw-bit parity over NaN-poisoned outputs with distinct activation rows at widths
 * 128 to 2048, and an opt-in screen of the complete pair against the direct kernel.
 */
// @formatter:on
public class Qwen35Q4_0KvProjectionDispatchAccelTest {

    private static final int N = 1024;
    private static final int K = 5120;
    private static final int BLOCK_BYTES = 18;

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

    private static WorkerGrid prefetchGrid(int m, int n) {
        WorkerGrid g =
                new WorkerGrid1D(
                        (m / Qwen35MMAKernels.BM)
                                * (n / Qwen35MMAKernels.BN)
                                * Qwen35MMAKernels.LOCAL);
        g.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid dequantGrid(int n, int k) {
        WorkerGrid g = new WorkerGrid1D(n * k / 2);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static void assertParity(int m, long seed) throws Exception {
        HalfFloatArray a = activations(m, K, seed);
        ByteArray w = toDevice(randomWeights(N, K, seed + 1));
        HalfFloatArray scratch = new HalfFloatArray(N * K);
        FloatArray control = new FloatArray(m * N);
        FloatArray candidate = new FloatArray(m * N);
        control.init(Float.NaN);
        candidate.init(Float.NaN);
        scratch.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("kv")
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
                                N,
                                K)
                        .task(
                                "dequant",
                                Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                new KernelContext(),
                                w,
                                scratch,
                                N,
                                K)
                        .task(
                                "gemm",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a,
                                scratch,
                                candidate,
                                m,
                                N,
                                K)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("kv.control", prefetchGrid(m, N));
        s.addWorkerGrid("kv.dequant", dequantGrid(N, K));
        s.addWorkerGrid("kv.gemm", gemmGrid(m, N));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int count = 0;
        for (int i = 0; i < m * N; i++) {
            assertTrue("m=" + m + ": control not finite at " + i, Float.isFinite(control.get(i)));
            assertTrue(
                    "m=" + m + ": candidate not finite at " + i, Float.isFinite(candidate.get(i)));
            if (Float.floatToRawIntBits(control.get(i))
                    != Float.floatToRawIntBits(candidate.get(i))) {
                count++;
            }
        }
        assertEquals("m=" + m + ": outputs differing", 0, count);
    }

    @Test
    public void thePairAgreesWithTheDirectKernelAtEveryWidth() throws Exception {
        for (int m : new int[] {128, 256, 512, 1024, 2048}) {
            assertParity(m, 10L + m);
        }
    }

    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        ByteArray w = toDevice(randomWeights(N, K, 7L));
        HalfFloatArray scratch = new HalfFloatArray(N * K);
        for (int m : new int[] {128, 256, 512, 1024, 2048}) {
            HalfFloatArray a = activations(m, K, 9L + m);
            FloatArray o1 = new FloatArray(m * N);
            FloatArray o2 = new FloatArray(m * N);
            TaskGraph ctl =
                    new TaskGraph("ctl")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                            .task(
                                    "p",
                                    Qwen35MMAKernels::projectionMMAQ4_0Prefetch,
                                    new KernelContext(),
                                    a,
                                    w,
                                    o1,
                                    m,
                                    N,
                                    K)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, o1);
            TaskGraph pair =
                    new TaskGraph("pair")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                            .task(
                                    "d",
                                    Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                    new KernelContext(),
                                    w,
                                    scratch,
                                    N,
                                    K)
                            .task(
                                    "g",
                                    Qwen35MMAKernels::gemmMMATiledB,
                                    new KernelContext(),
                                    a,
                                    scratch,
                                    o2,
                                    m,
                                    N,
                                    K)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, o2);
            GridScheduler sc = new GridScheduler();
            sc.addWorkerGrid("ctl.p", prefetchGrid(m, N));
            GridScheduler sp = new GridScheduler();
            sp.addWorkerGrid("pair.d", dequantGrid(N, K));
            sp.addWorkerGrid("pair.g", gemmGrid(m, N));
            try (TornadoExecutionPlan pc = new TornadoExecutionPlan(ctl.snapshot());
                    TornadoExecutionPlan pp = new TornadoExecutionPlan(pair.snapshot())) {
                pc.withGridScheduler(sc).withProfiler(ProfilerMode.SILENT);
                pp.withGridScheduler(sp).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 5; i++) {
                    pc.execute();
                    pp.execute();
                }
                int samples = 15;
                long[] tc = new long[samples];
                long[] tp = new long[samples];
                for (int i = 0; i < samples; i++) {
                    if ((i & 1) == 0) {
                        tc[i] = kernelNs(pc.execute());
                        tp[i] = kernelNs(pp.execute());
                    } else {
                        tp[i] = kernelNs(pp.execute());
                        tc[i] = kernelNs(pc.execute());
                    }
                }
                report("direct prefetch  m=" + m, tc);
                report("decoder+gemm     m=" + m, tp);
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
                "[screen] %-24s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
