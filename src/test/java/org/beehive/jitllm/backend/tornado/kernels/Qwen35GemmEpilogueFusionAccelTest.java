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
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The epilogue-fused GEMMs — {@code gemmMMATiledBResidual} ({@code X += A x B}) and {@code
 * gemmMMATiledBSwiGLU} ({@code hb16 = fp16(silu(gate) * (A x B))}) — against {@code gemmMMATiledB}
 * followed by {@code residualAdd} / {@code swiGLUBatchFP16}: raw-bit equal on every element at the
 * production shapes and widths 128 and 512, NaN-poisoned destinations; the accumulator element
 * mapping is what makes them equal, so a mapping with the rows swapped fails. An opt-in screen
 * times the fused and two-kernel forms.
 */
public class Qwen35GemmEpilogueFusionAccelTest {

    private static HalfFloatArray halves(int n, long seed) {
        Random rng = new Random(seed);
        HalfFloatArray a = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, new HalfFloat(rng.nextFloat() * 2 - 1));
        }
        return a;
    }

    private static FloatArray floats(int n, long seed, float scale) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, (rng.nextFloat() * 2 - 1) * scale);
        }
        return a;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid lanes(int n) {
        WorkerGrid g = new WorkerGrid1D(n);
        g.setLocalWork(128, 1, 1);
        return g;
    }

    /** B in the tiled layout: produced by the production decoder from random Q4_0 bytes. */
    private static HalfFloatArray tiledB(int n, int k, long seed) throws Exception {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * 18];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * 18;
            int bits = 0x2C00 | (b & 0xFF) | (((b & 1) == 1) ? 0x8000 : 0);
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        uk.ac.manchester.tornado.api.types.arrays.ByteArray w =
                new uk.ac.manchester.tornado.api.types.arrays.ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        HalfFloatArray out = new HalfFloatArray(n * k);
        TaskGraph g =
                new TaskGraph("dq")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, out)
                        .task(
                                "d",
                                Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                new KernelContext(),
                                w,
                                out,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler s = new GridScheduler();
        WorkerGrid lg = new WorkerGrid1D(n * k / 2);
        lg.setLocalWork(256, 1, 1);
        s.addWorkerGrid("dq.d", lg);
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        return out;
    }

    private static int mismatches(FloatArray a, FloatArray b) {
        int n = 0;
        for (int i = 0; i < a.getSize(); i++) {
            if (Float.floatToRawIntBits(a.get(i)) != Float.floatToRawIntBits(b.get(i))) {
                n++;
            }
        }
        return n;
    }

    private static int mismatches(HalfFloatArray a, HalfFloatArray b) {
        int n = 0;
        for (int i = 0; i < a.getSize(); i++) {
            if (a.get(i).getHalfFloatValue() != b.get(i).getHalfFloatValue()) {
                n++;
            }
        }
        return n;
    }

    private void checkResidual(int m, int n, int k, long seed) throws Exception {
        HalfFloatArray a = halves(m * k, seed);
        HalfFloatArray b = tiledB(n, k, seed + 1);
        FloatArray x1 = floats(m * n, seed + 2, 4.0f);
        FloatArray x2 = floats(m * n, seed + 2, 4.0f);
        FloatArray c = new FloatArray(m * n);
        c.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("r")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, x1, x2, c)
                        .task(
                                "g",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a,
                                b,
                                c,
                                m,
                                n,
                                k)
                        .task("add", Qwen35MMAKernels::residualAdd, new KernelContext(), x1, c)
                        .task(
                                "f",
                                Qwen35MMAKernels::gemmMMATiledBResidual,
                                new KernelContext(),
                                a,
                                b,
                                x2,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, x1, x2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("r.g", gemmGrid(m, n));
        s.addWorkerGrid("r.add", lanes(m * n));
        s.addWorkerGrid("r.f", gemmGrid(m, n));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        for (int i = 0; i < m * n; i += 4097) {
            assertTrue("not finite", Float.isFinite(x1.get(i)));
        }
        assertEquals("residual m=" + m + " n=" + n, 0, mismatches(x1, x2));
    }

    private void checkSwiglu(int m, int n, int k, long seed) throws Exception {
        HalfFloatArray a = halves(m * k, seed);
        HalfFloatArray b = tiledB(n, k, seed + 1);
        FloatArray gate = floats(m * n, seed + 3, 30.0f);
        FloatArray up = new FloatArray(m * n);
        HalfFloatArray hb1 = new HalfFloatArray(m * n);
        HalfFloatArray hb2 = new HalfFloatArray(m * n);
        hb1.init(new HalfFloat(Float.NaN));
        hb2.init(new HalfFloat(Float.NaN));
        TaskGraph g =
                new TaskGraph("s")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, a, b, gate, up, hb1, hb2)
                        .task(
                                "g",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a,
                                b,
                                up,
                                m,
                                n,
                                k)
                        .task(
                                "sw",
                                Qwen35MMAKernels::swiGLUBatchFP16,
                                new KernelContext(),
                                gate,
                                up,
                                hb1)
                        .task(
                                "f",
                                Qwen35MMAKernels::gemmMMATiledBSwiGLU,
                                new KernelContext(),
                                a,
                                b,
                                gate,
                                hb2,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb1, hb2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("s.g", gemmGrid(m, n));
        s.addWorkerGrid("s.sw", lanes(m * n));
        s.addWorkerGrid("s.f", gemmGrid(m, n));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        int nonFinite = 0;
        for (int i = 0; i < m * n; i++) {
            if ((hb1.get(i).getHalfFloatValue() & 0x7C00) == 0x7C00) {
                nonFinite++;
            }
        }
        assertTrue("every half non-finite", nonFinite < m * n / 4);
        assertEquals("swiglu m=" + m + " n=" + n, 0, mismatches(hb1, hb2));
    }

    @Test
    public void theResidualEpilogueIsRawBitEqual() throws Exception {
        checkResidual(128, 256, 64, 1L);
        checkResidual(512, 5120, 6144, 2L);
        checkResidual(128, 5120, 17408, 3L);
    }

    @Test
    public void theSwigluEpilogueIsRawBitEqual() throws Exception {
        checkSwiglu(128, 256, 64, 4L);
        checkSwiglu(512, 17408, 5120, 5L);
    }

    /**
     * The negative control: the two-kernel result with rows {@code r} and {@code r + 8} of the
     * product exchanged inside each 16-row tile — what a fused epilogue that mapped accumulator
     * elements 0/1 to the wrong row would produce — must differ from the fused result.
     */
    @Test
    public void aSwappedRowMappingWouldDiffer() throws Exception {
        int m = 128, n = 256, k = 64;
        HalfFloatArray a = halves(m * k, 6L);
        HalfFloatArray b = tiledB(n, k, 7L);
        FloatArray x = floats(m * n, 8L, 4.0f);
        FloatArray x2 = floats(m * n, 8L, 4.0f);
        FloatArray c = new FloatArray(m * n);
        TaskGraph g =
                new TaskGraph("n")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, x2, c)
                        .task(
                                "g",
                                Qwen35MMAKernels::gemmMMATiledB,
                                new KernelContext(),
                                a,
                                b,
                                c,
                                m,
                                n,
                                k)
                        .task(
                                "f",
                                Qwen35MMAKernels::gemmMMATiledBResidual,
                                new KernelContext(),
                                a,
                                b,
                                x2,
                                m,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, c, x2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("n.g", gemmGrid(m, n));
        s.addWorkerGrid("n.f", gemmGrid(m, n));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        FloatArray swapped = new FloatArray(m * n);
        for (int r = 0; r < m; r++) {
            int rr = (r % 16 < 8) ? r + 8 : r - 8;
            for (int col = 0; col < n; col++) {
                swapped.set(r * n + col, x.get(r * n + col) + c.get(rr * n + col));
            }
        }
        assertTrue("the swapped mapping agreed", mismatches(swapped, x2) > 0);
        FloatArray straight = new FloatArray(m * n);
        for (int i = 0; i < m * n; i++) {
            straight.set(i, x.get(i) + c.get(i));
        }
        assertEquals("host residual", 0, mismatches(straight, x2));
    }

    /** Opt in with JLLM_KERNEL_SCREEN=true: fused vs two-kernel forms at the production shapes. */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        for (int m : new int[] {512, 2048}) {
            // residual: attn_output shape (5120 x 6144) and ffn_down (5120 x 17408)
            for (int[] nk : new int[][] {{5120, 6144}, {5120, 17408}}) {
                int n = nk[0], k = nk[1];
                HalfFloatArray a = halves(m * k, 1L);
                HalfFloatArray b = tiledB(n, k, 2L);
                FloatArray x = floats(m * n, 3L, 1.0f);
                FloatArray c = new FloatArray(m * n);
                TaskGraph two =
                        new TaskGraph("t")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, x, c)
                                .task(
                                        "g",
                                        Qwen35MMAKernels::gemmMMATiledB,
                                        new KernelContext(),
                                        a,
                                        b,
                                        c,
                                        m,
                                        n,
                                        k)
                                .task(
                                        "add",
                                        Qwen35MMAKernels::residualAdd,
                                        new KernelContext(),
                                        x,
                                        c)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, x);
                TaskGraph one =
                        new TaskGraph("o")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, x)
                                .task(
                                        "f",
                                        Qwen35MMAKernels::gemmMMATiledBResidual,
                                        new KernelContext(),
                                        a,
                                        b,
                                        x,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, x);
                GridScheduler st = new GridScheduler();
                st.addWorkerGrid("t.g", gemmGrid(m, n));
                st.addWorkerGrid("t.add", lanes(m * n));
                GridScheduler so = new GridScheduler();
                so.addWorkerGrid("o.f", gemmGrid(m, n));
                time("residual two  m=" + m + " n=" + n + " k=" + k, two, st);
                time("residual fused m=" + m + " n=" + n + " k=" + k, one, so);
            }
            int n = 17408, k = 5120;
            HalfFloatArray a = halves(m * k, 1L);
            HalfFloatArray b = tiledB(n, k, 2L);
            FloatArray gate = floats(m * n, 3L, 4.0f);
            FloatArray up = new FloatArray(m * n);
            HalfFloatArray hb = new HalfFloatArray(m * n);
            TaskGraph two =
                    new TaskGraph("t")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, gate, up, hb)
                            .task(
                                    "g",
                                    Qwen35MMAKernels::gemmMMATiledB,
                                    new KernelContext(),
                                    a,
                                    b,
                                    up,
                                    m,
                                    n,
                                    k)
                            .task(
                                    "sw",
                                    Qwen35MMAKernels::swiGLUBatchFP16,
                                    new KernelContext(),
                                    gate,
                                    up,
                                    hb)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, hb);
            TaskGraph one =
                    new TaskGraph("o")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, gate, hb)
                            .task(
                                    "f",
                                    Qwen35MMAKernels::gemmMMATiledBSwiGLU,
                                    new KernelContext(),
                                    a,
                                    b,
                                    gate,
                                    hb,
                                    m,
                                    n,
                                    k)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, hb);
            GridScheduler st = new GridScheduler();
            st.addWorkerGrid("t.g", gemmGrid(m, n));
            st.addWorkerGrid("t.sw", lanes(m * n));
            GridScheduler so = new GridScheduler();
            so.addWorkerGrid("o.f", gemmGrid(m, n));
            time("swiglu two   m=" + m, two, st);
            time("swiglu fused m=" + m, one, so);
        }
    }

    private static void time(String label, TaskGraph g, GridScheduler s) throws Exception {
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).withProfiler(ProfilerMode.SILENT);
            for (int i = 0; i < 4; i++) {
                p.execute();
            }
            long[] t = new long[11];
            for (int i = 0; i < t.length; i++) {
                t[i] = p.execute().getProfilerResult().getDeviceKernelTime();
            }
            Arrays.sort(t);
            System.out.printf(
                    Locale.ROOT,
                    "[screen] %-40s min %.1f median %.1f max %.1f us%n",
                    label,
                    t[0] / 1e3,
                    t[5] / 1e3,
                    t[10] / 1e3);
        }
    }
}
