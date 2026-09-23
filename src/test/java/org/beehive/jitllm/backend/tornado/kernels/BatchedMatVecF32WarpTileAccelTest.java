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
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * {@code batchedMatVecF32WarpTile} against {@code batchedMatVecF32Warp}: raw-bit equal outputs on
 * the production shape (n = 5120, d = 48) at 32, 256, 300-of-304, 512 and 2048 active rows, random
 * and cancelling inputs, NaN-poisoned outputs with inactive rows untouched. Opt-in screen of the
 * two at 512 and 2048 rows.
 */
public class BatchedMatVecF32WarpTileAccelTest {

    private static final int N = 5120;
    private static final int D = 48;

    private static FloatArray inputs(int rows, long seed, boolean cancelling) {
        Random rng = new Random(seed);
        FloatArray x = new FloatArray(rows * N);
        for (int i = 0; i < rows * N; i++) {
            x.set(i, (rng.nextFloat() * 2 - 1) * (cancelling ? 64.0f : 1.0f));
        }
        return x;
    }

    private static FloatArray weights(long seed, boolean cancelling) {
        Random rng = new Random(seed);
        FloatArray w = new FloatArray(D * N);
        for (int r = 0; r < D; r++) {
            for (int j = 0; j < N; j++) {
                float v = rng.nextFloat() * 2 - 1;
                if (cancelling && (j & 1) == 1) {
                    v = -w.get(r * N + j - 1) + (rng.nextFloat() - 0.5f) * 1e-3f;
                }
                w.set(r * N + j, v);
            }
        }
        return w;
    }

    private static WorkerGrid warpGrid(int rows) {
        WorkerGrid g = new WorkerGrid1D(rows * D * 32);
        g.setLocalWork(128, 1, 1);
        return g;
    }

    private static WorkerGrid tileGrid(int rows) {
        int tiles = ((rows + 3) / 4) * (D / 4);
        WorkerGrid g = new WorkerGrid1D(tiles * 32);
        g.setLocalWork(128, 1, 1);
        return g;
    }

    private static void assertParity(int rows, int active, long seed, boolean cancelling)
            throws Exception {
        FloatArray x = inputs(rows, seed, cancelling);
        FloatArray w = weights(seed + 1, cancelling);
        FloatArray control = new FloatArray(rows * D);
        FloatArray candidate = new FloatArray(rows * D);
        control.init(Float.NaN);
        candidate.init(Float.NaN);
        TaskGraph graph =
                new TaskGraph("mv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, control, candidate)
                        .task(
                                "control",
                                TransformerBatchPrefillKernels::batchedMatVecF32Warp,
                                new KernelContext(),
                                x,
                                control,
                                w,
                                N,
                                D,
                                active)
                        .task(
                                "candidate",
                                TransformerBatchPrefillKernels::batchedMatVecF32WarpTile,
                                new KernelContext(),
                                x,
                                candidate,
                                w,
                                N,
                                D,
                                active)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("mv.control", warpGrid(rows));
        s.addWorkerGrid("mv.candidate", tileGrid(rows));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int count = 0;
        String first = null;
        String what = "rows=" + rows + " active=" + active + (cancelling ? " cancelling" : "");
        for (int i = 0; i < rows * D; i++) {
            float c = control.get(i);
            float k = candidate.get(i);
            if (i / D < active) {
                assertTrue(what + ": control not finite at " + i, Float.isFinite(c));
                assertTrue(what + ": candidate not finite at " + i, Float.isFinite(k));
            } else {
                assertTrue(what + ": control wrote an inactive row at " + i, Float.isNaN(c));
                assertTrue(what + ": candidate wrote an inactive row at " + i, Float.isNaN(k));
                continue;
            }
            if (Float.floatToRawIntBits(c) != Float.floatToRawIntBits(k)) {
                if (first == null) {
                    first = "row " + (i / D) + " out " + (i % D) + ": " + c + " vs " + k;
                }
                count++;
            }
        }
        assertEquals(what + ": " + count + " outputs differ, first at " + first, 0, count);
    }

    @Test
    public void theTileIsRawBitEqualToTheWarpKernel() throws Exception {
        assertParity(32, 32, 1L, false);
        assertParity(256, 256, 2L, false);
        assertParity(304, 300, 3L, false);
        assertParity(304, 301, 4L, true);
        assertParity(512, 512, 5L, true);
        assertParity(2048, 2048, 6L, false);
    }

    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        for (int rows : new int[] {512, 2048}) {
            FloatArray x = inputs(rows, 7L, false);
            FloatArray w = weights(8L, false);
            FloatArray o1 = new FloatArray(rows * D);
            FloatArray o2 = new FloatArray(rows * D);
            TaskGraph g1 =
                    new TaskGraph("c")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w)
                            .task(
                                    "k",
                                    TransformerBatchPrefillKernels::batchedMatVecF32Warp,
                                    new KernelContext(),
                                    x,
                                    o1,
                                    w,
                                    N,
                                    D,
                                    rows)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, o1);
            TaskGraph g2 =
                    new TaskGraph("t")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w)
                            .task(
                                    "k",
                                    TransformerBatchPrefillKernels::batchedMatVecF32WarpTile,
                                    new KernelContext(),
                                    x,
                                    o2,
                                    w,
                                    N,
                                    D,
                                    rows)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, o2);
            GridScheduler s1 = new GridScheduler();
            s1.addWorkerGrid("c.k", warpGrid(rows));
            GridScheduler s2 = new GridScheduler();
            s2.addWorkerGrid("t.k", tileGrid(rows));
            try (TornadoExecutionPlan p1 = new TornadoExecutionPlan(g1.snapshot());
                    TornadoExecutionPlan p2 = new TornadoExecutionPlan(g2.snapshot())) {
                p1.withGridScheduler(s1).withProfiler(ProfilerMode.SILENT);
                p2.withGridScheduler(s2).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < 5; i++) {
                    p1.execute();
                    p2.execute();
                }
                long[] t1 = new long[15];
                long[] t2 = new long[15];
                for (int i = 0; i < 15; i++) {
                    if ((i & 1) == 0) {
                        t1[i] = p1.execute().getProfilerResult().getDeviceKernelTime();
                        t2[i] = p2.execute().getProfilerResult().getDeviceKernelTime();
                    } else {
                        t2[i] = p2.execute().getProfilerResult().getDeviceKernelTime();
                        t1[i] = p1.execute().getProfilerResult().getDeviceKernelTime();
                    }
                }
                report("warp rows=" + rows, t1);
                report("tile rows=" + rows, t2);
            }
        }
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
                "[screen] %-20s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
