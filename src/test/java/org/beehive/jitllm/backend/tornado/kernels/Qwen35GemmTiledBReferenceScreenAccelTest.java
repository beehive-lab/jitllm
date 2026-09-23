package org.beehive.jitllm.backend.tornado.kernels;

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
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Opt-in screen: {@code gemmMMATiledB} alone on the production shapes at M = 512 and 2048, for
 * comparison with a standalone cuBLAS FP16/FP32 reference of the same shapes and operand types.
 * Device kernel time from the profiler; 15 samples after 5 warm-ups; reports microseconds and
 * TFLOPS. {@code JITLLM_KERNEL_SCREEN=true}.
 */
public class Qwen35GemmTiledBReferenceScreenAccelTest {

    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        int[][] shapes = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}};
        HalfFloatArray b = new HalfFloatArray(17408 * 5120);
        Random rng = new Random(1L);
        for (int i = 0; i < b.getSize(); i++) {
            b.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
        }
        for (int m : new int[] {512, 2048}) {
            for (int[] shape : shapes) {
                int n = shape[0];
                int k = shape[1];
                HalfFloatArray a = new HalfFloatArray(m * k);
                for (int i = 0; i < a.getSize(); i++) {
                    a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
                }
                FloatArray c = new FloatArray(m * n);
                TaskGraph g =
                        new TaskGraph("g")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
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
                                .transferToHost(DataTransferMode.UNDER_DEMAND, c);
                WorkerGrid grid = new WorkerGrid2D((m / 128) * 256, n / 128);
                grid.setLocalWork(256, 1, 1);
                GridScheduler s = new GridScheduler();
                s.addWorkerGrid("g.g", grid);
                try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                    p.withGridScheduler(s).withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        p.execute();
                    }
                    long[] t = new long[15];
                    for (int i = 0; i < t.length; i++) {
                        t[i] = p.execute().getProfilerResult().getDeviceKernelTime();
                    }
                    long[] sorted = t.clone();
                    Arrays.sort(sorted);
                    double med = sorted[sorted.length / 2] / 1e3;
                    System.out.printf(
                            Locale.ROOT,
                            "[screen] gemmMMATiledB M=%5d N=%6d K=%6d  min %8.1f  median %8.1f  max"
                                    + " %8.1f us  median %.1f TFLOPS%n",
                            m,
                            n,
                            k,
                            sorted[0] / 1e3,
                            med,
                            sorted[sorted.length - 1] / 1e3,
                            2.0 * m * n * k / (med * 1e-6) / 1e12);
                }
            }
        }
    }
}
