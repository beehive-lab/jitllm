package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The format-neutral primitives a recurrent mixer needs, <b>compiled and executed on a device</b>.
 *
 * <p>Their host parity tests settle the addressing. This settles that TornadoVM compiles them and
 * that a real launch — a grid larger than the data, a workgroup-per-row reduction — produces the
 * same answers. Both have failed independently of arithmetic in this backend before.
 *
 * <p>The F32 matrix-vector kernel is not new; it is the one an {@code ssm_alpha} or {@code
 * ssm_beta} projection needs, whose weights are F32 in every file that carries them, and nothing
 * had yet run it at that shape. A projection of 5120 inputs to 48 rows is a much wider row and a
 * much shorter output than any existing call site.
 */
public class SharedComputeKernelAccelTest {

    private static final float POISON = -999999f;

    private static FloatArray toDevice(float[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, v[i]);
        }
        return a;
    }

    private static void run(TaskGraph graph, String task, int global, int local) throws Exception {
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(local, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(graph.getTaskGraphName() + "." + task, worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    /** The delta-net's own {@code 2048 | 2048 | 6144}, with a grid rounded up past the data. */
    @Test
    public void theThreeWaySplitRunsOnTheDevice() throws Exception {
        final int dimA = 2048;
        final int dimB = 2048;
        final int dimC = 6144;
        final int total = dimA + dimB + dimC;

        Random random = new Random(11L);
        float[] fused = new float[total];
        for (int i = 0; i < total; i++) {
            fused[i] = (float) random.nextGaussian();
        }
        fused[dimA - 1] = -7.5f;
        fused[dimA] = 7.5f;
        fused[dimA + dimB] = 0.0f;

        FloatArray device = toDevice(fused);
        FloatArray a = new FloatArray(dimA);
        FloatArray b = new FloatArray(dimB);
        FloatArray c = new FloatArray(dimC);
        a.init(POISON);
        b.init(POISON);
        c.init(POISON);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("split")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, device, a, b, c)
                        .task(
                                "k",
                                TransformerComputeKernels::splitThreeWay,
                                context,
                                device,
                                a,
                                b,
                                c,
                                dimA,
                                dimB,
                                dimC)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, a, b, c);
        // 10496 lanes for 10240 elements: the guard has to hold, not the grid.
        run(graph, "k", 10496, 256);

        for (int i = 0; i < dimA; i++) {
            assertEquals("a[" + i + "]", fused[i], a.get(i), 0.0f);
        }
        for (int i = 0; i < dimB; i++) {
            assertEquals("b[" + i + "]", fused[dimA + i], b.get(i), 0.0f);
        }
        for (int i = 0; i < dimC; i++) {
            assertEquals("c[" + i + "]", fused[dimA + dimB + i], c.get(i), 0.0f);
        }
        for (int i = 0; i < total; i++) {
            assertEquals("the source was modified at " + i, fused[i], device.get(i), 0.0f);
        }
    }

    /** {@code 1/sqrt(128)} over a key-width vector, in place. */
    @Test
    public void theInPlaceScaleRunsOnTheDevice() throws Exception {
        final int size = 2048;
        final float factor = (float) (1.0 / Math.sqrt(128));

        Random random = new Random(12L);
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (float) random.nextGaussian() * 3.0f;
        }
        values[0] = 0.0f;
        values[size - 1] = -4.0f;

        FloatArray device = toDevice(values);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("scale")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, device)
                        .task(
                                "k",
                                TransformerComputeKernels::scaleInPlace,
                                context,
                                device,
                                factor,
                                size)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, device);
        run(graph, "k", 2048, 256);

        for (int i = 0; i < size; i++) {
            assertEquals("scaled[" + i + "]", values[i] * factor, device.get(i), 0.0f);
        }
    }

    /** An {@code ssm_alpha}-shaped F32 projection: 5120 inputs, 48 rows, one workgroup per row. */
    @Test
    public void theF32MatrixVectorRunsOnTheDevice() throws Exception {
        final int n = 5120;
        final int d = 48;
        final int local = 128;

        Random random = new Random(13L);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) random.nextGaussian() * 0.5f;
        }
        float[] w = new float[n * d];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) random.nextGaussian() * 0.05f;
        }

        float[] expected = new float[d];
        for (int row = 0; row < d; row++) {
            float sum = 0f;
            for (int col = 0; col < n; col++) {
                sum += w[row * n + col] * x[col];
            }
            expected[row] = sum;
        }

        FloatArray deviceX = toDevice(x);
        FloatArray deviceW = toDevice(w);
        FloatArray out = new FloatArray(d);
        out.init(POISON);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("f32matvec")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, deviceX, deviceW, out)
                        .task(
                                "k",
                                TransformerComputeKernelsLayered::matrixVectorGeneric,
                                context,
                                deviceX,
                                out,
                                deviceW,
                                n,
                                d,
                                local)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        run(graph, "k", d * local, local);

        for (int row = 0; row < d; row++) {
            assertTrue("row " + row + " was never written", out.get(row) != POISON);
            assertEquals(
                    "row " + row,
                    expected[row],
                    out.get(row),
                    Math.max(1e-3f, Math.abs(expected[row]) * 1e-4f));
        }
    }
}
