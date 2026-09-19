package org.beehive.jllm.backend.tornado.kernels;

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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * {@code causalConv1dBatch} + {@code causalConv1dWindowUpdate} against {@code causalConv1dScan}:
 * outputs and the final window raw-bit equal over chunks of 1, 2, 3, 300 and 2048 active rows in a
 * 2048-row buffer (inactive rows NaN-poisoned and untouched by both), a nonzero window offset,
 * random taps and a random initial window. A negative control (the update run before the
 * convolution) differs.
 */
public class Qwen35ParallelConvBatchAccelTest {

    private static final int CHANNELS = 10240;
    private static final int KERNEL = 4;
    private static final int ROWS = 2048;

    private static FloatArray random(int n, long seed) {
        FloatArray a = new FloatArray(n);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            a.set(i, rng.nextFloat() * 2.0f - 1.0f);
        }
        return a;
    }

    private static FloatArray input(int active, long seed) {
        FloatArray a = random(ROWS * CHANNELS, seed);
        for (int i = active * CHANNELS; i < ROWS * CHANNELS; i++) {
            a.set(i, Float.NaN);
        }
        return a;
    }

    private static int mismatches(String what, FloatArray a, FloatArray b) {
        int n = 0;
        String first = null;
        for (int i = 0; i < a.getSize(); i++) {
            if (Float.floatToRawIntBits(a.get(i)) != Float.floatToRawIntBits(b.get(i))) {
                if (first == null) {
                    first = "index " + i + ": " + a.get(i) + " vs " + b.get(i);
                }
                n++;
            }
        }
        if (n > 0) {
            System.out.println(what + ": " + n + " differ, first at " + first);
        }
        return n;
    }

    private static WorkerGrid lanes(int count, int local) {
        WorkerGrid g = new WorkerGrid1D(count);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    /** Returns {outputs differing, window slots differing}. */
    private static int[] compare(int active, boolean updateFirst) throws Exception {
        int offset = 3 * CHANNELS * (KERNEL - 1);
        int windowSize = offset + CHANNELS * (KERNEL - 1) + 7;
        FloatArray in = input(active, 11L + active);
        FloatArray taps = random(CHANNELS * KERNEL, 5L);
        FloatArray window1 = random(windowSize, 6L);
        FloatArray window2 = random(windowSize, 6L);
        FloatArray out1 = new FloatArray(ROWS * CHANNELS);
        FloatArray out2 = new FloatArray(ROWS * CHANNELS);
        out1.init(Float.NaN);
        out2.init(Float.NaN);
        IntArray info = new IntArray(4);
        info.set(1, active);
        TaskGraph g =
                new TaskGraph("cv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                in,
                                taps,
                                window1,
                                window2,
                                out1,
                                out2,
                                info)
                        .task(
                                "scan",
                                Qwen35BatchKernels::causalConv1dScan,
                                new KernelContext(),
                                in,
                                taps,
                                window1,
                                out1,
                                CHANNELS,
                                KERNEL,
                                offset,
                                info);
        if (updateFirst) {
            g.task(
                    "window",
                    Qwen35BatchKernels::causalConv1dWindowUpdate,
                    new KernelContext(),
                    in,
                    window2,
                    CHANNELS,
                    KERNEL,
                    offset,
                    info);
        }
        g.task(
                "batch",
                Qwen35ReferenceKernels::causalConv1dBatch,
                new KernelContext(),
                in,
                taps,
                window2,
                out2,
                CHANNELS,
                KERNEL,
                offset,
                info);
        if (!updateFirst) {
            g.task(
                    "window",
                    Qwen35BatchKernels::causalConv1dWindowUpdate,
                    new KernelContext(),
                    in,
                    window2,
                    CHANNELS,
                    KERNEL,
                    offset,
                    info);
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, window1, window2, out1, out2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("cv.scan", lanes(CHANNELS, 128));
        s.addWorkerGrid("cv.batch", lanes(ROWS * CHANNELS, 128));
        s.addWorkerGrid("cv.window", lanes(CHANNELS, 128));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        for (int i = 0; i < active * CHANNELS; i++) {
            assertTrue("scan output not finite at " + i, !Float.isNaN(out1.get(i)));
        }
        for (int i = active * CHANNELS; i < ROWS * CHANNELS; i++) {
            assertTrue("batch wrote an inactive row at " + i, Float.isNaN(out2.get(i)));
        }
        return new int[] {
            mismatches("outputs active=" + active, out1, out2),
            mismatches("window active=" + active, window1, window2)
        };
    }

    @Test
    public void outputsAndWindowAreRawBitEqualToTheScan() throws Exception {
        for (int active : new int[] {1, 2, 3, 300, 2048}) {
            int[] d = compare(active, false);
            assertEquals("outputs differ at " + active + " rows", 0, d[0]);
            assertEquals("window differs at " + active + " rows", 0, d[1]);
        }
    }

    /**
     * The fused convolution + SiLU + split against the three tasks: q, k and v raw-bit equal at 300
     * and 2048 active rows (dims 4096 / 4096 / 2048 of the 10240 channels), NaN-poisoned.
     */
    @Test
    public void theFusedSiluSplitIsRawBitEqualToTheThreeTasks() throws Exception {
        int dimA = 4096;
        int dimB = 4096;
        int dimC = CHANNELS - dimA - dimB;
        for (int active : new int[] {300, 2048}) {
            int offset = 3 * CHANNELS * (KERNEL - 1);
            int windowSize = offset + CHANNELS * (KERNEL - 1) + 7;
            FloatArray in = input(active, 21L + active);
            FloatArray taps = random(CHANNELS * KERNEL, 5L);
            FloatArray window1 = random(windowSize, 6L);
            FloatArray window2 = random(windowSize, 6L);
            FloatArray conv = new FloatArray(ROWS * CHANNELS);
            FloatArray[] ctl = {
                new FloatArray(ROWS * dimA),
                new FloatArray(ROWS * dimB),
                new FloatArray(ROWS * dimC)
            };
            FloatArray[] cand = {
                new FloatArray(ROWS * dimA),
                new FloatArray(ROWS * dimB),
                new FloatArray(ROWS * dimC)
            };
            for (FloatArray f : ctl) {
                f.init(Float.NaN);
            }
            for (FloatArray f : cand) {
                f.init(Float.NaN);
            }
            IntArray info = new IntArray(4);
            info.set(1, active);
            TaskGraph g =
                    new TaskGraph("fs")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    in,
                                    taps,
                                    window1,
                                    window2,
                                    conv,
                                    ctl[0],
                                    ctl[1],
                                    ctl[2],
                                    cand[0],
                                    cand[1],
                                    cand[2],
                                    info)
                            .task(
                                    "conv",
                                    Qwen35ReferenceKernels::causalConv1dBatch,
                                    new KernelContext(),
                                    in,
                                    taps,
                                    window1,
                                    conv,
                                    CHANNELS,
                                    KERNEL,
                                    offset,
                                    info)
                            .task(
                                    "silu",
                                    Qwen35BatchKernels::siluInPlaceBatch,
                                    new KernelContext(),
                                    conv,
                                    CHANNELS,
                                    info)
                            .task(
                                    "split",
                                    Qwen35BatchKernels::splitThreeWayBatch,
                                    new KernelContext(),
                                    conv,
                                    ctl[0],
                                    ctl[1],
                                    ctl[2],
                                    dimA,
                                    dimB,
                                    dimC,
                                    info)
                            .task(
                                    "fused",
                                    Qwen35BatchKernels::causalConv1dSiluSplitBatch,
                                    new KernelContext(),
                                    in,
                                    taps,
                                    window2,
                                    cand[0],
                                    cand[1],
                                    cand[2],
                                    dimA,
                                    dimB,
                                    dimC,
                                    KERNEL,
                                    offset,
                                    info)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION,
                                    ctl[0],
                                    ctl[1],
                                    ctl[2],
                                    cand[0],
                                    cand[1],
                                    cand[2]);
            GridScheduler s = new GridScheduler();
            for (String t : new String[] {"fs.conv", "fs.silu", "fs.split", "fs.fused"}) {
                s.addWorkerGrid(t, lanes(ROWS * CHANNELS, 128));
            }
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).execute();
            }
            String[] names = {"q", "k", "v"};
            for (int i = 0; i < 3; i++) {
                assertTrue(names[i] + " control not finite", !Float.isNaN(ctl[i].get(0)));
                assertEquals(
                        names[i] + " active=" + active, 0, mismatches(names[i], ctl[i], cand[i]));
            }
        }
    }

    @Test
    public void theWindowUpdateBeforeTheConvolutionDiffers() throws Exception {
        int[] d = compare(300, true);
        assertTrue("updating the window first agreed", d[0] > 0);
    }
}
