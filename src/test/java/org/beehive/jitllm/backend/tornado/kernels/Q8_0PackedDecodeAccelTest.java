package org.beehive.jitllm.backend.tornado.kernels;

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
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The packed-layout decode kernels ({@link TransformerComputeKernelsQ8_0Packed}) against the DP4A
 * kernels over the same Q8_0 weights in block layout, on the same quantized activation. The integer
 * products are the same; only the order of the FP32 scaling and summation differs, so the results
 * agree to within FP32 rounding of the sum of magnitudes.
 */
public class Q8_0PackedDecodeAccelTest {

    private static final int MATVEC_LOCAL = 128;

    private static byte[] randomWeights(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        byte[] raw = new byte[rows * (cols / 32) * 34];
        rng.nextBytes(raw);
        for (int b = 0; b < rows * (cols / 32); b++) {
            int bits = 0x2000 | rng.nextInt(0x400); // a positive normal FP16 scale
            raw[b * 34] = (byte) bits;
            raw[b * 34 + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static IntArray randomQuants(int cols, long seed) {
        Random rng = new Random(seed);
        IntArray q = new IntArray(cols / 4);
        for (int i = 0; i < cols / 4; i++) {
            q.set(i, rng.nextInt());
        }
        return q;
    }

    private static FloatArray randomScales(int cols, long seed) {
        Random rng = new Random(seed);
        FloatArray s = new FloatArray(cols / 32);
        for (int i = 0; i < cols / 32; i++) {
            s.set(i, 0.01f + rng.nextFloat() * 0.02f);
        }
        return s;
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid g = new WorkerGrid1D(global);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    /** Worst |a - b| over the scale of the outputs (their largest magnitude). */
    private static double worst(FloatArray a, FloatArray b) {
        double max = 0;
        double scale = 1e-6;
        for (int i = 0; i < a.getSize(); i++) {
            scale = Math.max(scale, Math.abs(a.get(i)));
        }
        for (int i = 0; i < a.getSize(); i++) {
            max = Math.max(max, Math.abs(a.get(i) - b.get(i)) / scale);
        }
        return max;
    }

    @Test
    public void theMatrixVectorProductsMatchDp4a() throws Exception {
        // The 27B's attn_q shape and a 5120 x 17408 down projection, so both k sizes are used.
        int[][] shapes = {{12288, 5120}, {5120, 17408}, {1024, 5120}};
        for (int[] shape : shapes) {
            int d = shape[0], n = shape[1];
            ByteArray w = ByteArray.fromArray(randomWeights(d, n, 11 + d));
            ByteArray packed = Int8GemmKernels.packQ8_0Tiles(w, d, n);
            IntArray xq = randomQuants(n, 12 + d);
            FloatArray xs = randomScales(n, 13 + d);
            FloatArray ref = new FloatArray(d);
            FloatArray got = new FloatArray(d);
            FloatArray refRes = new FloatArray(d);
            FloatArray gotRes = new FloatArray(d);
            for (int i = 0; i < d; i++) {
                refRes.set(i, 0.5f * i);
                gotRes.set(i, 0.5f * i);
            }
            TaskGraph g =
                    new TaskGraph("pd" + d)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    w,
                                    packed,
                                    xq,
                                    xs,
                                    ref,
                                    got,
                                    refRes,
                                    gotRes)
                            .task(
                                    "r",
                                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                    new KernelContext(),
                                    xq,
                                    xs,
                                    ref,
                                    w,
                                    n,
                                    d,
                                    MATVEC_LOCAL)
                            .task(
                                    "p",
                                    TransformerComputeKernelsQ8_0Packed::matrixVectorQ8_0Packed,
                                    new KernelContext(),
                                    xq,
                                    xs,
                                    got,
                                    packed,
                                    n,
                                    d,
                                    0)
                            .task(
                                    "rr",
                                    TransformerComputeKernelsQ8_0DP4A
                                            ::matrixVectorGenericWithResidualQ8_0DP4A,
                                    new KernelContext(),
                                    xq,
                                    xs,
                                    refRes,
                                    w,
                                    n,
                                    d,
                                    MATVEC_LOCAL)
                            .task(
                                    "pr",
                                    TransformerComputeKernelsQ8_0Packed::matrixVectorQ8_0Packed,
                                    new KernelContext(),
                                    xq,
                                    xs,
                                    gotRes,
                                    packed,
                                    n,
                                    d,
                                    1)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION, ref, got, refRes, gotRes);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("pd" + d + ".r", grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
            s.addWorkerGrid("pd" + d + ".rr", grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
            s.addWorkerGrid(
                    "pd" + d + ".p",
                    grid(
                            d / 8 * TransformerComputeKernelsQ8_0Packed.LOCAL,
                            TransformerComputeKernelsQ8_0Packed.LOCAL));
            s.addWorkerGrid(
                    "pd" + d + ".pr",
                    grid(
                            d / 8 * TransformerComputeKernelsQ8_0Packed.LOCAL,
                            TransformerComputeKernelsQ8_0Packed.LOCAL));
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(s).execute();
            }
            double store = worst(ref, got);
            double residual = worst(refRes, gotRes);
            assertTrue(d + "x" + n + " store: worst " + store, store < 1e-5);
            assertTrue(d + "x" + n + " residual: worst " + residual, residual < 1e-5);
        }
    }

    @Test
    public void theFp32ActivationProductMatchesTheBlockKernel() throws Exception {
        int d = 5120, n = 6144; // the 27B's attn_output shape
        ByteArray w = ByteArray.fromArray(randomWeights(d, n, 31));
        ByteArray packed = Int8GemmKernels.packQ8_0Tiles(w, d, n);
        Random rng = new Random(32);
        FloatArray x = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            x.set(i, (float) rng.nextGaussian());
        }
        FloatArray ref = new FloatArray(d);
        FloatArray got = new FloatArray(d);
        for (int i = 0; i < d; i++) {
            ref.set(i, 0.25f * i);
            got.set(i, 0.25f * i);
        }
        TaskGraph g =
                new TaskGraph("pf")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, packed, x, ref, got)
                        .task(
                                "r",
                                TransformerComputeKernelsLayered
                                        ::matrixVectorGenericWithResidualQ8_0Byte,
                                new KernelContext(),
                                x,
                                ref,
                                w,
                                n,
                                d,
                                MATVEC_LOCAL)
                        .task(
                                "p",
                                TransformerComputeKernelsQ8_0Packed::matrixVectorQ8_0PackedF32,
                                new KernelContext(),
                                x,
                                got,
                                packed,
                                n,
                                d,
                                1)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, ref, got);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("pf.r", grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
        s.addWorkerGrid(
                "pf.p",
                grid(
                        d / 8 * TransformerComputeKernelsQ8_0Packed.LOCAL,
                        TransformerComputeKernelsQ8_0Packed.LOCAL));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        double residual = worst(ref, got);
        assertTrue("worst " + residual, residual < 1e-5);
    }

    @Test
    public void theFusedGateUpMatchesDp4a() throws Exception {
        int d = 17408, n = 5120;
        ByteArray w1 = ByteArray.fromArray(randomWeights(d, n, 21));
        ByteArray w3 = ByteArray.fromArray(randomWeights(d, n, 22));
        ByteArray p1 = Int8GemmKernels.packQ8_0Tiles(w1, d, n);
        ByteArray p3 = Int8GemmKernels.packQ8_0Tiles(w3, d, n);
        IntArray xq = randomQuants(n, 23);
        FloatArray xs = randomScales(n, 24);
        FloatArray ref = new FloatArray(d);
        FloatArray got = new FloatArray(d);
        TaskGraph g =
                new TaskGraph("pg")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, w1, w3, p1, p3, xq, xs, ref, got)
                        .task(
                                "r",
                                TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                                new KernelContext(),
                                xq,
                                xs,
                                ref,
                                w1,
                                w3,
                                n,
                                d,
                                MATVEC_LOCAL)
                        .task(
                                "p",
                                TransformerComputeKernelsQ8_0Packed::fusedFFNGateUpSiLUQ8_0Packed,
                                new KernelContext(),
                                xq,
                                xs,
                                got,
                                p1,
                                p3,
                                n,
                                d)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, ref, got);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("pg.r", grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
        s.addWorkerGrid(
                "pg.p",
                grid(
                        d / 8 * TransformerComputeKernelsQ8_0Packed.FUSED_LOCAL,
                        TransformerComputeKernelsQ8_0Packed.FUSED_LOCAL));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        double fused = worst(ref, got);
        assertTrue("worst " + fused, fused < 1e-4);
    }
}
