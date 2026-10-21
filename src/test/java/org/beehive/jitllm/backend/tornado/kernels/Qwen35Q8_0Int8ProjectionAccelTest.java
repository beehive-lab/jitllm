package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The Q8_0 batched-prefill projections: the int8 GEMM that reads the Q8_0 blocks itself, its three
 * epilogues, and the warp kernel for the projections too narrow to tile.
 */
public class Qwen35Q8_0Int8ProjectionAccelTest {

    private static final int BLOCK_BYTES = 34;

    private static byte[] randomWeights(int n, int k, long seed) {
        byte[] raw = new byte[n * (k / 32) * BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int b = 0; b < n * (k / 32); b++) {
            // A positive or negative FP16 scale around 2^-7.
            int bits = 0x2000 | rng.nextInt(0x400) | (rng.nextBoolean() ? 0x8000 : 0);
            raw[b * BLOCK_BYTES] = (byte) (bits & 0xFF);
            raw[b * BLOCK_BYTES + 1] = (byte) (bits >> 8);
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

    private static FloatArray activations(int m, int k, long seed) {
        Random rng = new Random(seed);
        FloatArray a = new FloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a.set(i, (float) rng.nextGaussian());
        }
        return a;
    }

    private static float scale(byte[] raw, int k, int col, int block) {
        int base = (col * (k / 32) + block) * BLOCK_BYTES;
        return new HalfFloat((short) ((raw[base] & 0xFF) | ((raw[base + 1] & 0xFF) << 8)))
                .getFloat32();
    }

    private static byte quant(byte[] raw, int k, int col, int e) {
        return raw[(col * (k / 32) + e / 32) * BLOCK_BYTES + 2 + e % 32];
    }

    /** The rows holder a GEMM reads its active row count from: all {@code m} rows. */
    private static IntArray rowsOf(int m) {
        IntArray rows = new IntArray(3);
        rows.set(1, m);
        return rows;
    }

    private static WorkerGrid lanes(int count, int local) {
        WorkerGrid g = new WorkerGrid1D(count);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        return gemmGrid(m, n, 1);
    }

    private static WorkerGrid gemmGrid(int m, int n, int splits) {
        int local = Int8GemmKernels.Q8_GEMM_THREADS;
        WorkerGrid gemm = new WorkerGrid2D((m / 128) * local, n / 128 * splits);
        gemm.setLocalWork(local, 1, 1);
        return gemm;
    }

    /**
     * Quantize, then the GEMM over the Q8_0 blocks, against an FP64 product of the same quantized
     * activations and weights: only the FP32 accumulation separates them. Two row tiles, three
     * column tiles and five 64-k rounds, so every tile and round boundary is crossed.
     */
    @Test
    public void theGemmMatchesTheQuantizedProduct() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int m = 256, n = 384, k = 320;
        byte[] raw = randomWeights(n, k, 21);
        ByteArray w = toDevice(raw);
        FloatArray a = activations(m, k, 22);
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        FloatArray out = new FloatArray(m * n);
        out.init(Float.NaN);
        TaskGraph g =
                new TaskGraph("q8g")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w, q8, dA, out)
                        .task(
                                "q",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                a,
                                q8,
                                dA,
                                k)
                        .task(
                                "g",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                out,
                                out,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_STORE,
                                out,
                                1,
                                rowsOf(m),
                                n,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out, q8, dA);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("q8g.q", lanes(m * k, 256));
        s.addWorkerGrid("q8g.g", gemmGrid(m, n));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        double worst = 0;
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double expected = 0;
                double magnitude = 0;
                for (int b = 0; b < k / 32; b++) {
                    long dot = 0;
                    for (int i = 0; i < 32; i++) {
                        dot += (long) q8.get(row * k + b * 32 + i) * quant(raw, k, col, b * 32 + i);
                    }
                    double term = (double) dot * dA.get(row * (k / 32) + b) * scale(raw, k, col, b);
                    expected += term;
                    magnitude += Math.abs(term);
                }
                worst = Math.max(worst, Math.abs(out.get(row * n + col) - expected) / magnitude);
            }
        }
        assertTrue("worst relative error " + worst, worst < 1e-5);
    }

    /**
     * The GEMM over packed weights ({@link Int8GemmKernels#packQ8_0Tiles}) against {@link
     * Int8GemmKernels#gemmInt8Q8_0} on the same operands, unsplit and split three ways (ten 64-k
     * rounds: shares of 4, 4 and 2). Same summation order, so the stored results and the split
     * partial sums are equal bit for bit.
     */
    @Test
    public void thePackedGemmMatchesTheGemm() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int m = 256, n = 384, k = 640;
        byte[] raw = randomWeights(n, k, 41);
        ByteArray w = toDevice(raw);
        FloatArray a = activations(m, k, 42);
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        IntArray rows = rowsOf(m);
        ByteArray packed = Int8GemmKernels.packQ8_0Tiles(w, n, k);
        for (int splits : new int[] {1, 3}) {
            {
                FloatArray reference = new FloatArray(splits * m * n);
                FloatArray staged = new FloatArray(splits * m * n);
                reference.init(Float.NaN);
                staged.init(Float.NaN);
                // Split, every split writes its partial sums; unsplit, the output itself.
                TaskGraph g =
                        new TaskGraph("q8k")
                                .transferToDevice(
                                        DataTransferMode.EVERY_EXECUTION,
                                        a,
                                        w,
                                        packed,
                                        q8,
                                        dA,
                                        reference,
                                        staged,
                                        rows)
                                .task(
                                        "q",
                                        Int8GemmKernels::quantizeActivationsQ8Warp,
                                        new KernelContext(),
                                        a,
                                        q8,
                                        dA,
                                        k)
                                .task(
                                        "g",
                                        Int8GemmKernels::gemmInt8Q8_0,
                                        new KernelContext(),
                                        q8,
                                        dA,
                                        w,
                                        reference,
                                        reference,
                                        m,
                                        n,
                                        k,
                                        Int8GemmKernels.EPILOGUE_STORE,
                                        reference,
                                        splits,
                                        rows,
                                        n,
                                        0)
                                .task(
                                        "p",
                                        Int8GemmKernels::gemmInt8Q8_0Packed,
                                        new KernelContext(),
                                        q8,
                                        dA,
                                        packed,
                                        staged,
                                        staged,
                                        m,
                                        n,
                                        k,
                                        Int8GemmKernels.EPILOGUE_STORE,
                                        staged,
                                        splits,
                                        rows,
                                        n,
                                        0)
                                .transferToHost(
                                        DataTransferMode.EVERY_EXECUTION, reference, staged);
                GridScheduler s = new GridScheduler();
                s.addWorkerGrid("q8k.q", lanes(m * k, 256));
                s.addWorkerGrid("q8k.g", gemmGrid(m, n, splits));
                s.addWorkerGrid("q8k.p", gemmGrid(m, n, splits));
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                    plan.withGridScheduler(s).execute();
                }
                for (int i = 0; i < splits * m * n; i++) {
                    if (splits == 1 || !Float.isNaN(reference.get(i))) {
                        assertEquals(
                                "splits " + splits + ", element " + i,
                                reference.get(i),
                                staged.get(i),
                                0.0f);
                    }
                }
            }
        }
    }

    /**
     * The residual and SwiGLU epilogues against the stored product of the same operands: the
     * residual adds it to what was there, bit for bit; SwiGLU multiplies it by {@code silu(gate)}.
     */
    @Test
    public void theEpiloguesApplyToTheSameProduct() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int m = 128, n = 256, k = 192;
        byte[] raw = randomWeights(n, k, 41);
        ByteArray w = toDevice(raw);
        FloatArray a = activations(m, k, 42);
        FloatArray gate = activations(m, n, 43);
        FloatArray before = activations(m, n, 44);
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        FloatArray stored = new FloatArray(m * n);
        FloatArray residual = new FloatArray(m * n);
        for (int i = 0; i < m * n; i++) {
            residual.set(i, before.get(i));
        }
        FloatArray swiglu = new FloatArray(m * n);
        TaskGraph g =
                new TaskGraph("q8e")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                a,
                                w,
                                gate,
                                q8,
                                dA,
                                stored,
                                residual,
                                swiglu)
                        .task(
                                "q",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                a,
                                q8,
                                dA,
                                k)
                        .task(
                                "s",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                stored,
                                stored,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_STORE,
                                stored,
                                1,
                                rowsOf(m),
                                n,
                                0)
                        .task(
                                "r",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                residual,
                                residual,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_RESIDUAL,
                                residual,
                                1,
                                rowsOf(m),
                                n,
                                0)
                        .task(
                                "g",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                swiglu,
                                gate,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_SWIGLU,
                                swiglu,
                                1,
                                rowsOf(m),
                                n,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, stored, residual, swiglu);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("q8e.q", lanes(m * k, 256));
        for (String task : new String[] {"s", "r", "g"}) {
            s.addWorkerGrid("q8e." + task, gemmGrid(m, n));
        }
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        for (int i = 0; i < m * n; i++) {
            float product = stored.get(i);
            assertEquals("residual " + i, before.get(i) + product, residual.get(i), 0.0f);
            float g0 = gate.get(i);
            double expected = g0 / (1.0 + Math.exp(-g0)) * product;
            assertEquals("swiglu " + i, expected, swiglu.get(i), 1e-5 * (1 + Math.abs(expected)));
        }
    }

    /** The warp kernel, storing and accumulating, against an FP64 product. */
    @Test
    public void theWarpKernelMatchesTheProduct() throws Exception {
        assumeTrue("no tensor cores", TensorCoreSupport.isTensorCoreCapableBackend());
        int m = 7, n = 320, d = 48;
        byte[] raw = randomWeights(d, n, 31);
        ByteArray w = toDevice(raw);
        FloatArray x = activations(m, n, 32);
        FloatArray stored = new FloatArray(m * d);
        stored.init(Float.NaN);
        FloatArray accumulated = new FloatArray(m * d);
        accumulated.init(1.5f);
        TaskGraph g =
                new TaskGraph("q8w")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, x, w, stored, accumulated)
                        .task(
                                "s",
                                TransformerBatchPrefillKernels::batchedMatVecQ8_0Warp,
                                new KernelContext(),
                                x,
                                stored,
                                w,
                                n,
                                d,
                                m,
                                0)
                        .task(
                                "a",
                                TransformerBatchPrefillKernels::batchedMatVecQ8_0Warp,
                                new KernelContext(),
                                x,
                                accumulated,
                                w,
                                n,
                                d,
                                m,
                                1)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, stored, accumulated);
        GridScheduler s = new GridScheduler();
        int lanes = ((m * d * 32) + 127) / 128 * 128;
        s.addWorkerGrid("q8w.s", lanes(lanes, 128));
        s.addWorkerGrid("q8w.a", lanes(lanes, 128));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        for (int row = 0; row < m; row++) {
            for (int o = 0; o < d; o++) {
                double expected = 0;
                for (int e = 0; e < n; e++) {
                    expected +=
                            (double) scale(raw, n, o, e / 32)
                                    * quant(raw, n, o, e)
                                    * x.get(row * n + e);
                }
                assertEquals(expected, stored.get(row * d + o), 1e-4 * (1 + Math.abs(expected)));
                assertEquals(
                        expected + 1.5,
                        accumulated.get(row * d + o),
                        1e-4 * (1 + Math.abs(expected)));
            }
        }
    }

    /**
     * K split in three, then the partial sums added: against the unsplit GEMM on the same operands,
     * storing and adding to a residual. Only the order of the FP32 additions differs.
     */
    @Test
    public void aSplitGemmMatchesTheUnsplitOne() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int m = 128, n = 256, k = 448; // seven rounds: splits of three, three and one
        int splits = 3;
        byte[] raw = randomWeights(n, k, 51);
        ByteArray w = toDevice(raw);
        FloatArray a = activations(m, k, 52);
        FloatArray before = activations(m, n, 53);
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        FloatArray whole = new FloatArray(m * n);
        FloatArray partial = new FloatArray(splits * m * n);
        FloatArray stored = new FloatArray(m * n);
        FloatArray residual = new FloatArray(m * n);
        for (int i = 0; i < m * n; i++) {
            residual.set(i, before.get(i));
        }
        TaskGraph g =
                new TaskGraph("q8s")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                a,
                                w,
                                q8,
                                dA,
                                whole,
                                partial,
                                stored,
                                residual)
                        .task(
                                "q",
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                new KernelContext(),
                                a,
                                q8,
                                dA,
                                k)
                        .task(
                                "w",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                whole,
                                whole,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_STORE,
                                whole,
                                1,
                                rowsOf(m),
                                n,
                                0)
                        .task(
                                "s",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                stored,
                                stored,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_STORE,
                                partial,
                                splits,
                                rowsOf(m),
                                n,
                                0)
                        .task(
                                "sr",
                                Int8GemmKernels::reduceSplitsQ8_0,
                                new KernelContext(),
                                partial,
                                stored,
                                m * n,
                                splits,
                                Int8GemmKernels.EPILOGUE_STORE,
                                rowsOf(m),
                                n)
                        .task(
                                "r",
                                Int8GemmKernels::gemmInt8Q8_0,
                                new KernelContext(),
                                q8,
                                dA,
                                w,
                                residual,
                                residual,
                                m,
                                n,
                                k,
                                Int8GemmKernels.EPILOGUE_RESIDUAL,
                                partial,
                                splits,
                                rowsOf(m),
                                n,
                                0)
                        .task(
                                "rr",
                                Int8GemmKernels::reduceSplitsQ8_0,
                                new KernelContext(),
                                partial,
                                residual,
                                m * n,
                                splits,
                                Int8GemmKernels.EPILOGUE_RESIDUAL,
                                rowsOf(m),
                                n)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, whole, stored, residual);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("q8s.q", lanes(m * k, 256));
        s.addWorkerGrid("q8s.w", gemmGrid(m, n));
        s.addWorkerGrid("q8s.s", gemmGrid(m, n, splits));
        s.addWorkerGrid("q8s.r", gemmGrid(m, n, splits));
        s.addWorkerGrid("q8s.sr", lanes(m * n, 256));
        s.addWorkerGrid("q8s.rr", lanes(m * n, 256));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        for (int i = 0; i < m * n; i++) {
            float expected = whole.get(i);
            assertEquals("stored " + i, expected, stored.get(i), 1e-5f * (1 + Math.abs(expected)));
            assertEquals(
                    "residual " + i,
                    before.get(i) + expected,
                    residual.get(i),
                    1e-5f * (1 + Math.abs(expected)));
        }
    }
}
