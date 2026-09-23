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
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The workgroup-per-head norms of the batched plan against their one-lane forms: {@code
 * gatedNormPerHeadBatchGroup}, {@code l2NormPerHeadBatchGroup}, {@code fusedQKRmsNormBatchGroup}
 * and {@code rmsReduceBatchGroup}. Same inputs to both, raw-bit equality over every element,
 * including the rows past the active count (which neither form may touch), on random values with
 * the production head widths and row counts, plus a row of near-zero values (the epsilon path) and
 * a row of large values. A negative control (the group form with a tree fold) differs.
 */
// @formatter:on
public class Qwen35GroupNormBatchAccelTest {

    private static FloatArray random(int n, long seed, float scale) {
        FloatArray a = new FloatArray(n);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            a.set(i, (rng.nextFloat() * 2.0f - 1.0f) * scale);
        }
        return a;
    }

    /** Rows 0..active-1 random; row 1 near zero, row 2 large, the rest poison-like NaN. */
    private static FloatArray rows(int rows, int active, int width, long seed) {
        FloatArray a = random(rows * width, seed, 1.0f);
        for (int i = 0; i < width; i++) {
            a.set(1 * width + i, a.get(width + i) * 1e-20f);
            a.set(2 * width + i, a.get(2 * width + i) * 1e4f);
        }
        for (int r = active; r < rows; r++) {
            for (int i = 0; i < width; i++) {
                a.set(r * width + i, Float.NaN);
            }
        }
        return a;
    }

    private static IntArray batchInfo(int active) {
        IntArray info = new IntArray(4);
        info.set(0, 0);
        info.set(1, active);
        return info;
    }

    private static WorkerGrid lanes(int count, int local) {
        WorkerGrid g = new uk.ac.manchester.tornado.api.WorkerGrid1D(count);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static int mismatches(String what, FloatArray a, FloatArray b) {
        assertEquals(what + " sizes", a.getSize(), b.getSize());
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

    @Test
    public void theGatedNormIsRawBitEqualToTheOneLaneForm() throws Exception {
        int heads = 48;
        int headDim = 128;
        for (int[] rc : new int[][] {{256, 256}, {512, 300}, {2048, 2048}}) {
            int rows = rc[0];
            int active = rc[1];
            FloatArray v1 = rows(rows, active, heads * headDim, 1L);
            FloatArray v2 = rows(rows, active, heads * headDim, 1L);
            FloatArray gate = random(rows * heads * headDim, 2L, 4.0f);
            FloatArray weight = random(headDim, 3L, 2.0f);
            IntArray info = batchInfo(active);
            TaskGraph g =
                    new TaskGraph("gn")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, v1, v2, gate, weight, info)
                            .task(
                                    "one",
                                    Qwen35BatchKernels::gatedNormPerHeadBatch,
                                    new KernelContext(),
                                    v1,
                                    gate,
                                    weight,
                                    heads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .task(
                                    "group",
                                    Qwen35BatchKernels::gatedNormPerHeadBatchGroup,
                                    new KernelContext(),
                                    v2,
                                    gate,
                                    weight,
                                    heads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, v1, v2);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("gn.one", lanes(rows * heads, 1));
            s.addWorkerGrid("gn.group", lanes(rows * heads * headDim, headDim));
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).execute();
            }
            assertEquals("gated norm rows=" + rows, 0, mismatches("gated", v1, v2));
            assertTrue("gated norm changed nothing", !Float.isNaN(v2.get(0)) && v2.get(0) != 0.0f);
        }
    }

    @Test
    public void theL2NormIsRawBitEqualToTheOneLaneForm() throws Exception {
        int heads = 16;
        int headDim = 128;
        for (int[] rc : new int[][] {{256, 256}, {512, 300}, {2048, 2048}}) {
            int rows = rc[0];
            int active = rc[1];
            FloatArray v1 = rows(rows, active, heads * headDim, 4L);
            FloatArray v2 = rows(rows, active, heads * headDim, 4L);
            IntArray info = batchInfo(active);
            TaskGraph g =
                    new TaskGraph("l2")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, v1, v2, info)
                            .task(
                                    "one",
                                    Qwen35BatchKernels::l2NormPerHeadBatch,
                                    new KernelContext(),
                                    v1,
                                    heads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .task(
                                    "group",
                                    Qwen35BatchKernels::l2NormPerHeadBatchGroup,
                                    new KernelContext(),
                                    v2,
                                    heads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, v1, v2);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("l2.one", lanes(rows * heads, 1));
            s.addWorkerGrid("l2.group", lanes(rows * heads * headDim, headDim));
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).execute();
            }
            assertEquals("l2 norm rows=" + rows, 0, mismatches("l2", v1, v2));
        }
    }

    @Test
    public void theQKNormIsRawBitEqualToTheOneLaneForm() throws Exception {
        int heads = 24;
        int kvHeads = 4;
        int headDim = 256;
        for (int[] rc : new int[][] {{256, 256}, {512, 300}, {2048, 2048}}) {
            int rows = rc[0];
            int active = rc[1];
            FloatArray q1 = rows(rows, active, heads * headDim, 5L);
            FloatArray q2 = rows(rows, active, heads * headDim, 5L);
            FloatArray k1 = rows(rows, active, kvHeads * headDim, 6L);
            FloatArray k2 = rows(rows, active, kvHeads * headDim, 6L);
            FloatArray qw = random(headDim, 7L, 2.0f);
            FloatArray kw = random(headDim, 8L, 2.0f);
            IntArray info = batchInfo(active);
            TaskGraph g =
                    new TaskGraph("qk")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, q1, q2, k1, k2, qw, kw, info)
                            .task(
                                    "one",
                                    Qwen35BatchKernels::fusedQKRmsNormBatch,
                                    new KernelContext(),
                                    q1,
                                    k1,
                                    qw,
                                    kw,
                                    heads,
                                    kvHeads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .task(
                                    "group",
                                    Qwen35BatchKernels::fusedQKRmsNormBatchGroup,
                                    new KernelContext(),
                                    q2,
                                    k2,
                                    qw,
                                    kw,
                                    heads,
                                    kvHeads,
                                    headDim,
                                    1e-6f,
                                    info)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, q1, q2, k1, k2);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("qk.one", lanes(rows * (heads + kvHeads), 1));
            s.addWorkerGrid("qk.group", lanes(rows * (heads + kvHeads) * headDim, headDim));
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).execute();
            }
            assertEquals("qk norm queries rows=" + rows, 0, mismatches("q", q1, q2));
            assertEquals("qk norm keys rows=" + rows, 0, mismatches("k", k1, k2));
        }
    }

    @Test
    public void theRmsReductionIsRawBitEqualToTheOneLaneForm() throws Exception {
        int dim = 5120;
        for (int rows : new int[] {256, 512, 2048}) {
            FloatArray x = rows(rows, rows, dim, 9L);
            FloatArray s1 = new FloatArray(rows);
            FloatArray s2 = new FloatArray(rows);
            s1.init(Float.NaN);
            s2.init(Float.NaN);
            TaskGraph g =
                    new TaskGraph("rms")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, s1, s2)
                            .task(
                                    "one",
                                    TransformerBatchPrefillKernels::batchedRmsReduce,
                                    new KernelContext(),
                                    x,
                                    s1,
                                    dim,
                                    1e-6f)
                            .task(
                                    "group",
                                    Qwen35BatchKernels::rmsReduceBatchGroup,
                                    new KernelContext(),
                                    x,
                                    s2,
                                    dim,
                                    1e-6f)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, s1, s2);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("rms.one", lanes(rows, 1));
            s.addWorkerGrid(
                    "rms.group",
                    lanes(
                            rows * Qwen35BatchKernels.RMS_GROUP_LOCAL,
                            Qwen35BatchKernels.RMS_GROUP_LOCAL));
            try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
                p.withGridScheduler(s).execute();
            }
            for (int r = 0; r < rows; r++) {
                assertTrue("scale finite " + r, !Float.isNaN(s1.get(r)));
            }
            assertEquals("rms scale rows=" + rows, 0, mismatches("rms", s1, s2));
        }
    }

    @Test
    public void aTreeFoldDiffers() throws Exception {
        int heads = 16;
        int headDim = 128;
        int rows = 256;
        FloatArray v1 = rows(rows, rows, heads * headDim, 4L);
        FloatArray v2 = rows(rows, rows, heads * headDim, 4L);
        IntArray info = batchInfo(rows);
        TaskGraph g =
                new TaskGraph("neg")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, v1, v2, info)
                        .task(
                                "one",
                                Qwen35BatchKernels::l2NormPerHeadBatch,
                                new KernelContext(),
                                v1,
                                heads,
                                headDim,
                                1e-6f,
                                info)
                        .task(
                                "tree",
                                Qwen35GroupNormBatchAccelTest::l2NormTree,
                                new KernelContext(),
                                v2,
                                heads,
                                headDim,
                                1e-6f,
                                info)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, v1, v2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("neg.one", lanes(rows * heads, 1));
        s.addWorkerGrid("neg.tree", lanes(rows * heads * headDim, headDim));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        assertTrue("the tree fold agreed everywhere", mismatches("tree", v1, v2) > 0);
    }

    /** The group L2 norm with a shared tree instead of the left fold: the negative control. */
    public static void l2NormTree(
            KernelContext context,
            FloatArray values,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int group = context.groupIdx;
        if (group >= heads * batchInfo.get(1)) {
            return;
        }
        int lane = context.localIdx;
        int index = group * headDim + lane;
        float[] shared = context.allocateFloatLocalArray(headDim);
        float v = values.get(index);
        shared[lane] = v * v;
        context.localBarrier();
        for (int stride = headDim >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = shared[lane] + shared[lane + stride];
            }
            context.localBarrier();
        }
        float inv =
                1.0f
                        / uk.ac.manchester.tornado.api.math.TornadoMath.max(
                                uk.ac.manchester.tornado.api.math.TornadoMath.sqrt(shared[0]), eps);
        values.set(index, v * inv);
    }
}
