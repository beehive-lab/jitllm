package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The split of Phi-3's fused {@code [q | k | v]} row and the SwiGLU over its fused {@code [gate |
 * up]} row, on the device against the host.
 */
public class Phi3FusedRowsAccelTest {

    private static final int DIM = 384;
    private static final int KV_DIM = 128;
    private static final int HIDDEN = 512;
    private static final int ROWS = 3;

    private static FloatArray random(int n, long seed) {
        Random random = new Random(seed);
        FloatArray a = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, random.nextFloat() * 8f - 4f);
        }
        return a;
    }

    private static void run(TaskGraph graph, String task, int threads) {
        WorkerGrid1D grid = new WorkerGrid1D(threads);
        grid.setLocalWork(128, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler(task, grid)).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static float silu(float x) {
        return x / (1.0f + (float) Math.exp(-x));
    }

    @Test
    public void splitQkvCopiesEachPart() {
        FloatArray qkv = random(DIM + 2 * KV_DIM, 1);
        FloatArray q = new FloatArray(DIM);
        FloatArray k = new FloatArray(KV_DIM);
        FloatArray v = new FloatArray(KV_DIM);
        TaskGraph graph =
                new TaskGraph("split")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, qkv, q, k, v)
                        .task(
                                "t",
                                Phi3Kernels::splitQKV,
                                new KernelContext(),
                                qkv,
                                q,
                                k,
                                v,
                                DIM,
                                KV_DIM)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, q, k, v);
        run(graph, "split.t", DIM + 2 * KV_DIM);
        for (int i = 0; i < DIM; i++) {
            assertEquals("q " + i, qkv.get(i), q.get(i), 0f);
        }
        for (int i = 0; i < KV_DIM; i++) {
            assertEquals("k " + i, qkv.get(DIM + i), k.get(i), 0f);
            assertEquals("v " + i, qkv.get(DIM + KV_DIM + i), v.get(i), 0f);
        }
    }

    @Test
    public void siluGateUpMatchesHost() {
        FloatArray gateUp = random(2 * HIDDEN, 2);
        FloatArray hb = new FloatArray(HIDDEN);
        TaskGraph graph =
                new TaskGraph("glu")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, gateUp, hb)
                        .task("t", Phi3Kernels::siluGateUp, new KernelContext(), gateUp, hb, HIDDEN)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, "glu.t", HIDDEN);
        for (int i = 0; i < HIDDEN; i++) {
            float expected = silu(gateUp.get(i)) * gateUp.get(HIDDEN + i);
            assertEquals(
                    "element " + i, expected, hb.get(i), 1e-5f * Math.max(1f, Math.abs(expected)));
        }
    }

    @Test
    public void siluGateUpBatchMatchesHost() {
        FloatArray gateUp = random(ROWS * 2 * HIDDEN, 3);
        FloatArray hb = new FloatArray(ROWS * HIDDEN);
        TaskGraph graph =
                new TaskGraph("gluBatch")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, gateUp, hb)
                        .task(
                                "t",
                                Phi3Kernels::siluGateUpBatch,
                                new KernelContext(),
                                gateUp,
                                hb,
                                HIDDEN,
                                ROWS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        run(graph, "gluBatch.t", ROWS * HIDDEN);
        for (int b = 0; b < ROWS; b++) {
            for (int i = 0; i < HIDDEN; i++) {
                float expected =
                        silu(gateUp.get(b * 2 * HIDDEN + i))
                                * gateUp.get(b * 2 * HIDDEN + HIDDEN + i);
                assertEquals(
                        "row " + b + " element " + i,
                        expected,
                        hb.get(b * HIDDEN + i),
                        1e-5f * Math.max(1f, Math.abs(expected)));
            }
        }
    }
}
