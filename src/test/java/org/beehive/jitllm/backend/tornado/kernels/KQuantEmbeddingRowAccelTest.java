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
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** The Q4_K and Q6_K embedding-row decodes on the device against the host decode. */
public class KQuantEmbeddingRowAccelTest {

    /** A Qwen3-0.6B embedding row: 1024 weights, four super-blocks. */
    private static final int BLOCKS = 4;

    private static final int N = BLOCKS * 256;

    private static ByteArray randomBlocks(int blockBytes, int scaleOffset, long seed) {
        byte[] raw = new byte[BLOCKS * blockBytes];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < BLOCKS; b++) {
            int base = b * blockBytes + scaleOffset;
            raw[base] = (byte) (b * 13);
            raw[base + 1] = (byte) (0x24 | (b & 1) << 7);
            if (scaleOffset == 0) {
                raw[base + 2] = (byte) (b * 7);
                raw[base + 3] = (byte) 0x20;
            }
        }
        return ByteArray.fromArray(raw);
    }

    private static FloatArray run(String name, ByteArray x, boolean q4k) {
        FloatArray out = new FloatArray(N);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, x, out);
        if (q4k) {
            graph.task("updateX", TransformerComputeKernels::convertQ4_KtoFP32, context, x, out);
        } else {
            graph.task("updateX", TransformerComputeKernels::convertQ6_KtoFP32, context, x, out);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D grid = new WorkerGrid1D(N);
        grid.setLocalWork(128, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler(name + ".updateX", grid)).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return out;
    }

    @Test
    public void q4kRowMatchesHost() {
        ByteArray x = randomBlocks(144, 0, 1);
        FloatArray out = run("q4kRow", x, true);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ4_K.decode(x, (i / 256) * 144, i % 256);
            assertEquals("element " + i, expected, out.get(i), 0f);
        }
    }

    @Test
    public void q6kRowMatchesHost() {
        ByteArray x = randomBlocks(210, 208, 2);
        FloatArray out = run("q6kRow", x, false);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ6_K.decode(x, (i / 256) * 210, i % 256);
            assertEquals("element " + i, expected, out.get(i), 0f);
        }
    }
}
