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
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The Q4_K, Q6_K and packed Q4_0 to FP16 decodes on the device against the host decode, element by
 * element.
 */
public class KQuantDequantFP16AccelTest {

    private static final int BLOCKS = 8;

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
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static HalfFloatArray run(String name, ByteArray w, boolean q4k) {
        HalfFloatArray out = new HalfFloatArray(N);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, out);
        if (q4k) {
            graph.task(
                    "decode",
                    TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16,
                    context,
                    w,
                    out,
                    0,
                    N);
        } else {
            graph.task(
                    "decode",
                    TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                    context,
                    w,
                    out,
                    0,
                    N);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D grid = new WorkerGrid1D(N);
        grid.setLocalWork(256, 1, 1);
        GridScheduler scheduler = new GridScheduler(name + ".decode", grid);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return out;
    }

    @Test
    public void q4kMatchesHost() {
        ByteArray w = randomBlocks(144, 0, 1);
        HalfFloatArray out = run("q4k", w, true);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ4_K.decode(w, (i / 256) * 144, i % 256);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }

    @Test
    public void q6kMatchesHost() {
        ByteArray w = randomBlocks(210, 208, 2);
        HalfFloatArray out = run("q6k", w, false);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ6_K.decode(w, (i / 256) * 210, i % 256);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }

    @Test
    public void packedQ4_0MatchesHost() {
        int rows = 256;
        int cols = 128;
        byte[] raw = new byte[rows * (cols / 32) * 18];
        new Random(3).nextBytes(raw);
        for (int b = 0; b < rows * cols / 32; b++) {
            raw[b * 18] = (byte) (b * 11);
            raw[b * 18 + 1] = (byte) (0x24 | (b & 1) << 7);
        }
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        ByteArray packed = ByteArray.fromArray(PackedTilePacker.packQ4_0(w, rows, cols));
        int n = rows * cols;
        HalfFloatArray out = new HalfFloatArray(n);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("q4p")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, packed, out)
                        .task(
                                "decode",
                                TransformerComputeKernelsQ4_0::dequantizeQ4_0PackedToFP16,
                                context,
                                packed,
                                out,
                                0,
                                n,
                                cols)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("q4p.decode", grid)).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < n; i++) {
            float expected = TransformerComputeKernelsQ4_0.decode(w, (i / 32) * 18, i % 32);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }
}
