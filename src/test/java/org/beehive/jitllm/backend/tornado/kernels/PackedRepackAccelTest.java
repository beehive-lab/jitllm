package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertArrayEquals;

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

/**
 * The device repack ({@link PackedRepackKernels}) against the host packer, byte for byte: the
 * weight copied to a scratch buffer and repacked over itself, as the loader does on the GPU.
 */
public class PackedRepackAccelTest {

    private static final int[][] SHAPES = {{128, 64}, {384, 640}, {1024, 5120}};

    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static byte[] repackOnDevice(byte[] raw, int n, int k, boolean q4) throws Exception {
        ByteArray w = ByteArray.fromArray(raw);
        ByteArray scratch = new ByteArray(raw.length);
        int tiles = (n / 128) * (k / 64);
        String id = "rp" + (q4 ? "4" : "8") + n + "x" + k;
        TaskGraph g =
                new TaskGraph(id)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                        .task(
                                "c",
                                PackedRepackKernels::copy,
                                new KernelContext(),
                                w,
                                scratch,
                                raw.length);
        if (q4) {
            g.task("r", PackedRepackKernels::repackQ4_0, new KernelContext(), scratch, w, n, k);
        } else {
            g.task("r", PackedRepackKernels::repackQ8_0, new KernelContext(), scratch, w, n, k);
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, w);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid(id + ".c", grid((raw.length + 7) / 8));
        s.addWorkerGrid(
                id + ".r",
                grid(
                        tiles
                                * (q4
                                        ? PackedRepackKernels.Q4_0_LANES_PER_TILE
                                        : PackedRepackKernels.Q8_0_LANES_PER_TILE)));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        return w.toHeapArray();
    }

    @Test
    public void q8_0MatchesTheHostPacker() throws Exception {
        for (int[] shape : SHAPES) {
            byte[] raw = new byte[shape[0] * (shape[1] / 32) * 34];
            new Random(shape[0] + shape[1]).nextBytes(raw);
            byte[] expected =
                    Int8GemmKernels.packQ8_0TileBytes(ByteArray.fromArray(raw), shape[0], shape[1]);
            assertArrayEquals(
                    shape[0] + "x" + shape[1],
                    expected,
                    repackOnDevice(raw, shape[0], shape[1], false));
        }
    }

    @Test
    public void q4_0MatchesTheHostPacker() throws Exception {
        for (int[] shape : SHAPES) {
            byte[] raw = new byte[shape[0] * (shape[1] / 32) * 18];
            new Random(3L * shape[0] + shape[1]).nextBytes(raw);
            byte[] expected =
                    Int8GemmKernels.packQ4_0TileBytes(ByteArray.fromArray(raw), shape[0], shape[1]);
            assertArrayEquals(
                    shape[0] + "x" + shape[1],
                    expected,
                    repackOnDevice(raw, shape[0], shape[1], true));
        }
    }

    /**
     * The loader's arrangement: graph "owner" uploads the weight once and keeps it on the device; a
     * second graph of the same plan takes it with consumeFromDevice and repacks it in place; the
     * owner's next run reads the packed bytes without uploading again.
     */
    @Test
    public void aConsumingGraphRepacksTheOwnersBuffer() throws Exception {
        int n = 384, k = 640;
        byte[] raw = new byte[n * (k / 32) * 18];
        new Random(7).nextBytes(raw);
        byte[] expected = Int8GemmKernels.packQ4_0TileBytes(ByteArray.fromArray(raw), n, k);
        ByteArray w = ByteArray.fromArray(raw);
        ByteArray seen = new ByteArray(raw.length);
        ByteArray scratch = new ByteArray(raw.length);
        TaskGraph owner =
                new TaskGraph("owner")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                        .task(
                                "c",
                                PackedRepackKernels::copy,
                                new KernelContext(),
                                w,
                                seen,
                                raw.length)
                        .persistOnDevice(w)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, seen);
        TaskGraph repack =
                new TaskGraph("repack")
                        .consumeFromDevice("owner", w)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch)
                        .task(
                                "c",
                                PackedRepackKernels::copy,
                                new KernelContext(),
                                w,
                                scratch,
                                raw.length)
                        .task(
                                "r",
                                PackedRepackKernels::repackQ4_0,
                                new KernelContext(),
                                scratch,
                                w,
                                n,
                                k)
                        .persistOnDevice(w);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("owner.c", grid((raw.length + 7) / 8));
        s.addWorkerGrid("repack.c", grid((raw.length + 7) / 8));
        s.addWorkerGrid(
                "repack.r", grid((n / 128) * (k / 64) * PackedRepackKernels.Q4_0_LANES_PER_TILE));
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(owner.snapshot(), repack.snapshot())) {
            plan.withGridScheduler(s);
            plan.withGraph(0).execute();
            assertArrayEquals("before", raw, seen.toHeapArray());
            plan.withGraph(1).execute();
            plan.withGraph(0).execute();
            assertArrayEquals("after", expected, seen.toHeapArray());
            assertArrayEquals("host keeps the GGUF bytes", raw, w.toHeapArray());
        }
    }
}
