package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The K-quant matvec kernels, <b>on the device</b>, against the host tensors.
 *
 * <p>{@code Q4_KDecodeTest} checks the decode arithmetic in isolation on the host. This checks the
 * thing that arithmetic is for: that the kernel compiles for the device at all, and that a whole
 * row's dot product comes back correct after the reduction.
 *
 * <p>Written because a full 24B model run is a four-minute round trip and tells you only that
 * something is wrong. This runs one small matrix and says which kernel, in seconds — it is how the
 * Q6_K sketch failure was found and fixed rather than guessed at.
 *
 * <p>Sizes are one super-block per row ({@code n = 256}) and a handful of rows, so the weights are
 * a few KB and the comparison is exact enough to catch an indexing error in any lane.
 */
public class KQuantKernelAccelTest {

    private static final int QK_K = 256;
    private static final int Q4_K_BLOCK_BYTES = 144;
    private static final int Q6_K_BLOCK_BYTES = 210;
    private static final int ROWS = 4;
    private static final int N = QK_K * 2;
    private static final int LOCAL = 128;

    @Test
    public void q4_KMatvecOnTheDeviceMatchesTheHost() {
        assertMatvecMatchesHost(true);
    }

    @Test
    public void q6_KMatvecOnTheDeviceMatchesTheHost() {
        assertMatvecMatchesHost(false);
    }

    @Test
    public void q6_KWarpMatvecOnTheDeviceMatchesTheHost() {
        assertWarpQ6_KMatchesHost(2048);
    }

    /** Sixteen units a row: half of each warp has no unit to take. */
    @Test
    public void q6_KWarpMatvecWithFewerUnitsThanLanesMatchesTheHost() {
        assertWarpQ6_KMatchesHost(512);
    }

    /**
     * {@code TransformerComputeKernelsQ6_K.matrixVectorQ6_KWarp} over thirteen rows, so the last
     * workgroup holds rows past the matrix; the block scales are kept small and finite.
     */
    private void assertWarpQ6_KMatchesHost(int n) {
        int rows = 13;
        int blocksPerRow = n / QK_K;
        byte[] raw = new byte[rows * blocksPerRow * Q6_K_BLOCK_BYTES];
        Random rng = new Random(6060L + n);
        rng.nextBytes(raw);
        for (int b = 0; b < rows * blocksPerRow; b++) {
            int bits = 0x2000 + rng.nextInt(0x400);
            raw[b * Q6_K_BLOCK_BYTES + 208] = (byte) (bits & 0xFF);
            raw[b * Q6_K_BLOCK_BYTES + 209] = (byte) (bits >> 8);
        }
        ByteArray weights = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            weights.set(i, raw[i]);
        }
        FloatArray x = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            x.set(i, rng.nextFloat() * 2 - 1);
        }
        FloatArray out = new FloatArray(rows);
        out.init(0.0f);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("q6warp")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, weights, x)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, out)
                        .task(
                                "matvec",
                                TransformerComputeKernelsQ6_K::matrixVectorQ6_KWarp,
                                context,
                                x,
                                out,
                                weights,
                                n,
                                rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        int groups =
                (rows + TransformerComputeKernelsQ6_K.WARP_ROWS_PER_GROUP - 1)
                        / TransformerComputeKernelsQ6_K.WARP_ROWS_PER_GROUP;
        WorkerGrid worker = new WorkerGrid1D(groups * 256);
        worker.setLocalWork(256, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("q6warp.matvec", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new AssertionError("Q6_K warp matvec failed to build or execute: " + e, e);
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);
            var host = new Q6_KFloatTensor(rows * n, segment);
            for (int row = 0; row < rows; row++) {
                double expected = 0;
                for (int j = 0; j < n; j++) {
                    expected += (double) host.getFloat(row * n + j) * x.get(j);
                }
                assertEquals(
                        "row " + row + " of the Q6_K warp matvec at n = " + n,
                        expected,
                        out.get(row),
                        Math.max(1e-3, Math.abs(expected) * 1e-4));
            }
        }
    }

    private void assertMatvecMatchesHost(boolean q4k) {
        int blockBytes = q4k ? Q4_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
        int blocksPerRow = N / QK_K;
        byte[] raw = new byte[ROWS * blocksPerRow * blockBytes];
        new Random(q4k ? 4004L : 6006L).nextBytes(raw);

        ByteArray weights = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            weights.set(i, raw[i]);
        }

        FloatArray x = new FloatArray(N);
        Random rng = new Random(99L);
        for (int i = 0; i < N; i++) {
            x.set(i, rng.nextFloat() * 2 - 1);
        }
        FloatArray out = new FloatArray(ROWS);
        out.init(0.0f);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("kquant")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, weights, x)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, out);
        if (q4k) {
            graph.task(
                    "matvec",
                    TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                    context,
                    x,
                    out,
                    weights,
                    N,
                    ROWS,
                    LOCAL);
        } else {
            graph.task(
                    "matvec",
                    TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                    context,
                    x,
                    out,
                    weights,
                    N,
                    ROWS,
                    LOCAL);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(ROWS * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("kquant.matvec", worker);

        ImmutableTaskGraph immutable = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new AssertionError(
                    (q4k ? "Q4_K" : "Q6_K")
                            + " matvec failed to build or execute on the device: "
                            + e,
                    e);
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);
            int elements = ROWS * N;
            var host = q4k ? new Q4_KFloatTensor(elements, segment) : null;
            var host6 = q4k ? null : new Q6_KFloatTensor(elements, segment);

            for (int row = 0; row < ROWS; row++) {
                double expected = 0;
                for (int j = 0; j < N; j++) {
                    float w = q4k ? host.getFloat(row * N + j) : host6.getFloat(row * N + j);
                    expected += (double) w * x.get(j);
                }
                // Float accumulation in a different order on each side, so compared with a
                // tolerance relative to the row's own magnitude rather than bit-for-bit.
                assertEquals(
                        "row " + row + " of the " + (q4k ? "Q4_K" : "Q6_K") + " matvec",
                        expected,
                        out.get(row),
                        Math.max(1e-3, Math.abs(expected) * 1e-4));
            }
        }
    }
}
