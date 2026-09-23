package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.function.BiFunction;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
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

// @formatter:off
/**
 * The tiled batch projections for the two representations a {@code qwen35} layer holds beyond Q4_0,
 * run on a device against the host tensor's own dot product over the same bytes.
 *
 * <p>A tile is a scheduling change: one workgroup covers eight prompt rows instead of one and
 * decodes each weight once for all of them. What that can get wrong is addressing — the tile index
 * folded into the group id, the activation row offset, a row past {@code activeRows} contributing
 * to a sum — and none of it is visible from a per-element decode test, which never indexes a row.
 *
 * <p>A partially filled tile is the case worth being deliberate about. The chunk width is fixed and
 * the last chunk of a prompt is rarely full, so widths on both sides of the tile boundary are
 * checked, and the rows past {@code activeRows} must be left exactly as they were found rather than
 * written with a partial sum.
 *
 * <p>Comparison is to a tolerance rather than exact. The device sums a row in a different order
 * from the host — lane by lane, then through a tree — so the difference is reassociation, and the
 * bound is relative to the magnitude of the result.
 */
// @formatter:on
public class BatchProjectionTilingAccelTest {

    /** Columns: a whole number of Q4_1 blocks (32) and Q5_K super-blocks (256). */
    private static final int N = 512;

    private static final int D = 37;

    private static final int LOCAL = 128;

    /** Batch widths the kernels launch, chosen on both sides of the eight-row tile boundary. */
    private static final int[] WIDTHS = {1, 2, 7, 8, 9, 16};

    /** What an untouched output slot holds, so a write past {@code activeRows} is visible. */
    private static final float UNTOUCHED = -12345.5f;

    private interface BatchKernel {
        void run(
                TaskGraph graph,
                KernelContext context,
                FloatArray xBatch,
                FloatArray outBatch,
                ByteArray w,
                int activeRows);
    }

    private static byte[] randomBlocks(GGMLType type, long seed) {
        int blocks = D * (N / type.getBlockSize());
        byte[] raw = new byte[blocks * type.getTypeSize()];
        new Random(seed).nextBytes(raw);
        // Keep the block scales finite and modest, as NativeQuantizedKernelAccelTest does: a
        // random fp16 scale can be Inf, whose products are not comparable term by term. Q4_1 and
        // Q5_K both carry two fp16 scales, at offsets 0 and 2.
        for (int b = 0; b < blocks; b++) {
            int base = b * type.getTypeSize();
            for (int offset : new int[] {0, 2}) {
                raw[base + offset] = (byte) (b & 0xFF);
                raw[base + offset + 1] = 0x2C; // ~0.06
            }
        }
        return raw;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray array = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            array.set(i, raw[i]);
        }
        return array;
    }

    private static float activation(int row, int i) {
        return (((i + 3 * row) % 7) - 3) * 0.25f;
    }

    private static FloatArray execute(
            String name,
            BatchKernel kernel,
            byte[] raw,
            int width,
            int activeRows,
            int tileRows,
            int tileCols)
            throws Exception {
        FloatArray x = new FloatArray(width * N);
        for (int r = 0; r < width; r++) {
            for (int i = 0; i < N; i++) {
                x.set(r * N + i, activation(r, i));
            }
        }
        FloatArray out = new FloatArray(width * D);
        out.init(UNTOUCHED);

        TaskGraph graph = new TaskGraph(name);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, x, toDevice(raw));
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, out);
        kernel.run(graph, new KernelContext(), x, out, toDevice(raw), activeRows);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        int rowGroups = (width + tileRows - 1) / tileRows;
        int colGroups = (D + tileCols - 1) / tileCols;
        WorkerGrid worker = new WorkerGrid1D(rowGroups * colGroups * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".projection", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        return out;
    }

    /** {@code out[b][row] = w[row]·x[b]} on the host, over the same bytes. */
    private static float[] hostReference(
            byte[] raw, BiFunction<Integer, MemorySegment, FloatTensor> tensor, int activeRows) {
        float[] expected = new float[activeRows * D];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor host = tensor.apply(D * N, segment);
            for (int row = 0; row < activeRows; row++) {
                for (int col = 0; col < D; col++) {
                    float sum = 0f;
                    for (int k = 0; k < N; k++) {
                        sum += host.getFloat(col * N + k) * activation(row, k);
                    }
                    expected[row * D + col] = sum;
                }
            }
        }
        return expected;
    }

    private static void assertTiledMatchesHost(
            String what,
            GGMLType type,
            BiFunction<Integer, MemorySegment, FloatTensor> tensor,
            int tileRows,
            int tileCols,
            BatchKernel plain,
            BatchKernel residual)
            throws Exception {
        byte[] raw = randomBlocks(type, 20260909L + what.hashCode());
        for (int width : WIDTHS) {
            for (int activeRows : width == 1 ? new int[] {1} : new int[] {1, width}) {
                float[] expected = hostReference(raw, tensor, activeRows);

                FloatArray got =
                        execute(what + "Plain", plain, raw, width, activeRows, tileRows, tileCols);
                FloatArray accumulated =
                        execute(
                                what + "Residual",
                                residual,
                                raw,
                                width,
                                activeRows,
                                tileRows,
                                tileCols);

                for (int row = 0; row < width; row++) {
                    for (int col = 0; col < D; col++) {
                        int i = row * D + col;
                        String where =
                                what
                                        + " width="
                                        + width
                                        + " active="
                                        + activeRows
                                        + " ["
                                        + row
                                        + "]["
                                        + col
                                        + "]";
                        if (row >= activeRows) {
                            assertEquals(where + " (inactive)", UNTOUCHED, got.get(i), 0.0f);
                            assertEquals(
                                    where + " (inactive)", UNTOUCHED, accumulated.get(i), 0.0f);
                            continue;
                        }
                        float want = expected[i];
                        float tolerance = Math.max(1e-3f, Math.abs(want) * 1e-4f);
                        assertEquals(where, want, got.get(i), tolerance);
                        assertEquals(
                                where + " (residual)",
                                UNTOUCHED + want,
                                accumulated.get(i),
                                tolerance);
                    }
                }
            }
        }
    }

    @Test
    public void theTiledQ5_KProjectionMatchesTheHost() throws Exception {
        assertTiledMatchesHost(
                "q5k",
                GGMLType.Q5_K,
                (n, seg) -> new Q5_KFloatTensor(n, seg),
                TransformerComputeKernelsQ5_K.rowTile(),
                TransformerComputeKernelsQ5_K.colTile(),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ5_K::matrixVectorTiledBatchQ5_K,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ5_K
                                        ::matrixVectorTiledBatchWithResidualQ5_K,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL));
    }

    // @formatter:off
    /**
     * Q4_0 carries the most of this family's weight and is where the output tile was tuned, so a
     * wrong output-row offset or a group id decomposed against the wrong divisor shows up here
     * first. The fused gate/up kernel tiles both axes the same way and is not held here: it has a
     * different signature and its own SwiGLU, and the batched prefill parity tests cover it end to
     * end against the CPU.
     */
    // @formatter:on
    @Test
    public void theTiledQ4_0ProjectionMatchesTheHost() throws Exception {
        assertTiledMatchesHost(
                "q40",
                GGMLType.Q4_0,
                (n, seg) -> new Q4_0FloatTensor(n, seg),
                TransformerComputeKernelsQ4_0.rowTile(),
                TransformerComputeKernelsQ4_0.colTile(),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ4_0::matrixVectorTiledBatchQ4_0,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ4_0
                                        ::matrixVectorTiledBatchWithResidualQ4_0,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL));
    }

    @Test
    public void theTiledQ4_1ProjectionMatchesTheHost() throws Exception {
        assertTiledMatchesHost(
                "q41",
                GGMLType.Q4_1,
                (n, seg) -> new Q4_1FloatTensor(n, seg),
                TransformerComputeKernelsQ4_1.rowTile(),
                TransformerComputeKernelsQ4_1.colTile(),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ4_1::matrixVectorTiledBatchQ4_1,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL),
                (g, c, x, out, w, active) ->
                        g.task(
                                "projection",
                                TransformerComputeKernelsQ4_1
                                        ::matrixVectorTiledBatchWithResidualQ4_1,
                                c,
                                x,
                                out,
                                w,
                                N,
                                D,
                                active,
                                LOCAL));
    }
}
