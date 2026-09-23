package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.function.BiFunction;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
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
 * Every native quantized matrix-vector kernel, <b>compiled and executed on a device</b>, against
 * the host tensor for the same bytes.
 *
 * <p>{@code QuantizedDeviceDecodeParityTest} settles the arithmetic by running each decode on the
 * host. It cannot settle two other things, and both have bitten this backend before:
 *
 * <ul>
 *   <li><b>Whether the kernel compiles at all.</b> TornadoVM's sketcher rejects constructs that are
 *       perfectly good Java — {@code ByteArray.getHalfFloat} in one decode produced "Unable to
 *       build sketch for method", which is why {@code TransformerComputeKernelsQ6_K} assembles its
 *       half from two byte loads. A kernel that only ever runs on the host would never find out.
 *   <li><b>Whether the row addressing survives a real launch.</b> One workgroup per row, a shared
 *       reduction and a block offset computed from the row index: an error there is invisible to a
 *       per-element decode test, which never indexes a row.
 * </ul>
 *
 * <p>The shape is rectangular and not a multiple of anything convenient, so a row/column
 * transposition or an assumption that the row length divides the block size fails here.
 */
public class NativeQuantizedKernelAccelTest {

    /**
     * Rows and columns of the probe matrix. Columns are a whole number of Q4_K/Q5_K/Q6_K blocks.
     */
    private static final int N = 512;

    private static final int D = 37;

    private static final int LOCAL = 128;

    private static final float POISON = -999999f;

    /** A device kernel that computes {@code out[row] = w[row]·x}. */
    private interface Probe {
        void run(TaskGraph graph, KernelContext context, FloatArray x, FloatArray out, ByteArray w);
    }

    private static byte[] randomBlocks(GGMLType type, long seed) {
        int blocks = D * (N / type.getBlockSize());
        byte[] raw = new byte[blocks * type.getTypeSize()];
        new Random(seed).nextBytes(raw);
        // Keep the block scales finite and modest: this test is about launching and addressing,
        // and a random fp16 scale can be Inf, whose products are not comparable term by term.
        for (int b = 0; b < blocks; b++) {
            int base = b * type.getTypeSize();
            int[] scaleOffsets = scaleOffsets(type);
            for (int offset : scaleOffsets) {
                // 0x2C00..0x2CFF is around 0.06: small, normal, and different per block.
                raw[base + offset] = (byte) (b & 0xFF);
                raw[base + offset + 1] = 0x2C;
            }
        }
        return raw;
    }

    private static int[] scaleOffsets(GGMLType type) {
        return switch (type) {
            case Q4_0, Q8_0 -> new int[] {0};
            case Q4_1, Q4_K, Q5_K -> new int[] {0, 2};
            case Q6_K -> new int[] {208};
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray array = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            array.set(i, raw[i]);
        }
        return array;
    }

    /**
     * Runs one representation's matrix-vector kernel on the device and compares it to the host
     * tensor's own dot product over the same bytes.
     */
    private static void assertKernelMatchesHost(
            String name,
            GGMLType type,
            BiFunction<Integer, MemorySegment, FloatTensor> hostTensor,
            Probe probe)
            throws Exception {
        byte[] raw = randomBlocks(type, 20260908L + name.hashCode());

        FloatArray x = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            x.set(i, ((i % 7) - 3) * 0.25f);
        }
        FloatArray out = new FloatArray(D);
        out.init(POISON);

        float[] expected = new float[D];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor host = hostTensor.apply(D * N, segment);
            FloatTensor activations = new ArrayFloatTensor(new float[N]);
            for (int i = 0; i < N; i++) {
                activations.setFloat(i, x.get(i));
            }
            for (int row = 0; row < D; row++) {
                float sum = 0f;
                for (int col = 0; col < N; col++) {
                    sum += host.getFloat(row * N + col) * activations.getFloat(col);
                }
                expected[row] = sum;
            }
        }

        ByteArray w = toDevice(raw);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph(name + "Probe")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, out);
        probe.run(graph, context, x, out, w);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(D * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + "Probe." + name, worker);

        ImmutableTaskGraph immutable = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
            plan.withGridScheduler(scheduler).execute();
        }

        for (int row = 0; row < D; row++) {
            assertTrue(
                    name + " row " + row + " was never written by the kernel",
                    out.get(row) != POISON);
            // The device sums in a different order from the host's row-major loop, so this is a
            // relative tolerance rather than bit equality. The decode itself is bit-exact and is
            // held to that by QuantizedDeviceDecodeParityTest.
            assertEquals(
                    name + " row " + row,
                    expected[row],
                    out.get(row),
                    Math.max(1e-3f, Math.abs(expected[row]) * 1e-4f));
        }
    }

    @Test
    public void q4_0MatrixVectorRunsOnTheDevice() throws Exception {
        assertKernelMatchesHost(
                "q4_0",
                GGMLType.Q4_0,
                (n, seg) -> new Q4_0FloatTensor(n, seg),
                (graph, context, x, out, w) ->
                        graph.task(
                                "q4_0",
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                                context,
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL));
    }

    @Test
    public void q4_1MatrixVectorRunsOnTheDevice() throws Exception {
        assertKernelMatchesHost(
                "q4_1",
                GGMLType.Q4_1,
                (n, seg) -> new Q4_1FloatTensor(n, seg),
                (graph, context, x, out, w) ->
                        graph.task(
                                "q4_1",
                                TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1,
                                context,
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL));
    }

    @Test
    public void q4_KMatrixVectorRunsOnTheDevice() throws Exception {
        assertKernelMatchesHost(
                "q4_K",
                GGMLType.Q4_K,
                (n, seg) -> new Q4_KFloatTensor(n, seg),
                (graph, context, x, out, w) ->
                        graph.task(
                                "q4_K",
                                TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                                context,
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL));
    }

    @Test
    public void q5_KMatrixVectorRunsOnTheDevice() throws Exception {
        assertKernelMatchesHost(
                "q5_K",
                GGMLType.Q5_K,
                (n, seg) -> new Q5_KFloatTensor(n, seg),
                (graph, context, x, out, w) ->
                        graph.task(
                                "q5_K",
                                TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K,
                                context,
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL));
    }

    @Test
    public void q6_KMatrixVectorRunsOnTheDevice() throws Exception {
        assertKernelMatchesHost(
                "q6_K",
                GGMLType.Q6_K,
                (n, seg) -> new Q6_KFloatTensor(n, seg),
                (graph, context, x, out, w) ->
                        graph.task(
                                "q6_K",
                                TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                                context,
                                x,
                                out,
                                w,
                                N,
                                D,
                                LOCAL));
    }

    // @formatter:off
    /**
     * The fused gate/up projection with SwiGLU, which is the feed-forward every {@code qwen35}
     * block runs and the only Q4_0 kernel that decodes two weight matrices in one pass.
     *
     * <p>Held against the host's own {@code SwiGLU} over two separate dot products, so a defect in
     * the shared local array — gate and up occupy one allocation and one reduction tree — shows up
     * as a wrong row rather than as a plausible activation.
     */
    // @formatter:on
    @Test
    public void theFusedQ4_0GateUpSwiGluRunsOnTheDevice() throws Exception {
        byte[] rawGate = randomBlocks(GGMLType.Q4_0, 991L);
        byte[] rawUp = randomBlocks(GGMLType.Q4_0, 4242L);

        FloatArray x = new FloatArray(N);
        for (int i = 0; i < N; i++) {
            x.set(i, ((i % 5) - 2) * 0.5f);
        }
        FloatArray out = new FloatArray(D);
        out.init(POISON);

        float[] expected = new float[D];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment gateSegment = arena.allocate(rawGate.length);
            MemorySegment.copy(rawGate, 0, gateSegment, ValueLayout.JAVA_BYTE, 0, rawGate.length);
            MemorySegment upSegment = arena.allocate(rawUp.length);
            MemorySegment.copy(rawUp, 0, upSegment, ValueLayout.JAVA_BYTE, 0, rawUp.length);
            FloatTensor gate = new Q4_0FloatTensor(D * N, gateSegment);
            FloatTensor up = new Q4_0FloatTensor(D * N, upSegment);
            for (int row = 0; row < D; row++) {
                float gateSum = 0f;
                float upSum = 0f;
                for (int col = 0; col < N; col++) {
                    gateSum += gate.getFloat(row * N + col) * x.get(col);
                    upSum += up.getFloat(row * N + col) * x.get(col);
                }
                FloatTensor g = new ArrayFloatTensor(new float[] {gateSum});
                FloatTensor u = new ArrayFloatTensor(new float[] {upSum});
                org.beehive.jitllm.inference.op.CpuOperations.swiGLU(g, u);
                expected[row] = g.getFloat(0);
            }
        }

        ByteArray gateWeights = toDevice(rawGate);
        ByteArray upWeights = toDevice(rawUp);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("ffnProbe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION, x, gateWeights, upWeights, out)
                        .task(
                                "ffn",
                                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0,
                                context,
                                x,
                                out,
                                gateWeights,
                                upWeights,
                                N,
                                D,
                                LOCAL)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(D * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("ffnProbe.ffn", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        for (int row = 0; row < D; row++) {
            assertTrue("row " + row + " was never written", out.get(row) != POISON);
            assertEquals(
                    "row " + row,
                    expected[row],
                    out.get(row),
                    Math.max(1e-3f, Math.abs(expected[row]) * 1e-4f));
        }
    }
}
