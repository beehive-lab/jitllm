package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q8_0FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The {@code lfm2moe} device kernels, <b>on the device</b>, against the host operations.
 *
 * <p>Small shapes that still exercise the addressing that matters: six stacked experts of which two
 * are selected — neither the first nor in index order — so an expert kernel that ignored the
 * selection, or read the wrong expert's rows, disagrees with the host. Rows are two super-blocks
 * wide for the gate/up projections and one for the down projection.
 */
public class Lfm2KernelAccelTest {

    private static final int DIM = 512;
    private static final int EXPERT_HIDDEN = 256;
    private static final int EXPERTS = 6;
    private static final int USED = 2;
    private static final int LOCAL = 128;

    private final Random random = new Random(20261014L);

    private static FloatArray device(float[] values) {
        FloatArray array = new FloatArray(values.length);
        for (int i = 0; i < values.length; i++) {
            array.set(i, values[i]);
        }
        return array;
    }

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private static void run(TaskGraph graph, GridScheduler scheduler) {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new AssertionError("failed to build or execute on the device: " + e, e);
        }
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(local, 1, 1);
        return worker;
    }

    // ── quantized weights with sane scales ─────────────────────────────────────

    private static int blockBytes(DataType type) {
        return switch (type) {
            case Q4_K -> 144;
            case Q5_K -> 176;
            case Q6_K -> 210;
            case Q8_0 -> 34;
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    private static int blockWeights(DataType type) {
        return type == DataType.Q8_0 ? 32 : 256;
    }

    /** Random blocks whose FP16 scales are small and finite, so every weight is well-formed. */
    private byte[] blocks(DataType type, int elements) {
        int count = elements / blockWeights(type);
        int size = blockBytes(type);
        byte[] raw = new byte[count * size];
        random.nextBytes(raw);
        for (int b = 0; b < count; b++) {
            int base = b * size;
            switch (type) {
                case Q4_K, Q5_K -> {
                    putHalf(raw, base, 0x2000 + random.nextInt(0x400)); // d ~ 0.008
                    putHalf(raw, base + 2, 0x1C00 + random.nextInt(0x400)); // dmin ~ 0.004
                }
                case Q6_K -> putHalf(raw, base + 208, 0x1C00 + random.nextInt(0x400));
                case Q8_0 -> putHalf(raw, base, 0x1C00 + random.nextInt(0x400));
                default -> throw new IllegalArgumentException(type.toString());
            }
        }
        return raw;
    }

    private static void putHalf(byte[] raw, int at, int bits) {
        raw[at] = (byte) (bits & 0xFF);
        raw[at + 1] = (byte) ((bits >> 8) & 0xFF);
    }

    private static ByteArray device(byte[] raw) {
        ByteArray array = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            array.set(i, raw[i]);
        }
        return array;
    }

    private static FloatTensor host(DataType type, byte[] raw, int elements, Arena arena) {
        MemorySegment segment = arena.allocate(raw.length);
        MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
        return switch (type) {
            case Q4_K -> new Q4_KFloatTensor(elements, segment);
            case Q5_K -> new Q5_KFloatTensor(elements, segment);
            case Q6_K -> new Q6_KFloatTensor(elements, segment);
            case Q8_0 -> new Q8_0FloatTensor(elements, segment);
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    // ── tests ───────────────────────────────────────────────────────────────────

    @Test
    public void expertsQ4_K() {
        assertExpertsMatchHost(DataType.Q4_K);
    }

    @Test
    public void expertsQ5_K() {
        assertExpertsMatchHost(DataType.Q5_K);
    }

    @Test
    public void expertsQ6_K() {
        assertExpertsMatchHost(DataType.Q6_K);
    }

    @Test
    public void expertsQ8_0() {
        assertExpertsMatchHost(DataType.Q8_0);
    }

    private void assertExpertsMatchHost(DataType type) {
        int gateElements = EXPERTS * EXPERT_HIDDEN * DIM;
        byte[] gate = blocks(type, gateElements);
        byte[] up = blocks(type, gateElements);
        byte[] down = blocks(type, gateElements);
        float[] x = noise(DIM, 1f);
        float[] residual = noise(DIM, 1f);
        int[] chosen = {4, 1};
        float[] mix = {0.7f, 0.3f};

        FloatArray dx = device(x);
        FloatArray dResidual = device(residual);
        FloatArray hb = new FloatArray(USED * EXPERT_HIDDEN);
        IntArray selected = new IntArray(USED);
        FloatArray weights = device(mix);
        for (int j = 0; j < USED; j++) {
            selected.set(j, chosen[j]);
        }
        ByteArray dGate = device(gate);
        ByteArray dUp = device(up);
        ByteArray dDown = device(down);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                context,
                                dx,
                                dGate,
                                dUp,
                                dDown,
                                selected,
                                weights,
                                hb)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dResidual);
        switch (type) {
            case Q4_K -> {
                graph.task(
                        "gateUp",
                        Lfm2Kernels::expertGateUpSiLUQ4_K,
                        context,
                        dx,
                        hb,
                        dGate,
                        dUp,
                        selected,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
                graph.task(
                        "down",
                        Lfm2Kernels::expertDownResidualQ4_K,
                        context,
                        hb,
                        dResidual,
                        dDown,
                        selected,
                        weights,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
            }
            case Q5_K -> {
                graph.task(
                        "gateUp",
                        Lfm2Kernels::expertGateUpSiLUQ5_K,
                        context,
                        dx,
                        hb,
                        dGate,
                        dUp,
                        selected,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
                graph.task(
                        "down",
                        Lfm2Kernels::expertDownResidualQ5_K,
                        context,
                        hb,
                        dResidual,
                        dDown,
                        selected,
                        weights,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
            }
            case Q6_K -> {
                graph.task(
                        "gateUp",
                        Lfm2Kernels::expertGateUpSiLUQ6_K,
                        context,
                        dx,
                        hb,
                        dGate,
                        dUp,
                        selected,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
                graph.task(
                        "down",
                        Lfm2Kernels::expertDownResidualQ6_K,
                        context,
                        hb,
                        dResidual,
                        dDown,
                        selected,
                        weights,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
            }
            case Q8_0 -> {
                graph.task(
                        "gateUp",
                        Lfm2Kernels::expertGateUpSiLUQ8_0,
                        context,
                        dx,
                        hb,
                        dGate,
                        dUp,
                        selected,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
                graph.task(
                        "down",
                        Lfm2Kernels::expertDownResidualQ8_0,
                        context,
                        hb,
                        dResidual,
                        dDown,
                        selected,
                        weights,
                        DIM,
                        EXPERT_HIDDEN,
                        USED,
                        LOCAL);
            }
            default -> throw new IllegalArgumentException(type.toString());
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, dResidual);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("moe.gateUp", grid(USED * EXPERT_HIDDEN * LOCAL, LOCAL));
        scheduler.addWorkerGrid("moe.down", grid(DIM * LOCAL, LOCAL));
        run(graph, scheduler);

        // Shared: the host expert projection runs its rows on several threads.
        try (Arena arena = Arena.ofShared()) {
            FloatTensor hGate = host(type, gate, gateElements, arena);
            FloatTensor hUp = host(type, up, gateElements, arena);
            FloatTensor hDown = host(type, down, gateElements, arena);
            FloatTensor hx = new ArrayFloatTensor(x.clone());
            FloatTensor stream = new ArrayFloatTensor(residual.clone());
            FloatTensor hidden = ArrayFloatTensor.allocate(EXPERT_HIDDEN);
            FloatTensor hiddenUp = ArrayFloatTensor.allocate(EXPERT_HIDDEN);
            FloatTensor y = ArrayFloatTensor.allocate(DIM);
            for (int j = 0; j < USED; j++) {
                CpuOperations.expertFeedForward(
                        hx, chosen[j], hGate, hUp, hDown, hidden, hiddenUp, y, EXPERT_HIDDEN, DIM);
                CpuOperations.weightedAccumulate(stream, y, mix[j], DIM);
            }
            // Both sides accumulate in float, in different orders, over sums with large
            // cancellations; the bound is relative to the output's own scale rather than to each
            // element, which an indexing error (a wrong expert or row) exceeds by orders.
            float scale = 0f;
            for (int i = 0; i < DIM; i++) {
                scale = Math.max(scale, Math.abs(stream.getFloat(i)));
            }
            for (int i = 0; i < DIM; i++) {
                assertEquals(
                        type + " expert output[" + i + "]",
                        stream.getFloat(i),
                        dResidual.get(i),
                        Math.max(1e-4f, scale * 1e-4f));
            }
        }
    }

    @Test
    public void shortConvMatchesTheHost() {
        int kernel = 3;
        int layers = 2;
        int layer = 1;
        int stateSize = (kernel - 1) * DIM;
        float[] bcx = noise(3 * DIM, 1f);
        float[] taps = noise(DIM * kernel, 0.5f);
        float[] window = noise(layers * stateSize, 1f);

        FloatArray dState = device(window);
        FloatArray out = new FloatArray(DIM);
        KernelContext context = new KernelContext();
        TaskGraph graph = new TaskGraph("conv");
        FloatArray inBcx = device(bcx);
        FloatArray inTaps = device(taps);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, context, inBcx, inTaps, dState)
                .task(
                        "conv",
                        Lfm2Kernels::shortConv,
                        context,
                        inBcx,
                        inTaps,
                        dState,
                        out,
                        DIM,
                        kernel,
                        layer * stateSize)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out, dState);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("conv.conv", grid(DIM, 128));
        run(graph, scheduler);

        FloatTensor bx = ArrayFloatTensor.allocate(DIM);
        for (int c = 0; c < DIM; c++) {
            bx.setFloat(c, bcx[c] * bcx[2 * DIM + c]);
        }
        float[] layerWindow = new float[stateSize];
        System.arraycopy(window, layer * stateSize, layerWindow, 0, stateSize);
        FloatTensor hostWindow = new ArrayFloatTensor(layerWindow);
        FloatTensor conv = ArrayFloatTensor.allocate(DIM);
        CpuOperations.causalConv1d(bx, new ArrayFloatTensor(taps), hostWindow, conv, DIM, kernel);
        for (int c = 0; c < DIM; c++) {
            assertEquals("conv[" + c + "]", bcx[DIM + c] * conv.getFloat(c), out.get(c), 1e-5f);
        }
        for (int i = 0; i < stateSize; i++) {
            assertEquals(
                    "window[" + i + "]",
                    hostWindow.getFloat(i),
                    dState.get(layer * stateSize + i),
                    0f);
            // The other layer's slice is untouched.
            assertEquals("other window[" + i + "]", window[i], dState.get(i), 0f);
        }
    }

    @Test
    public void routerSelectionMatchesTheHost() {
        int experts = 64;
        int used = 4;
        float[] logits = noise(experts, 2f);
        float[] bias = noise(experts, 0.5f);

        FloatArray dLogits = device(logits);
        IntArray selected = new IntArray(used);
        FloatArray weights = new FloatArray(used);
        KernelContext context = new KernelContext();
        FloatArray dBias = device(bias);
        TaskGraph graph =
                new TaskGraph("router")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, dLogits, dBias)
                        .task(
                                "select",
                                Lfm2Kernels::selectExpertsSigmoid,
                                context,
                                dLogits,
                                dBias,
                                selected,
                                weights,
                                experts,
                                used)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, selected, weights);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("router.select", grid(1, 1));
        run(graph, scheduler);

        // The host router takes the projection; an identity router makes its input the logits.
        FloatTensor identity = ArrayFloatTensor.allocate(experts * experts);
        for (int e = 0; e < experts; e++) {
            identity.setFloat(e * experts + e, 1f);
        }
        int[] ids = new int[used];
        float[] mix = new float[used];
        CpuOperations.moeRouterSigmoid(
                new ArrayFloatTensor(logits),
                identity,
                new ArrayFloatTensor(bias),
                ArrayFloatTensor.allocate(experts),
                ids,
                mix,
                experts,
                used,
                experts);
        for (int j = 0; j < used; j++) {
            assertEquals("expert " + j, ids[j], selected.get(j));
            assertEquals("weight " + j, mix[j], weights.get(j), 1e-6f);
        }
    }
}
