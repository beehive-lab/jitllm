package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The {@code lfm2moe} batched-prefill kernels, <b>on the device</b>, against the host operations
 * the single-token path is held to.
 *
 * <p>A chunk of 37 rows, so the expert tiles are partial and the chunk is not a multiple of any
 * workgroup; eight experts of which each row picks two, so every expert gets a few rows and the
 * grouping, the gathered rows and the combine all matter. A grouped GEMM that read the wrong
 * expert, row or tile disagrees with the host by orders of magnitude.
 */
public class Lfm2BatchKernelAccelTest {

    private static final int DIM = 512;
    private static final int HIDDEN = 256;
    private static final int EXPERTS = 8;
    private static final int USED = 2;
    private static final int ROWS = 37;

    private final Random random = new Random(20261112L);

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private static FloatArray device(float[] values) {
        FloatArray array = new FloatArray(values.length);
        for (int i = 0; i < values.length; i++) {
            array.set(i, values[i]);
        }
        return array;
    }

    private static ByteArray device(byte[] raw) {
        ByteArray array = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            array.set(i, raw[i]);
        }
        return array;
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(local, 1, 1);
        return worker;
    }

    private static int blockBytes(DataType type) {
        return switch (type) {
            case Q4_K -> 144;
            case Q5_K -> 176;
            case Q6_K -> 210;
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /** Random K-quant blocks whose FP16 scales are small and finite. */
    private byte[] blocks(DataType type, int elements) {
        int size = blockBytes(type);
        byte[] raw = new byte[elements / 256 * size];
        random.nextBytes(raw);
        for (int b = 0; b < elements / 256; b++) {
            int base = b * size;
            if (type == DataType.Q6_K) {
                putHalf(raw, base + 208, 0x1C00 + random.nextInt(0x400));
            } else {
                putHalf(raw, base, 0x2000 + random.nextInt(0x400));
                putHalf(raw, base + 2, 0x1C00 + random.nextInt(0x400));
            }
        }
        return raw;
    }

    private static void putHalf(byte[] raw, int at, int bits) {
        raw[at] = (byte) (bits & 0xFF);
        raw[at + 1] = (byte) ((bits >> 8) & 0xFF);
    }

    private static FloatTensor host(DataType type, byte[] raw, int elements, Arena arena) {
        MemorySegment segment = arena.allocate(raw.length);
        MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
        return switch (type) {
            case Q4_K -> new Q4_KFloatTensor(elements, segment);
            case Q5_K -> new Q5_KFloatTensor(elements, segment);
            case Q6_K -> new Q6_KFloatTensor(elements, segment);
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    private static void run(TaskGraph graph, GridScheduler scheduler) {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new AssertionError("failed to build or execute on the device: " + e, e);
        }
    }

    // ── the short convolution ───────────────────────────────────────────────────

    /** The chunk's rows through {@code shortConvBatch} equal the host's row-at-a-time scan. */
    @Test
    public void shortConvBatchMatchesTheHostRowByRow() {
        int kernel = 3;
        float[] bcx = noise(ROWS * 3 * DIM, 1f);
        float[] taps = noise(DIM * kernel, 0.5f);
        float[] window = noise(DIM * (kernel - 1), 1f);

        IntArray info = batchInfo();
        FloatArray dBcx = device(bcx);
        FloatArray dTaps = device(taps);
        FloatArray dState = device(window);
        FloatArray out = new FloatArray(ROWS * DIM);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("conv")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                context,
                                info,
                                dBcx,
                                dTaps,
                                dState)
                        .task(
                                "scan",
                                Lfm2BatchKernels::shortConvBatch,
                                context,
                                info,
                                dBcx,
                                dTaps,
                                dState,
                                out,
                                DIM,
                                kernel,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out, dState);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("conv.scan", grid(DIM, 128));
        run(graph, scheduler);

        FloatTensor hTaps = new ArrayFloatTensor(taps);
        FloatTensor hWindow = new ArrayFloatTensor(window.clone());
        FloatTensor bx = ArrayFloatTensor.allocate(DIM);
        FloatTensor y = ArrayFloatTensor.allocate(DIM);
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < DIM; c++) {
                bx.setFloat(c, bcx[r * 3 * DIM + c] * bcx[r * 3 * DIM + 2 * DIM + c]);
            }
            CpuOperations.causalConv1d(bx, hTaps, hWindow, y, DIM, kernel);
            for (int c = 0; c < DIM; c++) {
                float expected = y.getFloat(c) * bcx[r * 3 * DIM + DIM + c];
                assertEquals("row " + r + " channel " + c, expected, out.get(r * DIM + c), 1e-5f);
            }
        }
        for (int i = 0; i < DIM * (kernel - 1); i++) {
            assertEquals("window " + i, hWindow.getFloat(i), dState.get(i), 0f);
        }
    }

    // ── the mixture of experts ──────────────────────────────────────────────────

    @Test
    public void mixtureQ4_K() {
        assertMixtureMatchesHost(DataType.Q4_K, DataType.Q4_K);
    }

    /** LFM2.5-24B-A2B's representations: Q4_K or Q5_K gate/up, Q5_K or Q6_K down. */
    @Test
    public void mixtureQ5_KGateUpQ6_KDown() {
        assertMixtureMatchesHost(DataType.Q5_K, DataType.Q6_K);
    }

    @Test
    public void mixtureQ4_KGateUpQ5_KDown() {
        assertMixtureMatchesHost(DataType.Q4_K, DataType.Q5_K);
    }

    private IntArray batchInfo() {
        IntArray info = new IntArray(3);
        info.set(0, 5);
        info.set(1, ROWS);
        info.set(2, 0);
        return info;
    }

    private void assertMixtureMatchesHost(DataType gateType, DataType downType) {
        assumeTrue(
                "no 32-lane shuffle on this device",
                SchedulerDetectionService.isSubgroupShuffle32Supported());
        int gateElements = EXPERTS * HIDDEN * DIM;
        byte[] gate = blocks(gateType, gateElements);
        byte[] up = blocks(gateType, gateElements);
        byte[] down = blocks(downType, gateElements);
        float[] router = noise(EXPERTS * DIM, 0.1f);
        float[] bias = noise(EXPERTS, 0.2f);
        float[] x = noise(ROWS * DIM, 1f);
        float[] residual = noise(ROWS * DIM, 1f);

        int assignments = ROWS * USED;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, EXPERTS);
        IntArray info = batchInfo();
        FloatArray dx = device(x);
        FloatArray dResidual = device(residual);
        FloatArray dRouter = device(router);
        FloatArray dBias = device(bias);
        ByteArray dGate = device(gate);
        ByteArray dUp = device(up);
        ByteArray dDown = device(down);
        FloatArray logits = new FloatArray(ROWS * EXPERTS);
        IntArray ids = new IntArray(assignments);
        FloatArray mix = new FloatArray(assignments);
        IntArray sorted = new IntArray(assignments);
        IntArray position = new IntArray(assignments);
        IntArray tiles = new IntArray(1 + 3 * maxTiles);
        FloatArray hidden = new FloatArray(assignments * HIDDEN);
        FloatArray hiddenUp = new FloatArray(assignments * HIDDEN);
        FloatArray expertOut = new FloatArray(assignments * DIM);

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                context,
                                info,
                                dx,
                                dRouter,
                                dBias,
                                dGate,
                                dUp,
                                dDown)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dResidual)
                        .task(
                                "router",
                                Lfm2BatchKernels::routerBatch,
                                context,
                                info,
                                dx,
                                dRouter,
                                logits,
                                DIM,
                                EXPERTS)
                        .task(
                                "select",
                                Lfm2BatchKernels::selectExpertsSigmoidBatch,
                                context,
                                info,
                                logits,
                                dBias,
                                ids,
                                mix,
                                EXPERTS,
                                USED)
                        .task(
                                "group",
                                Qwen35MoeBatchKernels::groupByExpert,
                                context,
                                ids,
                                info,
                                sorted,
                                position,
                                tiles,
                                EXPERTS,
                                USED,
                                ROWS)
                        .task(
                                "gate",
                                switch (gateType) {
                                    case Q4_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_K;
                                    case Q5_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_K;
                                    default -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_K;
                                },
                                context,
                                dx,
                                hidden,
                                dGate,
                                tiles,
                                sorted,
                                1,
                                DIM,
                                HIDDEN)
                        .task(
                                "up",
                                switch (gateType) {
                                    case Q4_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_K;
                                    case Q5_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_K;
                                    default -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_K;
                                },
                                context,
                                dx,
                                hiddenUp,
                                dUp,
                                tiles,
                                sorted,
                                1,
                                DIM,
                                HIDDEN)
                        .task(
                                "swiglu",
                                Lfm2BatchKernels::siluMultiplyAssignments,
                                context,
                                info,
                                hidden,
                                hiddenUp,
                                HIDDEN,
                                USED)
                        .task(
                                "down",
                                switch (downType) {
                                    case Q4_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_K;
                                    case Q5_K ->
                                            GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_K;
                                    default -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_K;
                                },
                                context,
                                hidden,
                                expertOut,
                                dDown,
                                tiles,
                                sorted,
                                0,
                                HIDDEN,
                                DIM)
                        .task(
                                "combine",
                                Lfm2BatchKernels::combineBatch,
                                context,
                                info,
                                expertOut,
                                position,
                                mix,
                                dResidual,
                                DIM,
                                USED)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dResidual, ids, mix);
        GridScheduler scheduler = new GridScheduler();
        int routerJobs = ROWS * EXPERTS;
        scheduler.addWorkerGrid(
                "moe.router",
                grid(
                        ((routerJobs + Lfm2BatchKernels.ROUTER_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.ROUTER_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.ROUTER_SIMDGROUPS,
                        32 * Lfm2BatchKernels.ROUTER_SIMDGROUPS));
        scheduler.addWorkerGrid(
                "moe.select",
                grid(
                        ((ROWS + Lfm2BatchKernels.SELECT_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.SELECT_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.SELECT_SIMDGROUPS,
                        32 * Lfm2BatchKernels.SELECT_SIMDGROUPS));
        scheduler.addWorkerGrid(
                "moe.group",
                grid(Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS));
        int threads = GroupedExpertSimdgroupKernels.THREADS;
        scheduler.addWorkerGrid("moe.gate", grid((HIDDEN / 64) * maxTiles * threads, threads));
        scheduler.addWorkerGrid("moe.up", grid((HIDDEN / 64) * maxTiles * threads, threads));
        scheduler.addWorkerGrid("moe.swiglu", grid(assignments * HIDDEN, 256));
        scheduler.addWorkerGrid("moe.down", grid((DIM / 64) * maxTiles * threads, threads));
        scheduler.addWorkerGrid("moe.combine", grid(ROWS * DIM, 256));
        run(graph, scheduler);

        try (Arena arena = Arena.ofShared()) {
            FloatTensor hGate = host(gateType, gate, gateElements, arena);
            FloatTensor hUp = host(gateType, up, gateElements, arena);
            FloatTensor hDown = host(downType, down, gateElements, arena);
            FloatTensor hRouter = new ArrayFloatTensor(router);
            FloatTensor hBias = new ArrayFloatTensor(bias);
            FloatTensor scores = ArrayFloatTensor.allocate(EXPERTS);
            FloatTensor h = ArrayFloatTensor.allocate(HIDDEN);
            FloatTensor hu = ArrayFloatTensor.allocate(HIDDEN);
            FloatTensor y = ArrayFloatTensor.allocate(DIM);
            int[] chosen = new int[USED];
            float[] weights = new float[USED];
            for (int t = 0; t < ROWS; t++) {
                float[] row = new float[DIM];
                System.arraycopy(x, t * DIM, row, 0, DIM);
                FloatTensor hx = new ArrayFloatTensor(row);
                CpuOperations.moeRouterSigmoid(
                        hx, hRouter, hBias, scores, chosen, weights, EXPERTS, USED, DIM);
                float[] stream = new float[DIM];
                System.arraycopy(residual, t * DIM, stream, 0, DIM);
                FloatTensor hs = new ArrayFloatTensor(stream);
                for (int j = 0; j < USED; j++) {
                    assertEquals("row " + t + " expert " + j, chosen[j], ids.get(t * USED + j));
                    assertEquals(
                            "row " + t + " weight " + j, weights[j], mix.get(t * USED + j), 1e-5f);
                    CpuOperations.expertFeedForward(
                            hx, chosen[j], hGate, hUp, hDown, h, hu, y, HIDDEN, DIM);
                    CpuOperations.weightedAccumulate(hs, y, weights[j], DIM);
                }
                float scale = 0f;
                for (int c = 0; c < DIM; c++) {
                    scale = Math.max(scale, Math.abs(hs.getFloat(c)));
                }
                for (int c = 0; c < DIM; c++) {
                    assertEquals(
                            gateType + "/" + downType + " row " + t + " column " + c,
                            hs.getFloat(c),
                            dResidual.get(t * DIM + c),
                            Math.max(1e-4f, scale * 1e-4f));
                }
            }
        }
    }
}
