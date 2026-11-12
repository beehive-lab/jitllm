package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q8_0FloatTensor;
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
 * Qwen2-MoE's routed-expert projections for a layer that keeps the file's own representations,
 * <b>compiled and executed on a device</b>, against the host tensor for the same bytes: the
 * single-token {@code Qwen2MoEKernels.routedExpertsMatvec*} and the batched {@code
 * Qwen2MoEBatchKernels.groupedRoutedExpertsMatvec*}, for every type the loader keeps, and the two
 * type-free kernels that combine their outputs.
 *
 * <p>The experts are stacked as the file stores them: expert {@code e}'s row {@code r} is row
 * {@code e * ROWS + r}. Each slot or grouped position reads its own expert at its own activation
 * offset, so a wrong row of the stack, a wrong slot window or a wrong token shows up as a wrong
 * output; a slot or position the kernel must skip has to keep its poison.
 */
public class Qwen2MoEExpertKernelsAccelTest {

    /** Columns: a whole number of K-quant super-blocks. */
    private static final int N = 512;

    private static final int ROWS = 24;

    private static final int EXPERTS = 5;

    private static final int LOCAL = 32;

    private static final float POISON = -999999f;

    private interface HostTensor {
        FloatTensor of(int size, MemorySegment segment);
    }

    private interface SingleKernel {
        void add(
                TaskGraph graph,
                KernelContext context,
                FloatArray x,
                IntArray selected,
                ByteArray w,
                FloatArray out);
    }

    private interface GroupedKernel {
        void add(
                TaskGraph graph,
                KernelContext context,
                FloatArray x,
                int byToken,
                IntArray selected,
                IntArray grouped,
                IntArray active,
                ByteArray w,
                FloatArray out);
    }

    private static byte[] randomExperts(GGMLType type, long seed) {
        int blocks = EXPERTS * ROWS * (N / type.getBlockSize());
        byte[] raw = new byte[blocks * type.getTypeSize()];
        new Random(seed).nextBytes(raw);
        // Finite, modest block scales (0x2C00..0x2CFF, around 0.06), different per block.
        int[] scaleOffsets =
                switch (type) {
                    case Q4_0, Q8_0 -> new int[] {0};
                    case Q4_1, Q4_K, Q5_K -> new int[] {0, 2};
                    case Q6_K -> new int[] {208};
                    default -> throw new IllegalArgumentException(type.toString());
                };
        for (int b = 0; b < blocks; b++) {
            for (int offset : scaleOffsets) {
                raw[b * type.getTypeSize() + offset] = (byte) (b & 0xFF);
                raw[b * type.getTypeSize() + offset + 1] = 0x2C;
            }
        }
        return raw;
    }

    private static HostTensor host(GGMLType type) {
        return switch (type) {
            case Q8_0 -> Q8_0FloatTensor::new;
            case Q4_0 -> Q4_0FloatTensor::new;
            case Q4_1 -> Q4_1FloatTensor::new;
            case Q4_K -> Q4_KFloatTensor::new;
            case Q5_K -> Q5_KFloatTensor::new;
            case Q6_K -> Q6_KFloatTensor::new;
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

    private static float hostDot(FloatTensor w, int row, FloatArray x, int xOffset) {
        float sum = 0f;
        for (int col = 0; col < N; col++) {
            sum += w.getFloat(row * N + col) * x.get(xOffset + col);
        }
        return sum;
    }

    private static void assertClose(String what, float expected, float actual) {
        assertEquals(what, expected, actual, Math.max(1e-3f, Math.abs(expected) * 1e-4f));
    }

    private static WorkerGrid rowsWorker(int groups) {
        WorkerGrid worker = new WorkerGrid1D(groups * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        return worker;
    }

    /**
     * Three slots, each its own hidden window ({@code xStride = N}), the middle one selecting an
     * expert out of range: the outer two match the host's rows of their experts, and the middle one
     * is never written.
     */
    private static void checkSingleToken(GGMLType type, SingleKernel kernel) throws Exception {
        byte[] raw = randomExperts(type, 20261028L + type.ordinal());
        int slots = 3;
        IntArray selected = IntArray.fromElements(3, EXPERTS, 1);
        FloatArray x = new FloatArray(slots * N);
        for (int i = 0; i < slots * N; i++) {
            x.set(i, ((i % 7) - 3) * 0.25f + (i / N) * 0.125f);
        }
        FloatArray out = new FloatArray(slots * ROWS);
        out.init(POISON);

        ByteArray w = toDevice(raw);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, selected, w, out);
        kernel.add(graph, context, x, selected, w, out);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("moe.routed", rowsWorker(slots * ROWS));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor experts = host(type).of(EXPERTS * ROWS * N, segment);
            for (int slot = 0; slot < slots; slot++) {
                int expert = selected.get(slot);
                for (int r = 0; r < ROWS; r++) {
                    String what = type + " slot " + slot + " row " + r;
                    if (expert >= EXPERTS) {
                        assertEquals(what, POISON, out.get(slot * ROWS + r), 0f);
                    } else {
                        assertClose(
                                what,
                                hostDot(experts, expert * ROWS + r, x, slot * N),
                                out.get(slot * ROWS + r));
                    }
                }
            }
        }
    }

    /**
     * Three tokens of a four-token chunk, top-2: six assignments in a grouped order, the last
     * grouped position past the active ones. By token, each position reads its assignment's token;
     * by position, its own row of the input.
     */
    private static void checkGrouped(GGMLType type, int byToken, GroupedKernel kernel)
            throws Exception {
        byte[] raw = randomExperts(type, 7_20261028L + type.ordinal());
        int topK = 2;
        int batch = 4;
        int active = 3;
        // Assignment a is token a / 2, slot a % 2.
        IntArray selected = IntArray.fromElements(4, 1, 0, 4, 2, 1, 3, 3);
        IntArray grouped = IntArray.fromElements(2, 1, 5, 4, 0, 3, 7, 6);
        IntArray activeHolder = IntArray.fromElements(active);
        FloatArray x = new FloatArray(batch * topK * N);
        for (int i = 0; i < x.getSize(); i++) {
            x.set(i, ((i % 5) - 2) * 0.5f + (i / N) * 0.0625f);
        }
        FloatArray out = new FloatArray(batch * topK * ROWS);
        out.init(POISON);

        ByteArray w = toDevice(raw);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                x,
                                selected,
                                grouped,
                                activeHolder,
                                w,
                                out);
        kernel.add(graph, context, x, byToken, selected, grouped, activeHolder, w, out);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("moe.routed", rowsWorker(batch * topK * ROWS));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor experts = host(type).of(EXPERTS * ROWS * N, segment);
            for (int p = 0; p < batch * topK; p++) {
                for (int r = 0; r < ROWS; r++) {
                    String what = type + " byToken " + byToken + " position " + p + " row " + r;
                    if (p >= active * topK) {
                        assertEquals(what, POISON, out.get(p * ROWS + r), 0f);
                        continue;
                    }
                    int assignment = grouped.get(p);
                    int xRow = byToken == 1 ? assignment / topK : p;
                    assertClose(
                            what,
                            hostDot(experts, selected.get(assignment) * ROWS + r, x, xRow * N),
                            out.get(p * ROWS + r));
                }
            }
        }
    }

    private static SingleKernel single(GGMLType type) {
        return (graph, context, x, selected, w, out) -> {
            switch (type) {
                case Q8_0 ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ8_0,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
                case Q4_0 ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ4_0,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
                case Q4_1 ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ4_1,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
                case Q4_K ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ4_K,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
                case Q5_K ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ5_K,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
                default ->
                        graph.task(
                                "routed",
                                Qwen2MoEKernels::routedExpertsMatvecQ6_K,
                                context,
                                x,
                                N,
                                selected,
                                3,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                LOCAL);
            }
        };
    }

    private static GroupedKernel grouped(GGMLType type) {
        return (graph, context, x, byToken, selected, grouped, active, w, out) -> {
            switch (type) {
                case Q8_0 ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ8_0,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
                case Q4_0 ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_0,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
                case Q4_1 ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_1,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
                case Q4_K ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_K,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
                case Q5_K ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ5_K,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
                default ->
                        graph.task(
                                "routed",
                                Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ6_K,
                                context,
                                x,
                                byToken,
                                selected,
                                grouped,
                                active,
                                w,
                                out,
                                N,
                                ROWS,
                                EXPERTS,
                                2,
                                LOCAL);
            }
        };
    }

    private static final GGMLType[] TYPES = {
        GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K
    };

    @Test
    public void eachSlotReadsItsExpertsRowsAtItsOwnWindow() throws Exception {
        for (GGMLType type : TYPES) {
            checkSingleToken(type, single(type));
        }
    }

    @Test
    public void eachGroupedPositionReadsItsAssignmentsToken() throws Exception {
        for (GGMLType type : TYPES) {
            checkGrouped(type, 1, grouped(type));
        }
    }

    @Test
    public void eachGroupedPositionReadsItsOwnHiddenRow() throws Exception {
        for (GGMLType type : TYPES) {
            checkGrouped(type, 0, grouped(type));
        }
    }

    /**
     * The routed slots' outputs, weighted, and the shared expert's behind its sigmoid gate, added
     * to the residual; a slot whose expert is out of range adds nothing.
     */
    @Test
    public void theRoutedAndSharedOutputsAreAddedToTheResidual() throws Exception {
        int dim = 300;
        int used = 3;
        FloatArray input = new FloatArray(dim);
        FloatArray gateInput = new FloatArray(dim);
        FloatArray shared = new FloatArray(dim);
        FloatArray down = new FloatArray(used * dim);
        FloatArray residual = new FloatArray(dim);
        for (int i = 0; i < dim; i++) {
            input.set(i, ((i % 9) - 4) * 0.1f);
            gateInput.set(i, ((i % 4) - 1.5f) * 0.01f);
            shared.set(i, (i % 11) * 0.3f - 1f);
            residual.set(i, (i % 3) - 1f);
            for (int s = 0; s < used; s++) {
                down.set(s * dim + i, (s + 1) * 0.5f - (i % 6) * 0.2f);
            }
        }
        IntArray selected = IntArray.fromElements(2, -1, 0);
        FloatArray weights = FloatArray.fromElements(0.5f, 0.3f, 0.2f);
        float[] expected = new float[dim];
        float score = 0f;
        for (int i = 0; i < dim; i++) {
            score += gateInput.get(i) * input.get(i);
        }
        float sharedWeight = (float) (1.0 / (1.0 + Math.exp(-score)));
        for (int i = 0; i < dim; i++) {
            expected[i] =
                    residual.get(i)
                            + 0.5f * down.get(i)
                            + 0.2f * down.get(2 * dim + i)
                            + sharedWeight * shared.get(i);
        }

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                input,
                                gateInput,
                                shared,
                                down,
                                selected,
                                weights,
                                residual)
                        .task(
                                "accumulate",
                                Qwen2MoEKernels::routedAndSharedExpertsAccumulate,
                                context,
                                input,
                                gateInput,
                                shared,
                                down,
                                selected,
                                weights,
                                residual,
                                used,
                                EXPERTS,
                                dim,
                                128)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, residual);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid one = new WorkerGrid1D(128);
        one.setLocalWork(128, 1, 1);
        scheduler.addWorkerGrid("moe.accumulate", one);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        for (int i = 0; i < dim; i++) {
            assertClose("residual " + i, expected[i], residual.get(i));
        }
    }

    /** The shared expert's SwiGLU with its per-token sigmoid gate, over the active tokens only. */
    @Test
    public void theSharedSwiGluCarriesEachTokensGate() throws Exception {
        int hidden = 96;
        int batch = 3;
        FloatArray gate = new FloatArray(batch * hidden);
        FloatArray up = new FloatArray(batch * hidden);
        for (int i = 0; i < batch * hidden; i++) {
            gate.set(i, ((i % 13) - 6) * 0.4f);
            up.set(i, ((i % 5) - 2) * 0.7f);
        }
        FloatArray tokenWeights = FloatArray.fromElements(0.25f, 0.75f, 0.5f);
        IntArray active = IntArray.fromElements(2);
        float[] expected = new float[batch * hidden];
        for (int i = 0; i < batch * hidden; i++) {
            float g = gate.get(i);
            expected[i] =
                    i / hidden < 2
                            ? (float) (g / (1.0 + Math.exp(-g)))
                                    * up.get(i)
                                    * tokenWeights.get(i / hidden)
                            : g;
        }
        TaskGraph graph =
                new TaskGraph("moe")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION, gate, up, tokenWeights, active)
                        .task(
                                "swiglu",
                                Qwen2MoEBatchKernels::batchedSharedExpertSwiGLUGated,
                                new KernelContext(),
                                gate,
                                up,
                                tokenWeights,
                                active,
                                hidden)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, gate);
        GridScheduler scheduler = new GridScheduler();
        WorkerGrid all = new WorkerGrid1D(batch * hidden);
        all.setLocalWork(32, 1, 1);
        scheduler.addWorkerGrid("moe.swiglu", all);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        for (int i = 0; i < batch * hidden; i++) {
            assertClose("element " + i, expected[i], gate.get(i));
        }
    }
}
