package org.beehive.jitllm.backend.tornado.ops.jit;

import static org.beehive.jitllm.program.op.OperationKind.ATTENTION;
import static org.beehive.jitllm.program.op.OperationKind.MAT_MUL;
import static org.beehive.jitllm.program.op.OperationKind.RESIDUAL_ADD;
import static org.beehive.jitllm.program.op.OperationKind.RMS_NORM;
import static org.beehive.jitllm.program.op.OperationKind.SWIGLU;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.ops.Fusion;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Instance;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * The JIT kernels of a batched-prefill layer class that adds its own tasks for each step: RMS norm
 * with q|k|v, the attention, the output projection with its residual add, RMS norm with gate|up and
 * SwiGLU, and the down projection with its residual add. The layer class keeps the kernels; this
 * class only runs them in the steps the plan splits a layer into.
 *
 * <p>Example: on Metal, Llama's Q8_0 layer runs {@code batch_attn_rms}, {@code
 * batch_attn_rms_apply} and {@code batch_qkv} for the first step; under MLX the same step is MLX's.
 */
public final class LayerKernels implements Implementation {

    /** The tasks of each step, added by the layer class that owns the kernels. */
    public interface Tasks {

        /** RMS norm, then q, k and v (or the fused q|k|v). */
        void attentionInput(TaskGraph graph, int layer);

        /** The output projection, with the residual add. */
        void attentionOutput(TaskGraph graph, int layer);

        /** RMS norm, gate and up (or the fused gate|up), and SwiGLU. */
        void feedForwardUp(TaskGraph graph, int layer);

        /** The down projection, with the residual add. */
        void feedForwardDown(TaskGraph graph, int layer);

        /** The attention, after RoPE has written the chunk's keys and values to the cache. */
        void attention(TaskGraph graph, int layer);

        /** The grids of the tasks above in layer {@code layer}, under its prefix {@code p}. */
        void addGrids(GridScheduler scheduler, String p, int layer);
    }

    private static final Fusion ATTENTION_INPUT =
            Fusion.of("RMS norm + q|k|v", RMS_NORM, MAT_MUL, MAT_MUL, MAT_MUL)
                    .when(Fusion::sameInput);

    private static final Fusion ATTENTION_INPUT_FUSED =
            Fusion.of("RMS norm + q|k|v", RMS_NORM, MAT_MUL)
                    .when(ops -> role(ops.get(1)) == TensorRole.ATTENTION_QKV);

    private static final Fusion GATE_UP =
            Fusion.of("RMS norm + gate|up + SwiGLU", RMS_NORM, MAT_MUL, MAT_MUL, SWIGLU);

    private static final Fusion GATE_UP_FUSED =
            Fusion.of("RMS norm + gate|up + SwiGLU", RMS_NORM, MAT_MUL, SWIGLU)
                    .when(ops -> role(ops.get(1)) == TensorRole.FFN_GATE_UP);

    private static final Fusion PROJECTION_ADD =
            Fusion.of("projection + residual add", MAT_MUL, RESIDUAL_ADD);

    private static final Fusion ATTENTION_ONLY = Fusion.of("attention", ATTENTION);

    private static final List<Fusion> FUSIONS =
            List.of(
                    ATTENTION_INPUT,
                    GATE_UP,
                    ATTENTION_INPUT_FUSED,
                    GATE_UP_FUSED,
                    PROJECTION_ADD,
                    ATTENTION_ONLY);

    private static TensorRole role(Operation op) {
        return ((MatMul) op).weight().role();
    }

    private final Tasks tasks;

    /** The JIT kernels of the layer class whose tasks {@code tasks} adds. */
    public LayerKernels(Tasks tasks) {
        this.tasks = tasks;
    }

    @Override
    public String library() {
        return "none";
    }

    @Override
    public List<Fusion> fusions() {
        return FUSIONS;
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return true;
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        return new Instance() {

            /** The layers whose graphs got tasks, by graph name. */
            private final Map<String, Integer> layers = new LinkedHashMap<>();

            @Override
            public void emit(
                    TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
                int i = step.layer();
                layers.put(graph.getTaskGraphName(), i);
                Fusion f = step.fusion();
                if (f == ATTENTION_INPUT || f == ATTENTION_INPUT_FUSED) {
                    tasks.attentionInput(graph, i);
                } else if (f == GATE_UP || f == GATE_UP_FUSED) {
                    tasks.feedForwardUp(graph, i);
                } else if (f == ATTENTION_ONLY) {
                    tasks.attention(graph, i);
                } else if (f == PROJECTION_ADD) {
                    if (role(step.ops().get(0)) == TensorRole.ATTENTION_OUTPUT) {
                        tasks.attentionOutput(graph, i);
                    } else {
                        tasks.feedForwardDown(graph, i);
                    }
                } else {
                    throw new IllegalArgumentException("not a fusion of the JIT kernels: " + f);
                }
                // The weights the step's tasks read, in the order they run.
                for (Operation op : step.ops()) {
                    if (op instanceof RmsNorm norm) {
                        bindings.record(
                                operands.weight(norm.weight().orElseThrow(), i).asFloatArray(),
                                graph.getTaskGraphName());
                    } else if (op instanceof MatMul m) {
                        bindings.record(
                                operands.weight(m.weight(), i).deviceArray(),
                                graph.getTaskGraphName());
                    }
                }
            }

            @Override
            public void addGrids(GridScheduler scheduler) {
                layers.forEach((graph, i) -> tasks.addGrids(scheduler, graph + ".", i));
            }
        };
    }
}
