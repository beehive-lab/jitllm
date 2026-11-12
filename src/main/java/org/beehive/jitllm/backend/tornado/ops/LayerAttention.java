package org.beehive.jitllm.backend.tornado.ops;

import static org.beehive.jitllm.program.op.OperationKind.ATTENTION;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * The attention of a batched-prefill layer class that adds its tasks itself, on the JIT kernels or
 * on cuDNN. The layer class keeps the kernels and the buffers; the plan picks which of the two runs
 * each layer's attention.
 *
 * <p>Its instance reads neither the operands nor the weight bindings it is given: the layer's tasks
 * name their own buffers.
 *
 * <p>Example: Qwen3-0.6B F16 on CUDA runs the first chunk's attention on cuDNN's fused attention;
 * its fallback family, which runs the later chunks, has no cuDNN form and runs the tensor-core
 * attention.
 */
public abstract class LayerAttention implements Implementation {

    /** The attention tasks, added by the layer class that owns the kernels. */
    public interface Tasks {
        /** Whether the layer can run {@code step}'s attention on cuDNN. */
        boolean cudnnForm(Step step);

        /** Adds the tasks of {@code step}'s attention, on cuDNN or on the JIT kernels. */
        void attend(TaskGraph graph, Step step, boolean cudnn);

        /**
         * Adds the worker grids of the tasks {@link #attend} added for {@code step} to {@code
         * graph}. Example: under cuDNN, {@code cudnn_pack_q}, {@code cudnn_gather_kv} and {@code
         * cudnn_scatter}; the library call takes none. A layer that registers these grids itself
         * keeps the default.
         */
        default void addAttentionGrids(
                GridScheduler scheduler, String graph, Step step, boolean cudnn) {}
    }

    /** One layer's attention. */
    public static final Fusion ATTENTION_ONLY = Fusion.of("attention", ATTENTION);

    protected final Tasks tasks;

    protected LayerAttention(Tasks tasks) {
        this.tasks = tasks;
    }

    /** Whether this implementation runs the attention on cuDNN. */
    protected abstract boolean cudnn();

    @Override
    public List<Fusion> fusions() {
        return List.of(ATTENTION_ONLY);
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        return new Instance() {

            /** Each step this instance added, with its graph's name, in order. */
            private final List<Map.Entry<String, Step>> added = new ArrayList<>();

            @Override
            public void emit(
                    TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
                tasks.attend(graph, step, cudnn());
                added.add(Map.entry(graph.getTaskGraphName(), step));
            }

            @Override
            public void addGrids(GridScheduler scheduler) {
                added.forEach(
                        e -> tasks.addAttentionGrids(scheduler, e.getKey(), e.getValue(), cudnn()));
            }
        };
    }
}
