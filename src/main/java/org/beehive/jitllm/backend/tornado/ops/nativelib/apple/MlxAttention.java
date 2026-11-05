package org.beehive.jitllm.backend.tornado.ops.nativelib.apple;

import static org.beehive.jitllm.program.op.OperationKind.ATTENTION;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.ops.Fusion;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Instance;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * A batched prefill's attention as MLX's fused causal attention, through {@link
 * MlxPrefillAttention}: the paged keys and values gathered into contiguous buffers, then one MLX
 * call per layer. Used where the layer's projections run on MLX too ({@link MlxPrefill#resolve}).
 *
 * <p>Example: Llama-3.2-1B Q8_0 under MLX runs {@code batch_attention} as MLX's attention; with
 * {@code --no-native-libraries} the layer's own flash attention kernel runs it.
 */
public final class MlxAttention implements Implementation {

    private static final Fusion ATTENTION_ONLY = Fusion.of("attention", ATTENTION);

    private final MlxPrefillAttention attention;

    /** MLX for the attention of the layers {@code attention} holds buffers for. */
    public MlxAttention(MlxPrefillAttention attention) {
        this.attention = attention;
    }

    @Override
    public String library() {
        return "MLX";
    }

    @Override
    public List<Fusion> fusions() {
        return List.of(ATTENTION_ONLY);
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return true;
    }

    @Override
    public Instance create(
            List<Step> steps, Target target, KernelContext context, WeightPrepacking prepacking) {
        return new Instance() {

            /** The graphs that got an attention task. */
            private final Set<String> graphs = new LinkedHashSet<>();

            @Override
            public void emit(
                    TaskGraph graph, Step step, Operands operands, WeightBindings bindings) {
                graphs.add(graph.getTaskGraphName());
                attention.attention(graph, "batch_attention", step.layer());
            }

            @Override
            public void addGrids(GridScheduler scheduler) {
                graphs.forEach(
                        graph -> attention.addGrids(scheduler, graph + ".", "batch_attention"));
            }
        };
    }
}
