package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;

// @formatter:off
/**
 * The single-token {@code qwen35} layers of one pipeline stage, or of the whole model, several
 * layers to a task graph.
 *
 * <p>Every task graph of a plan is launched, and synchronized, on its own: one graph per layer is a
 * launch and a stream synchronization per layer per token. That is invisible behind a dense model's
 * long layers and is most of the time between kernels for a mixture of experts, whose layers are
 * short — {@code qwen35moe} at one graph per layer synchronized about a hundred times a token.
 * Grouping is counted from the stage's first layer, so each graph is named for its first layer and
 * the layers after it in the graph carry an {@code l<slot>_} task prefix.
 */
// @formatter:on
public class Qwen35FFNLayersGrouped extends Qwen35FFNLayers {

    /** Layers per task graph; the whole stage at the default. */
    static final int LAYERS_PER_GRAPH = 64;

    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    public Qwen35FFNLayersGrouped(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            int firstLayer,
            int endLayer,
            Qwen35FeedForward feedForward) {
        super(
                state,
                weights,
                config,
                schedulerType,
                activationGraphName,
                firstLayer,
                endLayer,
                feedForward);
    }
}
