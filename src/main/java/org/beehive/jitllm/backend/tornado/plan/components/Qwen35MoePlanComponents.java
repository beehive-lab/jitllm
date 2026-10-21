package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.Qwen35BatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeBatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeFeedForward;
import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.weights.tornado.Qwen35MoeTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen35moe.Qwen35Moe;

/**
 * The {@code qwen35moe} plan components: the {@code qwen35} layer graphs, each layer's feed-forward
 * the routed and shared experts.
 */
public class Qwen35MoePlanComponents extends Qwen35PlanComponents {

    private final Qwen35MoeState state;
    private final Qwen35MoeTornadoWeights weights;
    private final Qwen35Moe model;

    public Qwen35MoePlanComponents(Qwen35MoeState state, Model model) {
        super(state, model);
        this.state = state;
        this.weights = (Qwen35MoeTornadoWeights) model.weights();
        this.model = (Qwen35Moe) model;
    }

    @Override
    protected Qwen35FeedForward feedForward() {
        return new Qwen35MoeFeedForward(
                state.workspace, model.experts(), weights.experts(), model.configuration().dim());
    }

    @Override
    protected Qwen35BatchFeedForward batchFeedForward(int batchSize) {
        return new Qwen35MoeBatchFeedForward(
                state.workspace,
                model.experts(),
                weights.experts(),
                model.configuration().dim(),
                batchSize);
    }
}
