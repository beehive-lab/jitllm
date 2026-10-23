package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Glm5NextLayers;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.weights.tornado.Glm5NextTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;

/**
 * The {@code glm5next} plan components (GLM-5.3-Flash), Q8_0: the embedding, the trunk's KDA and
 * sparse-attention layers with their hyper-connections and feed-forwards, and the logits. The
 * sequential prefill/decode mode runs the same graphs token by token.
 */
public class Glm5NextPlanComponents implements PrefillDecodeForwardPlanComponents {

    private final Glm5NextState state;
    private final Glm5NextTornadoWeights weights;
    private final Glm5NextConfiguration config;
    private final SchedulerType schedulerType;

    public Glm5NextPlanComponents(Glm5NextState state, Model model) {
        this.state = state;
        this.weights = (Glm5NextTornadoWeights) model.weights();
        this.config = (Glm5NextConfiguration) model.configuration();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Glm5NextLayers(
                "glm5next", state, weights, config, schedulerType, "activationUpdate");
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Glm5NextLayers(
                "glm5next", state, weights, config, schedulerType, "decodeActivation");
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }
}
