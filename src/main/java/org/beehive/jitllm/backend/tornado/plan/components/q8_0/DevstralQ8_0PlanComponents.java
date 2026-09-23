package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.DevstralQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;

public class DevstralQ8_0PlanComponents implements SingleTokenForwardPlanComponents {

    private final DevstralState state;
    private final LlamaTornadoWeights weights;
    private final DevstralConfiguration config;
    private final SchedulerType schedulerType;

    public DevstralQ8_0PlanComponents(DevstralState state, Model model) {
        this.state = state;
        this.config = (DevstralConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new DevstralQ8_0FFNLayers("devstralFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
