package org.beehive.jllm.backend.tornado.plan.components.q8_0;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.ActivationGranite;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.GraniteQ8_0FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsGraniteQ8_0Layer;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.GraniteState;
import org.beehive.jllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.granite.GraniteConfiguration;

public class GraniteQ8_0PlanComponents implements SingleTokenForwardPlanComponents {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final SchedulerType schedulerType;

    public GraniteQ8_0PlanComponents(GraniteState state, Model model) {
        this.state = state;
        this.config = (GraniteConfiguration) model.configuration();
        this.weights = (GraniteTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new ActivationGranite("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new GraniteQ8_0FFNLayers("graniteFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsGraniteQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
