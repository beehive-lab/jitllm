package org.beehive.jllm.backend.tornado.plan.components.fp16;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.ActivationGranite;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.fp16.LogitsGraniteFP16Layer;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.GraniteState;
import org.beehive.jllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.granite.GraniteConfiguration;

public class GraniteFP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final SchedulerType schedulerType;

    public GraniteFP16PlanComponents(GraniteState state, Model model) {
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
        return new GraniteFP16FFNLayers("graniteFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsGraniteFP16Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
