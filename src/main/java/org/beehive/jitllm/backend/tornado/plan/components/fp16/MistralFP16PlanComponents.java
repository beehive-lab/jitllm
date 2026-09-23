package org.beehive.jllm.backend.tornado.plan.components.fp16;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.fp16.LogitsFP16Layer;
import org.beehive.jllm.backend.tornado.layers.type.fp16.MistralFP16FFNLayers;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.mistral.MistralConfiguration;

public class MistralFP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final MistralConfiguration config;
    private final SchedulerType schedulerType;

    public MistralFP16PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (MistralConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new MistralFP16FFNLayers("mistralFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsFP16Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
