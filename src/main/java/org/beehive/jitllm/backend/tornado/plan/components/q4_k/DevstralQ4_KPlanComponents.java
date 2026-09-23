package org.beehive.jllm.backend.tornado.plan.components.q4_k;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.q4_k.DevstralQ4_KFFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.DevstralState;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.devstral.DevstralConfiguration;

/**
 * Devstral's plan when its per-layer weights are retained as Q4_K.
 *
 * <p>Mixed by construction, because the file is: {@code Q4_K_M} stores the 240 {@code blk.*} weight
 * tensors as Q4_K but {@code output.weight} as Q6_K, which has no device kernel and is still
 * materialized as Q8_0. So the transformer layers read Q4_K and the logits layer is the existing
 * Q8_0 one — each reading what its own tensors actually hold, which is why this needs no per-tensor
 * dispatch.
 */
public class DevstralQ4_KPlanComponents implements SingleTokenForwardPlanComponents {

    private final DevstralState state;
    private final LlamaTornadoWeights weights;
    private final DevstralConfiguration config;
    private final SchedulerType schedulerType;

    public DevstralQ4_KPlanComponents(DevstralState state, Model model) {
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
        return new DevstralQ4_KFFNLayers("devstralFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
