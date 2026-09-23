package org.beehive.jllm.backend.tornado.plan.components.q8_0;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.Qwen2MoEQ8_0FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.decode.Qwen2MoEQ8_0FFNLayersDecode;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.prefill.Qwen2MoEQ8_0LayersBatchPrefill;
import org.beehive.jllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen2MoEState;
import org.beehive.jllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.qwen2.Qwen2MoEConfiguration;

/** Assembles the single-token and batch-prefill Q8_0 GPU components for Qwen2-MoE. */
public final class Qwen2MoEQ8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Qwen2MoEState state;
    private final Qwen2MoETornadoWeights weights;
    private final Qwen2MoEConfiguration config;
    private final SchedulerType schedulerType;

    public Qwen2MoEQ8_0PlanComponents(Qwen2MoEState state, Model model) {
        this.state = state;
        this.config = (Qwen2MoEConfiguration) model.configuration();
        this.weights = (Qwen2MoETornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, true);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen2MoEQ8_0FFNLayers("qwen2MoEFFN", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen2MoEQ8_0FFNLayers("decode", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen2MoEQ8_0FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new Qwen2MoEQ8_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsQ8_0LayerDecode(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
