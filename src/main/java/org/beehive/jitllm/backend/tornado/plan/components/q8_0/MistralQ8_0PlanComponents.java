package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.MistralQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.mistral.MistralConfiguration;

public class MistralQ8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final MistralConfiguration config;
    private final SchedulerType schedulerType;
    private final LlamaQ8_0PlanComponents llama;

    public MistralQ8_0PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (MistralConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
        this.llama = new LlamaQ8_0PlanComponents(state, model, asLlama(config));
    }

    /**
     * Mistral's transformer layers are Llama's: the batched prefill/decode plan runs Llama's layer
     * graphs on this model's weights, under a Llama configuration with the same sizes.
     */
    private static LlamaConfiguration asLlama(MistralConfiguration c) {
        return new LlamaConfiguration(
                c.quantization(),
                c.dim(),
                c.hiddenDim(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.numberOfKeyValueHeads(),
                c.vocabularySize(),
                c.contextLength(),
                c.rmsNormEps(),
                c.ropeTheta());
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return llama.prefillDecodeActivation();
    }

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return llama.batchPrefillActivation(batchSize);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return llama.batchDecodeActivation(lastBatchLayerId);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return llama.prefillDecodeTransformerLayers();
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return llama.batchDecodeTransformerLayers();
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return llama.batchPrefillTransformerLayers(batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return llama.decodeLogits(previousGraphId);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new MistralQ8_0FFNLayers("mistralFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
