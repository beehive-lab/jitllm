package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayerDecode;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.qwen2.Qwen2Q8_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.qwen2.Qwen2Q8_0LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.qwen2.Qwen2Q8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Qwen2Q8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;

/**
 * Qwen2 (and DeepSeek-R1-Distill-Qwen) plans: single-token, and batched prefill with single-token
 * decode. The sequential prefill/decode plan is not offered.
 */
public class Qwen2Q8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Qwen2State state;
    private final Qwen2TornadoWeights weights;
    private final Qwen2Configuration config;
    private final SchedulerType schedulerType;

    public Qwen2Q8_0PlanComponents(Qwen2State state, Model model) {
        this.state = state;
        this.config = (Qwen2Configuration) model.configuration();
        this.weights = (Qwen2TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation(GraphNames.DECODE_ACTIVATION, state, weights, config);
    }

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(
                state, config, lastBatchLayerId, weights.getTokenEmbeddingTable().dataType());
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen2Q8_0FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen2Q8_0FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen2Q8_0FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new Qwen2Q8_0LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new Qwen2Q8_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayerDecode(
                state, weights, config, previousGraphId, schedulerType);
    }
}
