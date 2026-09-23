package org.beehive.jllm.backend.tornado.plan.components.q8_0;

import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.decode.Qwen3Q8_0FFNLayersDecode;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.decode.Qwen3Q8_0FFNLayersPrefillDecode;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.prefill.Qwen3Q8_0LayersBatchPrefill;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.prefill.Qwen3Q8_0LayersBatchPrefillMMA;
import org.beehive.jllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;

public class Qwen3Q8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final SchedulerType schedulerType;

    public Qwen3Q8_0PlanComponents(Qwen3State state, Model model) {
        this.state = state;
        this.config = (Qwen3Configuration) model.configuration();
        this.weights = (Qwen3TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    // ── Activations ───────────────────────────────────────────────────────────

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

    // ── Transformer layer TaskGraphs ──────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen3Q8_0FFNLayers("qwen3FFN", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen3Q8_0FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen3Q8_0FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new Qwen3Q8_0LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new Qwen3Q8_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    // ── Logits layers ─────────────────────────────────────────────────────────

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
