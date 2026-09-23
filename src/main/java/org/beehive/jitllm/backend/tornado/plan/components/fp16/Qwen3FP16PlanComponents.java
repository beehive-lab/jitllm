package org.beehive.jitllm.backend.tornado.plan.components.fp16;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.LogitsFP16Layer;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LogitsFP16LayerDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16FFNLayersPrefillDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;

public class Qwen3FP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final SchedulerType schedulerType;

    public Qwen3FP16PlanComponents(Qwen3State state, Model model) {
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
        return new BatchPrefillActivation(state, config, batchSize, false);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, false);
    }

    // ── Transformer layer TaskGraphs ──────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen3FP16FFNLayers("qwen3FFN", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen3FP16FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen3FP16FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new Qwen3FP16LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new Qwen3FP16LayersBatchPrefill(state, weights, config, batchSize);
    }

    // ── Logits layers ─────────────────────────────────────────────────────────

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsFP16Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsFP16LayerDecode(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }
}
