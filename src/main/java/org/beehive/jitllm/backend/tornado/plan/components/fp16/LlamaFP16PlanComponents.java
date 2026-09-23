package org.beehive.jllm.backend.tornado.plan.components.fp16;

import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.fp16.LogitsFP16Layer;
import org.beehive.jllm.backend.tornado.layers.type.fp16.decode.LlamaFP16FFNLayersDecode;
import org.beehive.jllm.backend.tornado.layers.type.fp16.decode.LlamaFP16FFNLayersPrefillDecode;
import org.beehive.jllm.backend.tornado.layers.type.fp16.decode.LogitsFP16LayerDecode;
import org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill;
import org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.llama.LlamaConfiguration;

/**
 * {@link BatchPrefillDecodeForwardPlanComponents} for Llama + FP16.
 *
 * <p>Provides all layer objects and the FP16 embedding preparer (raw half-float byte copy).
 */
public class LlamaFP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final SchedulerType schedulerType;

    public LlamaFP16PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (LlamaConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
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

    // ── Transformer layer TaskGraphs ──────────────────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new LlamaFP16FFNLayers("llamaFFN", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new LlamaFP16FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new LlamaFP16FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new LlamaFP16LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new LlamaFP16LayersBatchPrefill(state, weights, config, batchSize);
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
