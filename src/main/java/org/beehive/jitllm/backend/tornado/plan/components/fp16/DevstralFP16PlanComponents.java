package org.beehive.jitllm.backend.tornado.plan.components.fp16;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.devstral.DevstralFP16FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.DevstralFP16FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.LogitsFP16Layer;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LogitsFP16LayerDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;

public class DevstralFP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final DevstralState state;
    private final LlamaTornadoWeights weights;
    private final DevstralConfiguration config;
    private final SchedulerType schedulerType;

    public DevstralFP16PlanComponents(DevstralState state, Model model) {
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
        return new DevstralFP16FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsFP16Layer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Batched prefill/decode ───────────────────────────────────────────────

    /** Llama's F16 tensor-core layers, sized by {@code qDim != dim}. */
    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new LlamaFP16LayersBatchPrefillMMA(state, weights, config, batchSize);
    }

    /**
     * An F16 embedding: copied by the host into the FP16 staging buffer, converted on the device.
     */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        boolean f16Embedding = weights.getTokenEmbeddingTable().dataType() == DataType.F16;
        return new BatchPrefillActivation(state, config, batchSize, !f16Embedding);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        boolean f16Embedding = weights.getTokenEmbeddingTable().dataType() == DataType.F16;
        return new BatchDecodeActivation(state, config, lastBatchLayerId, !f16Embedding);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new DevstralFP16FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsFP16LayerDecode(state, weights, config, previousGraphId, schedulerType);
    }

    /** Devstral has no sequential prefill/decode plan; its provider refuses the mode. */
    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException("devstral has no sequential prefill/decode plan");
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException("devstral has no sequential prefill/decode plan");
    }
}
