package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayerDecode;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.devstral.DevstralQ8_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.DevstralQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;

public class DevstralQ8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final DevstralState state;
    private final LlamaTornadoWeights weights;
    private final DevstralConfiguration config;
    private final SchedulerType schedulerType;

    public DevstralQ8_0PlanComponents(DevstralState state, Model model) {
        this.state = state;
        this.config = (DevstralConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new DevstralQ8_0FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Batched prefill/decode ───────────────────────────────────────────────

    /**
     * Llama's tensor-core layers: Devstral's layer is Llama's with a Q wider than the model ({@code
     * qDim != dim}), which those layers size by, and RoPE over the precomputed (YaRN) tables.
     */
    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new LlamaQ8_0LayersBatchPrefillMMA(state, weights, config, batchSize);
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
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new DevstralQ8_0FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayerDecode(
                state, weights, config, previousGraphId, schedulerType);
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
