package org.beehive.jitllm.backend.tornado.plan.components.q4_k;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayerDecode;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q4_k.DevstralQ4_KFFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q4_k.decode.DevstralQ4_KFFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.q4_k.prefill.DevstralQ4_KLayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;

/**
 * Devstral's plan when its per-layer weights are retained as Q4_K.
 *
 * <p>Mixed by construction, because the file is: {@code Q4_K_M} stores the 240 {@code blk.*} weight
 * tensors as Q4_K but {@code output.weight} as Q6_K, which has no device kernel and is still
 * materialized as Q8_0. So the transformer layers read Q4_K and the logits layer is the existing
 * Q8_0 one — each reading what its own tensors actually hold, which is why this needs no per-tensor
 * dispatch.
 */
public class DevstralQ4_KPlanComponents implements BatchPrefillDecodeForwardPlanComponents {

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
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new DevstralQ4_KFFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Batched prefill/decode ───────────────────────────────────────────────

    /** the F16 tensor-core layers, each Q4_K (or Q6_K) projection decoded into FP16 first. */
    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new DevstralQ4_KLayersBatchPrefillMMA(state, weights, config, batchSize);
    }

    /** The embedding is materialized as Q8_0; the host writes the FP32 batch directly. */
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
        return new DevstralQ4_KFFNLayersDecode(state, weights, config, schedulerType);
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
