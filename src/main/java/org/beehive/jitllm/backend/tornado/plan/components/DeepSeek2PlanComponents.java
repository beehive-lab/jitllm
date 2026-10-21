package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.DeepSeek2BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.DeepSeek2Layers;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek2TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;

// @formatter:off
/**
 * The {@code deepseek2} plan components (GLM-4.7-Flash), Q8_0: the embedding, the latent-attention
 * and mixture-of-experts layers, and the logits, in all three modes.
 *
 * <p>The batched plan's decode layers take the cache and every weight the prefill graphs bind from
 * those graphs, by name, so the plan holds one copy of each. That is why the decode layers are
 * built from the prefill layers this object built, which the plan asks for first.
 */
// @formatter:on
public class DeepSeek2PlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final DeepSeek2State state;
    private final DeepSeek2TornadoWeights weights;
    private final DeepSeek2Configuration config;
    private final SchedulerType schedulerType;

    /** The batched prefill layers, once built: what the batched plan's decode layers read. */
    private DeepSeek2BatchPrefillLayers prefill;

    public DeepSeek2PlanComponents(DeepSeek2State state, Model model) {
        this.state = state;
        this.weights = (DeepSeek2TornadoWeights) model.weights();
        this.config = (DeepSeek2Configuration) model.configuration();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new DeepSeek2Layers("deepseek2", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    // ── Sequential prefill/decode ─────────────────────────────────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new DeepSeek2Layers(
                "deepseek2",
                state,
                weights,
                config,
                schedulerType,
                "decodeActivation",
                0,
                config.numberOfLayers());
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }

    // ── Batched prefill/decode ────────────────────────────────────────────────

    /** The embedding is Q8_0: the chunk's rows are decoded on the device. */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillQ8DeviceActivation(state, config, batchSize);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return batchPrefillTransformerLayers(batchSize, 0, config.numberOfLayers());
    }

    /** The decode layers bind the cache from the prefill graphs directly; no pass-through. */
    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return batchDecodeTransformerLayers(0, config.numberOfLayers());
    }

    // ── Pipeline stages ───────────────────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer) {
        return new DeepSeek2Layers(
                "deepseek2",
                state,
                weights,
                config,
                schedulerType,
                "activationUpdate",
                firstLayer,
                endLayer);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        prefill =
                new DeepSeek2BatchPrefillLayers(
                        state, weights, config, batchSize, firstLayer, endLayer);
        return prefill;
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        if (prefill == null) {
            throw new IllegalStateException(
                    "the deepseek2 batched decode layers are built after the prefill layers");
        }
        return new DeepSeek2Layers(
                "deepseek2",
                state,
                weights,
                config,
                schedulerType,
                "decodeActivation",
                firstLayer,
                endLayer,
                prefill);
    }
}
