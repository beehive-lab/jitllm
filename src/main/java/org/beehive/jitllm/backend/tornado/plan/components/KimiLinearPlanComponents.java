package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.KimiLinearBatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.KimiLinearLayers;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.weights.tornado.KimiLinearTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;

/**
 * The {@code kimi-linear} plan components (Kimi-Linear-48B-A3B): the embedding, the KDA, latent
 * attention and mixture-of-experts layers, and the logits, one token at a time or as a sequential
 * prefill followed by decode, on one device or split across several.
 */
public class KimiLinearPlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final KimiLinearState state;
    private final KimiLinearTornadoWeights weights;
    private final KimiLinearConfiguration config;
    private final SchedulerType schedulerType;

    /** The batched prefill layers, once built: what the batched plan's decode layers read. */
    private KimiLinearBatchPrefillLayers prefill;

    public KimiLinearPlanComponents(KimiLinearState state, Model model) {
        this.state = state;
        this.weights = (KimiLinearTornadoWeights) model.weights();
        this.config = (KimiLinearConfiguration) model.configuration();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return singleTokenTransformerLayers(0, config.numberOfLayers());
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation(GraphNames.DECODE_ACTIVATION, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new KimiLinearLayers(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.DECODE_ACTIVATION,
                0,
                config.numberOfLayers());
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer) {
        return new KimiLinearLayers(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.ACTIVATION_UPDATE,
                firstLayer,
                endLayer);
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

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Activation(GraphNames.DECODE_ACTIVATION, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return batchDecodeTransformerLayers(0, config.numberOfLayers());
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        prefill =
                new KimiLinearBatchPrefillLayers(
                        state, weights, config, batchSize, firstLayer, endLayer);
        return prefill;
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        if (prefill == null) {
            throw new IllegalStateException(
                    "the kimi-linear batched decode layers are built after the prefill layers");
        }
        return new KimiLinearLayers(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.DECODE_ACTIVATION,
                firstLayer,
                endLayer,
                prefill);
    }
}
