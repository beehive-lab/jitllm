package org.beehive.jitllm.backend.tornado.plan.components.fp16;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Gemma4BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.Gemma4FP16FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.Gemma4LogitsFP16Layer;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.Gemma4BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * FP16 plan components for the Gemma 4 architecture.
 *
 * <p>Wires together the (model-agnostic) {@link Activation} task graph, the Gemma4-specific {@link
 * Gemma4FP16FFNLayers transformer layers} (Q/K-norm, pre/post normalization sandwich,
 * sliding-window attention, GeGLU FFN, per-layer embeddings) and the Gemma4-specific {@link
 * Gemma4LogitsFP16Layer logits layer} (final logit soft-cap), for the single-token plan and the
 * batched prefill/decode plan on the tensor cores. The batched prefill runs the same layers as the
 * quantized trunks, reading the F16 projections copied into their FP16 scratch; there is no
 * sequential prefill/decode plan.
 */
public class Gemma4FP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Gemma4State state;
    private final Gemma4TornadoWeights weights;
    private final Gemma4Configuration config;
    private final SchedulerType schedulerType;

    public Gemma4FP16PlanComponents(Gemma4State state, Model model) {
        this.state = state;
        this.config = (Gemma4Configuration) model.configuration();
        this.weights = (Gemma4TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Gemma4FP16FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new Gemma4LogitsFP16Layer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── The batched prefill/decode plan ───────────────────────────────────────

    /**
     * An F16 embedding is copied by the host into the FP16 staging buffer and converted on the
     * device; any other the host writes into the FP32 batch directly.
     */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        boolean f16Embedding = weights.getTokenEmbeddingTable().dataType() == DataType.F16;
        return new BatchPrefillActivation(state, config, batchSize, !f16Embedding);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new Gemma4BatchPrefillLayers(state, weights, config, batchSize);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Gemma4BatchDecodeActivation(state, weights, config, lastBatchLayerId);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Gemma4FP16FFNLayers(state, weights, config, schedulerType, true);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new Gemma4LogitsFP16Layer(state, weights, config, previousGraphId, schedulerType);
    }

    /** Not declared by the provider: the batched plan is this family's prefill path. */
    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException(
                "gemma4 has no PREFILL_DECODE layer graphs; the batched plan is its prefill path");
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException(
                "gemma4 has no PREFILL_DECODE layer graphs; the batched plan is its prefill path");
    }
}
