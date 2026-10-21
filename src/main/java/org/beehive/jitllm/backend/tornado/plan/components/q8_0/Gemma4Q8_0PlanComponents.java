package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Gemma4BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Gemma4LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Gemma4Q8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.StagedForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.Gemma4BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Q8_0 single-token plan components for the Gemma 4 architecture.
 *
 * <p>The Q8_0 counterpart of {@code Gemma4FP16PlanComponents}: same wiring (Activation,
 * Gemma4-specific transformer layers, Gemma4-specific logits layer with the final logit soft-cap),
 * but using the Q8_0 layer implementations. STANDARD execution mode only.
 */
public class Gemma4Q8_0PlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final Gemma4State state;
    private final Gemma4TornadoWeights weights;
    private final Gemma4Configuration config;
    private final SchedulerType schedulerType;

    public Gemma4Q8_0PlanComponents(Gemma4State state, Model model) {
        this.state = state;
        this.config = (Gemma4Configuration) model.configuration();
        this.weights = (Gemma4TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Gemma4Q8_0FFNLayers("gemma4FFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new Gemma4LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    // ── The batched prefill/decode plan ───────────────────────────────────────

    /**
     * A Q8_0 or Q4_0 embedding is decoded on the device: the host copies each token's raw row
     * instead of decoding the chunk in Java before the JIT has compiled that loop.
     */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        if (embedding == DataType.Q8_0 || embedding == DataType.Q4_0) {
            return new BatchPrefillQ8DeviceActivation(state, config, batchSize, embedding);
        }
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new Gemma4BatchPrefillLayers(state, weights, config, batchSize);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Gemma4BatchDecodeActivation(state, weights, config, lastBatchLayerId);
    }

    // @formatter:off
    /**
     * The decode layers of the batched plan: the same class the single-token plan builds, told that
     * it is in the batched one.
     *
     * <p>A flag rather than a subclass. The two differ only in which graph layer 0 names as the
     * producer of its activation and of the key/value caches, and the dispatch ledger exists to
     * stop a class per plan for a difference that small.
     */
    // @formatter:on
    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Gemma4Q8_0FFNLayers("gemma4FFN", state, weights, config, schedulerType, true);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new Gemma4LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    // @formatter:off
    /**
     * Not implemented and not declared by the provider: this family has no layer graphs for the
     * single-token prefill/decode plan, and the batched one is its prefill path. Refused by name
     * rather than answered with something that would be wrong.
     */
    // @formatter:on
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

    // ── Pipeline stages ───────────────────────────────────────────────────────

    /** A later stage's first layer takes its activation from the graph that received it. */
    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer) {
        return new Gemma4Q8_0FFNLayers(
                "gemma4FFN",
                state,
                weights,
                config,
                schedulerType,
                false,
                firstLayer,
                endLayer,
                firstLayer == 0 ? null : "activationUpdate");
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        return new Gemma4BatchPrefillLayers(
                state, weights, config, batchSize, firstLayer, endLayer);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        return new Gemma4Q8_0FFNLayers(
                "gemma4FFN",
                state,
                weights,
                config,
                schedulerType,
                true,
                firstLayer,
                endLayer,
                firstLayer == 0 ? null : "decodeActivation");
    }
}
