package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Glm5NextBatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.Glm5NextLayers;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.weights.tornado.Glm5NextTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;

/**
 * The {@code glm5next} plan components (GLM-5.3-Flash), Q8_0: the embedding, the trunk's KDA and
 * sparse-attention layers with their hyper-connections and feed-forwards, and the logits. The
 * sequential prefill/decode mode runs the same graphs token by token; the batched mode prefills
 * chunk by chunk, and its decode layers take the caches, the recurrent state and the weights from
 * the prefill graph, which the plan builds first.
 */
public class Glm5NextPlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final Glm5NextState state;
    private final Glm5NextTornadoWeights weights;
    private final Glm5NextConfiguration config;
    private final SchedulerType schedulerType;

    /** The batched prefill layers, once built: what the batched plan's decode layers read. */
    private Glm5NextBatchPrefillLayers prefill;

    public Glm5NextPlanComponents(Glm5NextState state, Model model) {
        this.state = state;
        this.weights = (Glm5NextTornadoWeights) model.weights();
        this.config = (Glm5NextConfiguration) model.configuration();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Glm5NextLayers(
                "glm5next", state, weights, config, schedulerType, "activationUpdate");
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Glm5NextLayers(
                "glm5next", state, weights, config, schedulerType, "decodeActivation");
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
        prefill = new Glm5NextBatchPrefillLayers(state, weights, config, batchSize);
        return prefill;
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        if (prefill == null) {
            throw new IllegalStateException(
                    "the glm5next batched decode layers are built after the prefill layers");
        }
        return new Glm5NextLayers(
                "glm5next", state, weights, config, schedulerType, "decodeActivation", prefill);
    }
}
