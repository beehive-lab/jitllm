package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.DeepSeek4BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.DeepSeek4Layers;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek4TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The {@code deepseek4} plan components (DeepSeek-V4), Q8_0: the embedding, the trunk's layers with
 * their hyper-connections, compressors, indexers and mixtures of experts, and the logits. The
 * sequential prefill/decode mode runs the same graphs token by token; the batched mode prefills
 * chunk by chunk, and its decode layers take the caches and the weights from the prefill graph,
 * which the plan builds first. Split across devices, each stage builds its range of layers.
 */
public class DeepSeek4PlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final DeepSeek4State state;
    private final DeepSeek4TornadoWeights weights;
    private final DeepSeek4Configuration config;
    private final SchedulerType schedulerType;

    /** The batched prefill layers, once built: what the batched plan's decode layers read. */
    private DeepSeek4BatchPrefillLayers prefill;

    public DeepSeek4PlanComponents(DeepSeek4State state, Model model) {
        this.state = state;
        this.weights = (DeepSeek4TornadoWeights) model.weights();
        this.config = (DeepSeek4Configuration) model.configuration();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new DeepSeek4Layers(state, weights, config, schedulerType, "activationUpdate");
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new DeepSeek4Layers(state, weights, config, schedulerType, "decodeActivation");
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
        return new DeepSeek4Layers(
                state,
                weights,
                config,
                schedulerType,
                "activationUpdate",
                null,
                firstLayer,
                endLayer);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        prefill =
                new DeepSeek4BatchPrefillLayers(
                        state, weights, config, batchSize, firstLayer, endLayer);
        return prefill;
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        if (prefill == null) {
            throw new IllegalStateException(
                    "the deepseek4 batched decode layers are built after the prefill layers");
        }
        return new DeepSeek4Layers(
                state,
                weights,
                config,
                schedulerType,
                "decodeActivation",
                prefill,
                firstLayer,
                endLayer);
    }

    /** Between layers deepseek4 carries its residual streams, not one hidden state. */
    @Override
    public FloatArray stageCarrier(State stageState) {
        return ((DeepSeek4State) stageState).workspace.deepseek4.streams;
    }

    /** A chunk's residual streams. */
    @Override
    public FloatArray batchStageCarrier(State stageState) {
        return ((DeepSeek4State) stageState).workspace.deepseek4Batch.streams;
    }
}
