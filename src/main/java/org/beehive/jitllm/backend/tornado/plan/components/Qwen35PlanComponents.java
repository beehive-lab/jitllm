package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedQ8_0LogitsLayer;
import org.beehive.jitllm.backend.tornado.layers.Qwen35BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.layers.Qwen35BatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.Qwen35BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayersBatchDecode;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayersGrouped;
import org.beehive.jitllm.backend.tornado.layers.Qwen35FeedForward;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code qwen35} single-token plan: an activation graph, one graph per trunk layer, and a
 * logits graph.
 *
 * <p>Not in a per-representation package, because there is no single representation to name it
 * after. The activation and logits graphs already dispatch on the tensor they read — the embedding
 * table and the vocabulary projection — and the layer graphs do the same per weight, so one set of
 * components serves the whole family rather than one per dtype.
 *
 * <p>Sequential prefill reuses the same layer graphs: prompt ingestion is the decode computation
 * with the logits graph skipped, and the recurrent state it advances is the same device buffer
 * decode continues from. Only the graph layer 0 consumes from differs.
 *
 * <p>Batched prefill has its own layer graphs — a chunk is not a token, and two of this family's
 * kernels have to scan it in order — but the same decode graphs behind them, reading what those
 * chunks left on the device.
 */
public class Qwen35PlanComponents
        implements BatchPrefillDecodeForwardPlanComponents, StagedForwardPlanComponents {

    private final Qwen35State state;
    private final Qwen35TornadoWeights weights;
    private final Qwen35Configuration config;
    private final SchedulerType schedulerType;

    public Qwen35PlanComponents(Qwen35State state, Model model) {
        this.state = state;
        this.config = (Qwen35Configuration) model.configuration();
        this.weights = (Qwen35TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        // The whole model as one stage: several layers to a graph, as in the pipeline stages.
        return singleTokenTransformerLayers(0, config.numberOfLayers());
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new PackedQ8_0LogitsLayer(state, weights, config, previousGraphId, schedulerType);
    }

    // ── Sequential prefill/decode ─────────────────────────────────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation(GraphNames.DECODE_ACTIVATION, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen35FFNLayers(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.DECODE_ACTIVATION,
                0,
                config.numberOfLayers(),
                feedForward());
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }

    // ── Batched prefill/decode ────────────────────────────────────────────────

    /**
     * The chunk's activation.
     *
     * <p>A Q8_0 or Q4_0 embedding is decoded on the device: the host copies each token's raw row,
     * which costs nothing next to decoding the chunk in Java before the JIT has compiled that loop.
     * Any other embedding the host decodes straight into the FP32 batch carrier, and this graph
     * transfers it and runs the shared pass-through, which exists to give TornadoVM a task to
     * attach the transfer to.
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
        return new Qwen35BatchPrefillLayers(
                state,
                weights,
                config,
                batchSize,
                0,
                config.numberOfLayers(),
                batchFeedForward(batchSize));
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Qwen35BatchDecodeActivation(state, weights, config, lastBatchLayerId);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen35FFNLayersBatchDecode(
                state, weights, config, schedulerType, 0, config.numberOfLayers(), feedForward());
    }

    // ── A family built on this one ────────────────────────────────────────────

    /** The single-token feed-forward in place of the dense SwiGLU, or {@code null} for it. */
    protected Qwen35FeedForward feedForward() {
        return null;
    }

    /** The batched-prefill feed-forward in place of the dense SwiGLU, or {@code null} for it. */
    protected Qwen35BatchFeedForward batchFeedForward(int batchSize) {
        return null;
    }

    // ── Pipeline stages ───────────────────────────────────────────────────────

    /**
     * Several layers to a graph: each graph is a launch and a synchronization per token, which is
     * most of the time between kernels for the mixture-of-experts layers.
     */
    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers(int firstLayer, int endLayer) {
        return new Qwen35FFNLayersGrouped(
                state,
                weights,
                config,
                schedulerType,
                GraphNames.ACTIVATION_UPDATE,
                firstLayer,
                endLayer,
                feedForward());
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(
            int batchSize, int firstLayer, int endLayer) {
        return new Qwen35BatchPrefillLayers(
                state,
                weights,
                config,
                batchSize,
                firstLayer,
                endLayer,
                batchFeedForward(batchSize));
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers(int firstLayer, int endLayer) {
        return new Qwen35FFNLayersBatchDecode(
                state, weights, config, schedulerType, firstLayer, endLayer, feedForward());
    }
}
