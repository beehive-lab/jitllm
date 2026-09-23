package org.beehive.jllm.backend.tornado.plan.components;

import org.beehive.jllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jllm.backend.tornado.layers.Activation;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.Qwen35BatchDecodeActivation;
import org.beehive.jllm.backend.tornado.layers.Qwen35BatchPrefillLayers;
import org.beehive.jllm.backend.tornado.layers.Qwen35FFNLayers;
import org.beehive.jllm.backend.tornado.layers.Qwen35FFNLayersBatchDecode;
import org.beehive.jllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;

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
public class Qwen35PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

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
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen35FFNLayers("qwen35FFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    // ── Sequential prefill/decode ─────────────────────────────────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen35FFNLayers(
                "qwen35FFN", state, weights, config, schedulerType, "decodeActivation");
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }

    // ── Batched prefill/decode ────────────────────────────────────────────────

    /**
     * The chunk's activation.
     *
     * <p>The host decodes the chunk's embedding rows straight into the FP32 batch carrier — a Q4_0
     * row is 18 bytes per 32 weights and there is no batched device conversion for it — so this
     * graph transfers that carrier and runs the shared pass-through, which exists to give TornadoVM
     * a task to attach the transfer to.
     */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new Qwen35BatchPrefillLayers(state, weights, config, batchSize);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Qwen35BatchDecodeActivation(state, weights, config, lastBatchLayerId);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen35FFNLayersBatchDecode("qwen35FFN", state, weights, config, schedulerType);
    }
}
