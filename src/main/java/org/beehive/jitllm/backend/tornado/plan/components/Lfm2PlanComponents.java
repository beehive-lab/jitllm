package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Lfm2Layers;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;

/**
 * The {@code lfm2moe} single-token plan: an activation graph, one graph per layer, and a logits
 * graph.
 *
 * <p>One set of components for the whole family rather than one per representation: the activation
 * and logits graphs dispatch on the tensor they read, and the layer graphs per weight.
 *
 * <p>Sequential prefill reuses the same layer graphs with the logits graph skipped: prompt
 * ingestion is the decode computation without the vocabulary projection, and the convolution
 * windows it advances are the same device buffer decode continues from. Only the graph layer 0
 * consumes from differs.
 */
public class Lfm2PlanComponents implements PrefillDecodeForwardPlanComponents {

    private final Lfm2State state;
    private final Lfm2TornadoWeights weights;
    private final Lfm2Configuration config;
    private final SchedulerType schedulerType;

    public Lfm2PlanComponents(Lfm2State state, Model model) {
        this.state = state;
        this.config = (Lfm2Configuration) model.configuration();
        this.weights = (Lfm2TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Lfm2Layers("lfm2", state, weights, config, schedulerType);
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
        return new Lfm2Layers("lfm2", state, weights, config, schedulerType, "decodeActivation");
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }
}
