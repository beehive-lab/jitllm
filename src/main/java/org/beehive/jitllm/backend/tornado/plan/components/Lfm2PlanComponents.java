package org.beehive.jitllm.backend.tornado.plan.components;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.Lfm2BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.layers.Lfm2BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.Lfm2Layers;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;

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
public class Lfm2PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

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
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Lfm2Layers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(state, weights, config, previousGraphId, schedulerType) {
            @Override
            protected boolean simdgroupQ8_0Vocabulary() {
                // Metal: on CUDA the per-row kernel is the faster of the two.
                return SchedulerDetectionService.isSubgroupShuffle32Supported();
            }

            @Override
            protected boolean warpQ8_0Vocabulary() {
                // CUDA: one warp per row instead of a workgroup per row.
                return TornadoDevices.current().backend().equals(BackendId.CUDA);
            }
        };
    }

    // ── Sequential prefill/decode ─────────────────────────────────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation(GraphNames.DECODE_ACTIVATION, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Lfm2Layers(state, weights, config, schedulerType, GraphNames.DECODE_ACTIVATION);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }

    // ── Batched prefill/decode ────────────────────────────────────────────────

    /**
     * The chunk's activation: a Q8_0 embedding decoded on the device from the rows the host copies,
     * any other (the 8B's Q6_K) decoded by the host into the FP32 carrier and passed through.
     */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        DataType embedding = weights.getTokenEmbeddingTable().dataType();
        if (embedding == DataType.Q8_0) {
            return new BatchPrefillQ8DeviceActivation(state, config, batchSize, embedding);
        }
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new Lfm2BatchPrefillLayers(state, weights, config, batchSize);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new Lfm2BatchDecodeActivation(state, weights, config, lastBatchLayerId);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Lfm2Layers(
                state, weights, config, schedulerType, GraphNames.DECODE_ACTIVATION, true);
    }
}
