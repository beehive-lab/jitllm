package org.beehive.jitllm.backend.tornado.plan.components.gptoss;

import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.gptoss.GptOssBatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.layers.gptoss.GptOssLayers;
import org.beehive.jitllm.backend.tornado.layers.gptoss.GptOssLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.gptoss.GptOssLogitsLayers;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.weights.tornado.GptOssTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;

/**
 * The gpt-oss plans: single-token (the Q8_0 embedding row, the layers, the Q8_0 logits), and the
 * batched prefill with its single-token decode reading what the prefill graph uploaded.
 */
public final class GptOssPlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final GptOssState state;
    private final GptOssTornadoWeights weights;
    private final GptOssConfiguration config;
    private final SchedulerType schedulerType;

    public GptOssPlanComponents(GptOssState state, Model model) {
        this.state = state;
        this.config = (GptOssConfiguration) model.configuration();
        this.weights = (GptOssTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new GptOssLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new GptOssLogitsLayers.Single(
                state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException("gpt-oss has no sequential prefill/decode plan");
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException("gpt-oss has no sequential prefill/decode plan");
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new GptOssLogitsLayers.Decode(
                state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(
                state, config, lastBatchLayerId, weights.getTokenEmbeddingTable().dataType());
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new GptOssLayersDecode(
                state,
                weights,
                config,
                schedulerType,
                org.beehive.jitllm.backend.tornado.layers.GraphNames.batchPrefillLayer(0));
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        return new GptOssBatchPrefillLayers(state, weights, config, batchSize);
    }
}
