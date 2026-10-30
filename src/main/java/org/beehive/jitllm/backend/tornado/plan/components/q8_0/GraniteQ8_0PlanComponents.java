package org.beehive.jitllm.backend.tornado.plan.components.q8_0;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationGranite;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteBatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteQ8_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteQ8_0LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteQ8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.layers.granite.LogitsGraniteQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.GraniteQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsGraniteQ8_0Layer;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.granite.GraniteConfiguration;

public class GraniteQ8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final SchedulerType schedulerType;

    public GraniteQ8_0PlanComponents(GraniteState state, Model model) {
        this.state = state;
        this.config = (GraniteConfiguration) model.configuration();
        this.weights = (GraniteTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new ActivationGranite(GraphNames.ACTIVATION_UPDATE, state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new GraniteQ8_0FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsGraniteQ8_0Layer(state, weights, config, previousGraphId, schedulerType);
    }

    /** Sequential prefill/decode has no Granite layers. */
    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException("PREFILL_DECODE not yet supported for Granite");
    }

    /** The embedding multiplier is applied by the first batch layer. */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new GraniteBatchDecodeActivation(state, config, lastBatchLayerId, true);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException("PREFILL_DECODE not yet supported for Granite");
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new GraniteQ8_0FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new GraniteQ8_0LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new GraniteQ8_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsGraniteQ8_0LayerDecode(
                state, weights, config, previousGraphId, schedulerType);
    }
}
