package org.beehive.jitllm.backend.tornado.plan.components.fp16;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationGranite;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteBatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteFP16FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteFP16LayersBatchPrefill;
import org.beehive.jitllm.backend.tornado.layers.granite.GraniteFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.layers.granite.LogitsGraniteFP16LayerDecode;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.LogitsGraniteFP16Layer;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.granite.GraniteConfiguration;

public class GraniteFP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final SchedulerType schedulerType;

    public GraniteFP16PlanComponents(GraniteState state, Model model) {
        this.state = state;
        this.config = (GraniteConfiguration) model.configuration();
        this.weights = (GraniteTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new ActivationGranite("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new GraniteFP16FFNLayers(state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsGraniteFP16Layer(state, weights, config, previousGraphId, schedulerType);
    }

    /** Sequential prefill/decode has no Granite layers. */
    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        throw new UnsupportedOperationException("PREFILL_DECODE not yet supported for Granite");
    }

    /** The embedding multiplier is applied by the first batch layer. */
    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, false);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new GraniteBatchDecodeActivation(state, config, lastBatchLayerId, false);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        throw new UnsupportedOperationException("PREFILL_DECODE not yet supported for Granite");
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new GraniteFP16FFNLayersDecode(state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new GraniteFP16LayersBatchPrefillMMA(state, weights, config, batchSize);
        }
        return new GraniteFP16LayersBatchPrefill(state, weights, config, batchSize);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsGraniteFP16LayerDecode(
                state, weights, config, previousGraphId, schedulerType);
    }
}
