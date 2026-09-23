package org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan (Qwen3 Q8_0).
 *
 * <p>Layer 0 delegates to the base-class which allocates wrapKeyCache/wrapValueCache with
 * FIRST_EXECUTION. Layers 1+ consume all live buffers from the explicit predecessor graph.
 */
public class Qwen3Q8_0FFNLayersPrefillDecode extends Qwen3Q8_0FFNLayers {

    public Qwen3Q8_0FFNLayersPrefillDecode(
            String taskGraph,
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = "layer_" + (layerIndex - 1);
        layer.consumeFromDevice(
                pred,
                context,
                qwen3State.workspace.wrapXb,
                qwen3State.workspace.wrapXb2,
                qwen3State.workspace.wrapQ,
                qwen3State.workspace.wrapK,
                qwen3State.workspace.wrapV,
                keyCache(),
                valueCache(),
                qwen3State.workspace.wrapAtt,
                qwen3State.workspace.wrapHb,
                qwen3State.workspace.positionHolder);
        layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        layer.consumeFromDevice(qwen3State.workspace.tempQcur, qwen3State.workspace.tempKcur);
        layer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
        return layer;
    }
}
