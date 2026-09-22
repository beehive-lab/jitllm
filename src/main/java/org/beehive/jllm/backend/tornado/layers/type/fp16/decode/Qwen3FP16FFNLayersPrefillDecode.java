package org.beehive.jllm.backend.tornado.layers.type.fp16.decode;

import org.beehive.jllm.backend.tornado.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan (Qwen3 FP16).
 *
 * <p>Layer 0 delegates to the base-class {@link Qwen3FP16FFNLayers#configureLayerDataTransfers}
 * which allocates wrapKeyCache/wrapValueCache with FIRST_EXECUTION. Layers 1+ consume all live
 * buffers from the explicit predecessor graph to satisfy TornadoVM interpreter mode.
 *
 * <p>Note: Qwen3FP16FFNLayers does not use wrapXbFP16 in any kernel task, so it is intentionally
 * excluded from the consume list.
 */
public class Qwen3FP16FFNLayersPrefillDecode extends Qwen3FP16FFNLayers {

    public Qwen3FP16FFNLayersPrefillDecode(
            String taskGraph,
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    /**
     * The prefill/decode graph variants share the FP32 KV cache with the batch-prefill layers, so
     * the FP16 KV cache path (standard single-token mode only) is disabled here.
     */
    @Override
    protected boolean useFp16KVCache() {
        return false;
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : layerGraphName(layerIndex - 1);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = layerGraphName(layerIndex - 1);
        layer.consumeFromDevice(
                pred,
                context,
                qwen3State.workspace.wrapXb,
                qwen3State.workspace.wrapXb2,
                qwen3State.workspace.wrapQ,
                qwen3State.workspace.wrapK,
                qwen3State.workspace.wrapV,
                qwen3State.workspace.wrapKeyCache,
                qwen3State.workspace.wrapValueCache,
                qwen3State.workspace.wrapAtt,
                qwen3State.workspace.wrapHb,
                qwen3State.workspace.positionHolder);
        layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        layer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
        return layer;
    }
}
