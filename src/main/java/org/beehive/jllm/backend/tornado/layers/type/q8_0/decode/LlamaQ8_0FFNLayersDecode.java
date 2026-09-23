package org.beehive.jllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jllm.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan ({@link
 * org.beehive.jllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>Layer 0 consumes the KV cache from device (passed through by the decode activation graph,
 * which relays it from the last batch prefill layer). No FIRST_EXECUTION allocation for the KV
 * cache — it was already allocated in the batch prefill phase.
 */
public class LlamaQ8_0FFNLayersDecode extends LlamaQ8_0FFNLayers {

    // @formatter:off
    public LlamaQ8_0FFNLayersDecode(
            String taskGraph,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    // @formatter:on

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    // @formatter:off
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.positionHolder,
                    state.workspace.temp,
                    state.workspace.tempFFN);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb);
            layer.consumeFromDevice("decodeActivation", keyCache(), valueCache());
            layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        } else {
            String pred = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.positionHolder);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return layer;
    }

    // @formatter:on

    /**
     * This class builds the decode half of a batched prefill/decode plan, where the matching {@code
     * batchPrefillLayer_<i>} graph has already uploaded this layer's weights and always runs first,
     * so the decode graph binds that copy instead of a second one.
     */
    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return "batchPrefillLayer_" + layerIndex;
    }
}
