package org.beehive.jllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jllm.backend.tornado.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan (Qwen3 Q8_0).
 *
 * <p>Layer 0: KV cache consumed from "decodeActivation" (already allocated by batch prefill).
 * Layers 1+: all consumed objects use explicit predecessor name for interpreter mode.
 */
public class Qwen3Q8_0FFNLayersDecode extends Qwen3Q8_0FFNLayers {

    public Qwen3Q8_0FFNLayersDecode(
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
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.tempQcur,
                    qwen3State.workspace.tempKcur);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
            // KV cache already allocated by batch prefill; relay from decode activation graph.
            layer.consumeFromDevice("decodeActivation", keyCache(), valueCache());
            layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        } else {
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
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
            layer.consumeFromDevice(qwen3State.workspace.tempQcur, qwen3State.workspace.tempKcur);
            layer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
        }
        return layer;
    }

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
