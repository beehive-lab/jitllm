package org.beehive.jllm.backend.tornado.layers.type.fp16.decode;

import org.beehive.jllm.backend.tornado.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillMMA;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan (Qwen3 FP16).
 *
 * <p>Layer 0: KV cache is consumed from "decodeActivation" (already allocated by the batch prefill
 * phase). Working buffers get FIRST_EXECUTION allocation. Layers 1+: all consumed objects use the
 * explicit predecessor name to satisfy TornadoVM interpreter mode.
 *
 * <p>Qwen3FP16FFNLayers does not use wrapXbFP16 in any task, so it is excluded.
 */
public class Qwen3FP16FFNLayersDecode extends Qwen3FP16FFNLayers {

    public Qwen3FP16FFNLayersDecode(
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
        Object keyCache =
                useFp16KVCache()
                        ? qwen3State.workspace.wrapKeyCacheFP16
                        : qwen3State.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? qwen3State.workspace.wrapValueCacheFP16
                        : qwen3State.workspace.wrapValueCache;
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
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
            layer.consumeFromDevice("decodeActivation", keyCache, valueCache);
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
                    keyCache,
                    valueCache,
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
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

    /**
     * With the native gate/up projection the batch-prefill layer graph computes from one stacked
     * [gate|up] weight, so {@code w1} and {@code w3} are arguments to no task in it and it never
     * uploads them. This graph's fused gate/up kernel still reads both, so it uploads them itself.
     */
    @Override
    protected Object[] weightsNotProvidedBySource(int layerIndex) {
        if (!Qwen3FP16LayersBatchPrefillMMA.nativeGateUpProjection()) {
            return super.weightsNotProvidedBySource(layerIndex);
        }
        return new Object[] {
            weights.w1Layered[layerIndex].asHalfFloatArray(),
            weights.w3Layered[layerIndex].asHalfFloatArray()
        };
    }
}
