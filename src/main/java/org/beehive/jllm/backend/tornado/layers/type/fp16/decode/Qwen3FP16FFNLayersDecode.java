package org.beehive.jllm.backend.tornado.layers.type.fp16.decode;

import java.util.ArrayList;
import java.util.List;
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
        // Not "batchPrefillLayer_" + layerIndex: when the batch-prefill side groups layers into
        // one graph, that graph is named after the first layer of its group and is the one that
        // uploaded this layer's weights.
        return Qwen3FP16LayersBatchPrefillMMA.prefillGraphOwning(layerIndex);
    }

    /**
     * Every weight the batch-prefill layer graph declares but hands to no task, so it never
     * uploads it and this graph must.
     *
     * <p>Each native projection replaces a JIT task that read the model's own weights with one
     * cuBLAS GEMM over a stacked copy: gate/up takes {@code w1}/{@code w3} out of the graph, QKV
     * takes {@code wq}/{@code wk}/{@code wv}. The flags are independent, so the two contributions
     * combine. The predicates are the producer's own, which is what keeps the two sides from
     * drifting apart — including the backend gate, since neither flag selects anything where the
     * MMA batch-prefill class is not the one being built.
     *
     * <p>This graph's own kernels still read all five: {@code rms_ffn_gate_up} reads w1 and w3,
     * {@code attn_rms_qkv_projection} reads wq, wk and wv.
     */
    @Override
    protected Object[] weightsNotProvidedBySource(int layerIndex) {
        boolean gateUp = Qwen3FP16LayersBatchPrefillMMA.nativeGateUpProjection();
        boolean qkv = Qwen3FP16LayersBatchPrefillMMA.nativeQkvProjection();
        if (!gateUp && !qkv) {
            return super.weightsNotProvidedBySource(layerIndex);
        }
        List<Object> notProvided = new ArrayList<>(5);
        if (qkv) {
            notProvided.add(weights.wqLayered[layerIndex].asHalfFloatArray());
            notProvided.add(weights.wkLayered[layerIndex].asHalfFloatArray());
            notProvided.add(weights.wvLayered[layerIndex].asHalfFloatArray());
        }
        if (gateUp) {
            notProvided.add(weights.w1Layered[layerIndex].asHalfFloatArray());
            notProvided.add(weights.w3Layered[layerIndex].asHalfFloatArray());
        }
        return notProvided.toArray();
    }
}
