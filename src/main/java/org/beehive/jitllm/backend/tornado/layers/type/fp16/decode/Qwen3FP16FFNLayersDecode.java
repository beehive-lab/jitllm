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

    // @formatter:off
    /**
     * Ten transformer layers to a decode graph.
     *
     * <p>Decode submits a graph and waits for it, once per graph, per token: {@code
     * TaskGraph.execute} is {@code taskGraphImpl.execute(frame).waitOn()}, with no way to defer the
     * wait. This family was submitting thirty graphs — one activation, twenty-eight layers, one
     * logits — and an nsys timeline showed 810 us of a 3.49 ms token going into the gaps between
     * them. Grouping is what removes those gaps, by having fewer of them.
     *
     * <p><b>Ten, and why not four.</b> Four was the figure the batch-prefill side settled on and
     * was carried over here without being measured against alternatives. It is not a capacity
     * limit: {@code TornadoTaskGraph} encodes a graph into a {@code private byte[8192]} that never
     * grows, a decode layer costs about 620 bytes of it, and four layers use 2480 — 30%. Thirteen
     * would fit. Measured in one interleaved session at four, seven and ten layers, tg128 reads
     * 458.5 / 465.3 / <b>466.8</b> tok/s at depth 0, 435.4 / 441.3 / <b>443.1</b> at 512 and 391.1
     * / 395.7 / <b>397.3</b> at 2048 — 1.6 to 1.8% over four, consistent across four rounds to the
     * third decimal.
     *
     * <p>Ten rather than thirteen leaves a quarter of the buffer spare, which is what absorbs a
     * layer gaining a task; {@code DecodeGraphBytecodeAccelTest} fails before that margin is gone.
     * The gain is sublinear in the submissions removed — going from nine graphs a token to five
     * removed 98 us of device idle, about 24 us per graph rather than the 47 us each one costs,
     * because part of the per-launch host cost scales with the graph's size rather than with the
     * number of launches.
     *
     * <p>Groups need not divide the layer count: twenty-eight layers become graphs of ten, ten and
     * eight, and every name is derived through {@link Qwen3FP16FFNLayers#layerGraphName} rather
     * than assumed. Not a tuning property — see {@link Qwen3FP16FFNLayers#layersPerGraph()}.
     */
    // @formatter:on
    public static final int LAYERS_PER_GRAPH = 10;

    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    /**
     * The graph that ran before this one: the decode activation for the first group, the previous
     * group's graph otherwise. Only the layer that opens a graph asks — the rest are already in the
     * graph that holds the buffer.
     */
    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : layerGraphName(layerIndex - 1);
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
            String pred = layerGraphName(layerIndex - 1);
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
     * Every weight the batch-prefill layer graph declares but hands to no task, so it never uploads
     * it and this graph must.
     *
     * <p>The native projections replace JIT tasks that read the model's own weights with cuBLAS
     * GEMMs over stacked copies: gate/up takes {@code w1}/{@code w3} out of the graph, QKV takes
     * {@code wq}/{@code wk}/{@code wv}. All five go together, because the producer decides all four
     * projections with one answer. The predicate asked here is the producer's own — that is what
     * keeps the two sides from drifting apart, including the backend gate, since it selects nothing
     * where the MMA batch-prefill class is not the one being built.
     *
     * <p>This graph's own kernels still read all five: {@code rms_ffn_gate_up} reads w1 and w3,
     * {@code attn_rms_qkv_projection} reads wq, wk and wv.
     */
    @Override
    protected Object[] weightsNotProvidedBySource(int layerIndex) {
        if (!Qwen3FP16LayersBatchPrefillMMA.nativeProjections(state.executionPolicy())) {
            return super.weightsNotProvidedBySource(layerIndex);
        }
        List<Object> notProvided = new ArrayList<>(5);
        notProvided.add(weights.wqLayered[layerIndex].asHalfFloatArray());
        notProvided.add(weights.wkLayered[layerIndex].asHalfFloatArray());
        notProvided.add(weights.wvLayered[layerIndex].asHalfFloatArray());
        notProvided.add(weights.w1Layered[layerIndex].asHalfFloatArray());
        notProvided.add(weights.w3Layered[layerIndex].asHalfFloatArray());
        return notProvided.toArray();
    }
}
