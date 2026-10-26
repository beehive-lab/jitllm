package org.beehive.jitllm.backend.tornado.layers.type.q4_0.decode;

import org.beehive.jitllm.backend.tornado.layers.type.q4_0.LlamaQ4_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * The decode layers of the batched prefill/decode plan for Llama with Q4_0 weights: the Q4_0
 * single-token layers, bound to what the batched prefill graphs have already put on the device.
 *
 * <p>Layer 0 consumes the key/value cache the decode activation relays from the last prefill layer,
 * and every layer binds the weights its {@code batchPrefillLayer_<i>} graph uploaded rather than a
 * second copy. The Q8_0 counterpart is {@code LlamaQ8_0FFNLayersDecode}.
 */
public class LlamaQ4_0FFNLayersDecode extends LlamaQ4_0FFNLayers {

    public LlamaQ4_0FFNLayersDecode(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     */
    public LlamaQ4_0FFNLayersDecode(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType,
            int firstLayer,
            int endLayer) {
        super(state, weights, config, schedulerType, firstLayer, endLayer);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == firstLayer) ? "decodeActivation" : layerGraphName(layerIndex - 1);
    }

    // @formatter:off
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == firstLayer) {
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
                    state.workspace.wrapHb,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            layer.consumeFromDevice(cacheSource(), keyCache(), valueCache());
            layer.consumeFromDevice(cacheSource(), state.workspace.wrapBlockTable);
        } else {
            String pred = layerGraphName(layerIndex - 1);
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
                    state.workspace.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return layer;
    }

    // @formatter:on

    /**
     * The graph the first layer takes the key/value cache and block table from. On the whole model
     * it is the decode activation, which relays them from the last prefill layer. A later pipeline
     * stage starts with a graph that receives the hidden state from the previous stage, and its
     * first layer takes them straight from the stage's own last prefill layer instead.
     */
    private String cacheSource() {
        return firstLayer == 0
                ? "decodeActivation"
                : "batchPrefillLayer_" + (endLayer(config.numberOfLayers()) - 1);
    }

    /** The matching prefill graph has uploaded this layer's weights and always runs first. */
    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return "batchPrefillLayer_" + layerIndex;
    }
}
