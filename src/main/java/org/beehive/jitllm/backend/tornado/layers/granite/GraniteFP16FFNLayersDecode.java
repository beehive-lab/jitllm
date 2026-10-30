package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * The decode half of a Granite batched prefill/decode plan: the {@link GraniteFP16FFNLayers}
 * graphs, fed by the decode activation, binding the weights, key/value cache and block table that
 * the matching {@code batchPrefillLayer_<i>} graph already holds on the device.
 */
public class GraniteFP16FFNLayersDecode extends GraniteFP16FFNLayers {

    public GraniteFP16FFNLayersDecode(
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? GraphNames.DECODE_ACTIVATION : layerGraphName(layerIndex - 1);
    }

    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return GraphNames.batchPrefillLayer(layerIndex);
    }

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
                    state.workspace.wrapHb,
                    state.workspace.wrapXbFP16);
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, keyCache(), valueCache());
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, state.workspace.wrapBlockTable);
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
                    state.workspace.wrapXbFP16);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return layer;
    }
}
