package org.beehive.jitllm.backend.tornado.layers.type.q4_k.decode;

import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.type.q4_k.DevstralQ4_KFFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * The decode half of Devstral's batched prefill/decode plan: Devstral's single-token layers, with
 * the key/value cache and block table relayed by {@code decodeActivation} from the batched prefill,
 * and each layer's weights read from the {@code batchPrefillLayer_<i>} graph that already uploaded
 * them.
 */
public class DevstralQ4_KFFNLayersDecode extends DevstralQ4_KFFNLayers {

    public DevstralQ4_KFFNLayersDecode(
            State state,
            LlamaTornadoWeights weights,
            DevstralConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return layerIndex == 0 ? GraphNames.DECODE_ACTIVATION : GraphNames.layer(layerIndex - 1);
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
                    state.workspace.wrapHb2,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, keyCache(), valueCache());
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, state.workspace.wrapBlockTable);
        } else {
            String pred = GraphNames.layer(layerIndex - 1);
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
                    state.workspace.wrapHb2,
                    state.workspace.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return layer;
    }
}
