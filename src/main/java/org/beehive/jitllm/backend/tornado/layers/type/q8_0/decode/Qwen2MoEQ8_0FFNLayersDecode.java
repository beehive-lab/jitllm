package org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Qwen2MoEQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/** Single-token decode layers that continue from the batch-prefill KV cache. */
public final class Qwen2MoEQ8_0FFNLayersDecode extends Qwen2MoEQ8_0FFNLayers {

    public Qwen2MoEQ8_0FFNLayersDecode(
            String taskGraph,
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    /** Connects decode layer 0 to the decode activation and later layers in order. */
    @Override
    protected String predecessorGraphName(int layerIndex) {
        return layerIndex == 0 ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    /** Reuses the weights already uploaded by the corresponding batch-prefill layer. */
    @Override
    protected TaskGraph configureLayerWeights(TaskGraph layer, int layerIndex) {
        return layer.consumeFromDevice(
                "batchPrefillLayer_" + layerIndex,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.routerGateLayered[layerIndex].asFloatArray(),
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
    }

    /** Reuses the KV cache produced by batch prefill instead of allocating a new cache. */
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    moeState.workspace.positionHolder,
                    moeState.workspace.temp,
                    moeState.workspace.tempFFN);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    moeState.workspace.wrapXb,
                    moeState.workspace.wrapXb2,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    moeState.workspace.wrapAtt,
                    moeState.workspace.wrapRouterLogits,
                    moeState.workspace.wrapSelectedExperts,
                    moeState.workspace.wrapRoutingWeights,
                    moeState.workspace.wrapExpertGate,
                    moeState.workspace.wrapSharedGate,
                    moeState.workspace.wrapSharedOutput);
            layer.consumeFromDevice(
                    "decodeActivation",
                    moeState.workspace.wrapKeyCache,
                    moeState.workspace.wrapValueCache);
            layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        } else {
            String predecessor = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    moeState.workspace.wrapXb,
                    moeState.workspace.wrapXb2,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    moeState.workspace.wrapKeyCache,
                    moeState.workspace.wrapValueCache,
                    moeState.workspace.wrapAtt,
                    moeState.workspace.wrapRouterLogits,
                    moeState.workspace.wrapSelectedExperts,
                    moeState.workspace.wrapRoutingWeights,
                    moeState.workspace.wrapExpertGate,
                    moeState.workspace.wrapSharedGate,
                    moeState.workspace.wrapSharedOutput,
                    moeState.workspace.positionHolder,
                    moeState.workspace.temp,
                    moeState.workspace.tempFFN);
            layer.consumeFromDevice(predecessor, state.workspace.wrapBlockTable);
        }
        return layer;
    }
}
