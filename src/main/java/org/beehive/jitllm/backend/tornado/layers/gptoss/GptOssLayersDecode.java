package org.beehive.jitllm.backend.tornado.layers.gptoss;

import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.weights.tornado.GptOssTornadoWeights;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * The single-token layers of a batched plan: the same tasks as {@link GptOssLayers}, reading the
 * weights, the caches and the RoPE tables the batched-prefill graph uploaded, and chained by name.
 */
public final class GptOssLayersDecode extends GptOssLayers {

    public GptOssLayersDecode(
            GptOssState state,
            GptOssTornadoWeights weights,
            GptOssConfiguration config,
            SchedulerType schedulerType,
            String prefillGraph) {
        super(state, weights, config, schedulerType, prefillGraph);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return layerIndex == firstLayer
                ? GraphNames.DECODE_ACTIVATION
                : GraphNames.layer(layerIndex - 1);
    }

    @Override
    protected void uploadWeights(TaskGraph layer, int l) {
        Object[] provided =
                UnifiedMemoryWeights.consume(
                        layer, state.workspace, GptOssBatchPrefillLayers.layerWeights(weights, l));
        if (provided.length > 0) {
            layer.consumeFromDevice(prefillGraph(), provided);
        }
        if (l == firstLayer) {
            layer.consumeFromDevice(
                    prefillGraph(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        } else {
            layer.consumeFromDevice(
                    GraphNames.layer(l - 1),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        }
    }

    private String prefillGraph() {
        return prefillGraphName;
    }

    @Override
    protected void transfers(TaskGraph layer, int l) {
        var ws = gptState.workspace;
        if (l == firstLayer) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    ws.wrapXb,
                    ws.wrapXb2,
                    ws.wrapQ,
                    ws.wrapK,
                    ws.wrapV,
                    ws.wrapRouterLogits,
                    ws.wrapSelectedExperts,
                    ws.wrapRoutingWeights,
                    ws.wrapExpertGate,
                    ws.wrapExpertDown,
                    ws.wrapAttSplit,
                    ws.wrapXbQuants,
                    ws.wrapXbScales,
                    ws.wrapXbSums);
            layer.consumeFromDevice(
                    GraphNames.DECODE_ACTIVATION, keyCache(), valueCache(), ws.wrapBlockTable);
        } else {
            String predecessor = GraphNames.layer(l - 1);
            layer.consumeFromDevice(predecessor, scratch());
            layer.consumeFromDevice(predecessor, ws.positionHolder, ws.wrapBlockTable);
        }
    }
}
