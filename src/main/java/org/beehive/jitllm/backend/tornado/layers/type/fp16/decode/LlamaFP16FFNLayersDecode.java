package org.beehive.jitllm.backend.tornado.layers.type.fp16.decode;

import org.beehive.jitllm.backend.tornado.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs of the unified batched prefill-decode plan ({@link
 * org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>Overrides data-transfer declarations so that all cross-graph boundaries use the
 * explicit-source form of {@code consumeFromDevice}. The no-arg form (used by the base class)
 * passes the <em>current</em> graph's own name as the source key. In CUDA-graph mode this is
 * harmless (device pointers are frozen at capture time), but in interpreter mode {@code
 * updatePersistedObjectState} looks up the <em>predecessor's</em> name, so the lookup always misses
 * and the XPUBuffer is never propagated — causing either a null-pointer crash or a silent re-upload
 * from host (zeros), corrupting the hidden state and KV cache.
 */
public class LlamaFP16FFNLayersDecode extends LlamaFP16FFNLayers {
    public LlamaFP16FFNLayersDecode(
            String taskGraph,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    /**
     * Supplies the correct predecessor graph name for {@code consumeFromDevice(wrapX)}.
     *
     * <p>Layer 0 receives {@code wrapX} from the decode activation graph; layers 1+ receive it from
     * the previous decode layer. Must match the {@code TaskGraph} names used in {@code
     * buildDecodeActivationGraph()} and {@code createFFNLayerTaskGraph()}.
     */
    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
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

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        LlamaState llamaState = (LlamaState) state;
        Object keyCache =
                useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;
        if (layerIndex == 0) {
            // Same as parent layer 0, but wrapKeyCache/wrapValueCache come from device
            // (passed through by the decode activation graph, which relays them from
            // the last batch prefill layer).  No FIRST_EXECUTION for KV cache here.
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
            if (splitKvAttentionEnabled()) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
            }
            // Explicit source — must match the TaskGraph name in buildDecodeActivationGraph().
            layer.consumeFromDevice("decodeActivation", keyCache, valueCache);
            layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        } else {
            // Layers 1+: use explicit predecessor name for ALL consumed objects.
            // Calling super here would use the no-arg form (source key = own graph name),
            // which silently fails in interpreter mode and causes re-upload from host.
            String pred = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    keyCache,
                    valueCache,
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.positionHolder,
                    state.workspace.wrapXbFP16);
            if (splitKvAttentionEnabled()) {
                layer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
            }
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return layer;
    }
}
