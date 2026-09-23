package org.beehive.jitllm.backend.tornado.layers.type.fp16.decode;

import org.beehive.jitllm.backend.tornado.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan ({@link
 * org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanPrefillDecode}).
 *
 * <p>Combines two concerns:
 *
 * <ol>
 *   <li><b>Correct predecessor names</b> — overrides {@link #predecessorGraphName} so that every
 *       cross-graph {@code consumeFromDevice} uses the explicit-source form required by TornadoVM's
 *       interpreter (non-CUDA-graph) mode. Layer 0 names {@code "decodeActivation"}; layers 1+ name
 *       {@code "layer_"+(k-1)}.
 *   <li><b>KV-cache allocation</b> — layer 0 delegates to the base-class {@link
 *       #configureLayerDataTransfers} which includes {@code FIRST_EXECUTION} for {@code
 *       wrapKeyCache} and {@code wrapValueCache}. This allocates the KV-cache device buffers on the
 *       very first forward pass; subsequent passes skip the re-upload and the GPU accumulates
 *       entries in place. Layers 1+ use {@code consumeFromDevice} with an explicit predecessor name
 *       for all objects, matching {@link LlamaFP16FFNLayersDecode}.
 * </ol>
 *
 * <p>The activation graph ("decodeActivation") only persists {@code wrapX} — it does not touch the
 * KV cache. Layer 0 is therefore the sole allocator of the KV cache, which avoids the NPE in {@code
 * executeAlloc} that occurs when {@code consumeFromDevice} targets an object whose device buffer
 * was never properly allocated via {@code FIRST_EXECUTION}.
 */
public class LlamaFP16FFNLayersPrefillDecode extends LlamaFP16FFNLayers {

    public LlamaFP16FFNLayersPrefillDecode(
            String taskGraph,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    /**
     * Layer 0 receives {@code wrapX} from the decode activation graph; layers 1+ receive it from
     * the previous decode layer.
     */
    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    /**
     * Layer 0: delegates to the base class (FIRST_EXECUTION for wrapKeyCache/wrapValueCache + all
     * working buffers). KV cache is allocated here on the first forward pass.
     *
     * <p>Layers 1+: mirrors {@link LlamaFP16FFNLayersDecode} — {@code consumeFromDevice} with an
     * explicit predecessor name for every object, required by interpreter mode.
     */
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = "layer_" + (layerIndex - 1);
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
        return layer;
    }
}
