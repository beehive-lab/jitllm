package org.beehive.jllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Logits layer for the unified batched prefill-decode plan (Q8_0).
 *
 * <p>Extends {@link LogitsQ8_0Layer} with KV-cache pass-through so the device pointers for {@code
 * wrapKeyCache} and {@code wrapValueCache} survive the logits → decode-activation boundary between
 * decode tokens.
 *
 * <p>As in {@code BatchDecodeActivation}, this is host-side buffer-state aliasing rather than
 * device work: no task in the logits graph takes the KV cache as an argument, so the graph emits no
 * bytecode for it. Naming the producer keeps this graph's device-buffer state pointing at the last
 * decode layer's live buffers instead of falling back to whatever ran previously.
 */
public class LogitsQ8_0LayerDecode extends LogitsQ8_0Layer {

    // @formatter:off
    public LogitsQ8_0LayerDecode(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:on

    /** The cache the decode layers persist: FP16 when the state holds one, as with FP16 weights. */
    private Object keyCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private Object valueCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    @Override
    protected void configureAdditionalConsumes(TaskGraph logits) {
        logits.consumeFromDevice(lastTaskGraphID, keyCache(), valueCache());
    }

    @Override
    protected void configureAdditionalPersists(TaskGraph logits) {
        logits.persistOnDevice(keyCache(), valueCache());
    }
}
