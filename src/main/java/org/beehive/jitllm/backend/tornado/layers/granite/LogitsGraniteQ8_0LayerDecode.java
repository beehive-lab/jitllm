package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsGraniteQ8_0Layer;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * {@link LogitsGraniteQ8_0Layer} for the decode half of a batched prefill/decode plan: it consumes
 * the key/value cache from the last decode layer by name and persists it, as {@code
 * LogitsFP16LayerDecode} does.
 */
public class LogitsGraniteQ8_0LayerDecode extends LogitsGraniteQ8_0Layer {

    public LogitsGraniteQ8_0LayerDecode(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

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
