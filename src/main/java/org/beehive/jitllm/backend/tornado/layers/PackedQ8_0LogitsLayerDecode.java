package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;

/**
 * {@link PackedQ8_0LogitsLayer} for the decode graphs of a prefill-decode plan: the logits of a
 * family whose layers run their Q8_0 projections as packed integer dot products, read the same way.
 */
public class PackedQ8_0LogitsLayerDecode extends LogitsQ8_0LayerDecode {

    public PackedQ8_0LogitsLayerDecode(
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(state, weights, config, lastTaskGraphID, schedulerType);
    }

    @Override
    protected boolean familyProjectsQ8_0Packed() {
        return true;
    }
}
