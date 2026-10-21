package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;

/**
 * The logits of a family whose layers run their Q8_0 projections as packed integer dot products
 * (qwen35, deepseek2): a Q8_0 output projection is read the same way.
 */
public class PackedQ8_0LogitsLayer extends LogitsQ8_0Layer {

    public PackedQ8_0LogitsLayer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    @Override
    protected boolean familyProjectsQ8_0Packed() {
        return true;
    }
}
