package org.beehive.jitllm.backend.tornado.layers.gptoss;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * gpt-oss's logits: the 201088-row Q8_0 vocabulary projection read with the packed int8 dot product
 * over the final-normed hidden state quantized to Q8 blocks, in the buffers the layers' own packed
 * projections use.
 */
public final class GptOssLogitsLayers {

    private GptOssLogitsLayers() {}

    /** The single-token plan's logits. */
    public static final class Single extends LogitsQ8_0Layer {
        public Single(
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

        @Override
        protected void configureAdditionalConsumes(TaskGraph logits) {
            logits.consumeFromDevice(
                    lastTaskGraphID, PackedActivationTasks.buffers(state.workspace));
        }
    }

    /** The batched plan's decode logits, which also pass the caches along. */
    public static final class Decode extends LogitsQ8_0LayerDecode {
        public Decode(
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

        @Override
        protected void configureAdditionalConsumes(TaskGraph logits) {
            super.configureAdditionalConsumes(logits);
            logits.consumeFromDevice(
                    lastTaskGraphID, PackedActivationTasks.buffers(state.workspace));
        }
    }
}
