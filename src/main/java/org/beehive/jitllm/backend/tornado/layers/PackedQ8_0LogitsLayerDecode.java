/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
