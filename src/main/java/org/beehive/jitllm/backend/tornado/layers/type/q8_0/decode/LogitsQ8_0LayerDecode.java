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
package org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
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
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(state, weights, config, lastTaskGraphID, schedulerType);
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
