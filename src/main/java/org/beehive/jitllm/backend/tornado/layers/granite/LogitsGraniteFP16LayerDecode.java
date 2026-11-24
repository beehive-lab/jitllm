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
package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.layers.type.fp16.LogitsGraniteFP16Layer;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * {@link LogitsGraniteFP16Layer} for the decode half of a batched prefill/decode plan: it consumes
 * the key/value cache from the last decode layer by name and persists it, as {@code
 * LogitsFP16LayerDecode} does.
 */
public class LogitsGraniteFP16LayerDecode extends LogitsGraniteFP16Layer {

    public LogitsGraniteFP16LayerDecode(
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(state, weights, config, lastTaskGraphID, schedulerType);
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
