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

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanPrefillDecode;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan ({@link
 * TornadoVMMasterPlanPrefillDecode}).
 *
 * <p>Layer 0 delegates to {@link LlamaQ8_0FFNLayers#configureLayerDataTransfers} which includes
 * {@code FIRST_EXECUTION} for {@code wrapKeyCache} and {@code wrapValueCache}, allocating the KV
 * cache on the very first forward pass. Layers 1+ use explicit predecessor names for all consumed
 * objects, required by TornadoVM's interpreter mode.
 */
public class LlamaQ8_0FFNLayersPrefillDecode extends LlamaQ8_0FFNLayers {

    // @formatter:off
    public LlamaQ8_0FFNLayersPrefillDecode(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    // @formatter:on

    /**
     * One layer per graph: the sequential prefill-decode plan addresses its layer graphs by layer
     * index, so it cannot run grouped ones.
     */
    @Override
    protected int layersPerGraph() {
        return 1;
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? GraphNames.DECODE_ACTIVATION : layerGraphName(layerIndex - 1);
    }

    // @formatter:off
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = layerGraphName(layerIndex - 1);
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
                state.workspace.positionHolder);
        layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        if (splitsGateUp()) {
            layer.consumeFromDevice(pred, state.workspace.wrapHb2);
        }
        return layer;
    }
    // @formatter:on
}
