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

import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan (Qwen3 Q8_0).
 *
 * <p>Layer 0: KV cache consumed from "decodeActivation" (already allocated by batch prefill).
 * Layers 1+: all consumed objects use explicit predecessor name for interpreter mode.
 */
public class Qwen3Q8_0FFNLayersDecode extends Qwen3Q8_0FFNLayers {

    public Qwen3Q8_0FFNLayersDecode(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? GraphNames.DECODE_ACTIVATION : layerGraphName(layerIndex - 1);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.tempQcur,
                    qwen3State.workspace.tempKcur);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
            if (splitsGateUp()) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, qwen3State.workspace.wrapHb2);
            }
            // KV cache already allocated by batch prefill; relay from decode activation graph.
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, keyCache(), valueCache());
            layer.consumeFromDevice(GraphNames.DECODE_ACTIVATION, state.workspace.wrapBlockTable);
        } else {
            String pred = layerGraphName(layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
            layer.consumeFromDevice(qwen3State.workspace.tempQcur, qwen3State.workspace.tempKcur);
            layer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
            if (splitsGateUp()) {
                layer.consumeFromDevice(pred, qwen3State.workspace.wrapHb2);
            }
        }
        return layer;
    }
}
