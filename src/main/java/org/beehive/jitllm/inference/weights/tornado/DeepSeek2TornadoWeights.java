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
package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Device weights of a {@code deepseek2} model, every tensor in the representation the file gave it.
 *
 * <p>The base class's slots carry what the shared graphs read — the embedding table, the final norm
 * and the vocabulary projection — and the leading dense blocks' feed-forward. Everything this
 * family has beyond them is in {@link #layers}.
 */
public final class DeepSeek2TornadoWeights extends TornadoWeights {

    public final DeepSeek2LayerWeights<TornadoTensor> layers;

    public DeepSeek2TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            DeepSeek2LayerWeights<TornadoTensor> layers,
            TornadoTensor outputNorm,
            TornadoTensor output,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                layers.attnNorm(),
                layers.qA(),
                layers.kvAMqa(),
                layers.kvAMqa(),
                layers.wo(),
                layers.ffnNorm(),
                layers.ffnGate(),
                layers.ffnDown(),
                layers.ffnUp(),
                outputNorm,
                freqCisReal,
                freqCisImag,
                output,
                weightType);
        this.layers = layers;
    }
}
