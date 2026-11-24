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
package org.beehive.jitllm.backend.tornado.layers.phi3;

import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections.LayerWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.Phi3TornadoWeights;

/**
 * Phi-3's projections in the grouping its MLX prefill multiplies them in: the fused QKV and gate/up
 * tensors as they are. One definition for the prefill layers that build the GEMMs and the decode
 * layers that learn from it which weights the prefill graphs bind.
 */
public final class Phi3MlxLayerWeights {

    private Phi3MlxLayerWeights() {}

    public static LayerWeights of(Phi3TornadoWeights weights, int layer) {
        return new LayerWeights(
                new TornadoTensor[] {weights.wqkvLayered[layer]},
                weights.woLayered[layer],
                new TornadoTensor[] {weights.wUpLayered[layer]},
                weights.wDownLayered[layer]);
    }
}
