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
