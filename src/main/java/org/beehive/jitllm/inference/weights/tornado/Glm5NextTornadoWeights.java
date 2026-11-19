package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.Glm5NextLayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Device weights of a {@code glm5next} model, every tensor in the representation the file gave it.
 *
 * <p>The base class's slots carry what the shared graphs read — the embedding table, the final norm
 * and the vocabulary projection. Everything else is in {@link #layers}. There are no RoPE tables.
 */
public final class Glm5NextTornadoWeights extends TornadoWeights {

    public final Glm5NextLayerWeights<TornadoTensor> layers;

    public Glm5NextTornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            Glm5NextLayerWeights<TornadoTensor> layers,
            TornadoTensor outputNorm,
            TornadoTensor output,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                layers.attnNorm(),
                layers.sparse().qA(),
                layers.kda().k(),
                layers.kda().v(),
                layers.kda().output(),
                layers.ffnNorm(),
                layers.ffn().gate(),
                layers.ffn().down(),
                layers.ffn().up(),
                outputNorm,
                null,
                null,
                output,
                weightType);
        this.layers = layers;
    }
}
