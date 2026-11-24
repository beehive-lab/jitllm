package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Device weights of a {@code kimi-linear} model, every tensor in the representation the file gave
 * it.
 *
 * <p>The base class's slots carry what the shared graphs read: the embedding table, the final norm
 * and the vocabulary projection. Everything else is in {@link #layers}. The RoPE slots hold the
 * identity tables the MLA key tail is "rotated" by.
 */
public final class KimiLinearTornadoWeights extends TornadoWeights {

    public final KimiLinearLayerWeights<TornadoTensor> layers;

    public KimiLinearTornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            KimiLinearLayerWeights<TornadoTensor> layers,
            TornadoTensor outputNorm,
            TornadoTensor output,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                layers.attnNorm(),
                layers.mla().q(),
                layers.kda().k(),
                layers.kda().v(),
                layers.kda().output(),
                layers.ffnNorm(),
                layers.ffn().gate(),
                layers.ffn().down(),
                layers.ffn().up(),
                outputNorm,
                freqCisReal,
                freqCisImag,
                output,
                weightType);
        this.layers = layers;
    }
}
