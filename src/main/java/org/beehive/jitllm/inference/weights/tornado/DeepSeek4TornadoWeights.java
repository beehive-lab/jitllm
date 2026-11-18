package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Device weights of a {@code deepseek4} model, every tensor in the representation the file gave it.
 *
 * <p>The base class's slots carry what the shared graphs read — the embedding table, the final norm
 * and the vocabulary projection. Everything else is in {@link #layers} and the mHC head fields.
 * There are no RoPE tables: the kernels compute their rotations.
 */
public final class DeepSeek4TornadoWeights extends TornadoWeights {

    public final DeepSeek4LayerWeights<TornadoTensor> layers;
    public final TornadoTensor hcHeadFn;
    public final TornadoTensor hcHeadBase;
    public final TornadoTensor hcHeadScale;

    public DeepSeek4TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            DeepSeek4LayerWeights<TornadoTensor> layers,
            TornadoTensor hcHeadFn,
            TornadoTensor hcHeadBase,
            TornadoTensor hcHeadScale,
            TornadoTensor outputNorm,
            TornadoTensor output,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                layers.attnNorm(),
                layers.attention().qA(),
                layers.attention().kv(),
                layers.attention().kv(),
                layers.attention().outputB(),
                layers.ffnNorm(),
                layers.ffn().sharedGate(),
                layers.ffn().sharedDown(),
                layers.ffn().sharedUp(),
                outputNorm,
                null,
                null,
                output,
                weightType);
        this.layers = layers;
        this.hcHeadFn = hcHeadFn;
        this.hcHeadBase = hcHeadBase;
        this.hcHeadScale = hcHeadScale;
    }
}
