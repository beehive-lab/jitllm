package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Host weights of a {@code deepseek4} model, every tensor in the representation the file gave it.
 *
 * <p>Besides the layers: the embedding, the mHC head that weighs the streams into one ({@code
 * output_hc_fn}, {@code _base}, {@code _scale}), the final norm and the vocabulary projection.
 */
public final class DeepSeek4StandardWeights implements Weights {

    public final FloatTensor tokenEmbeddingTable;
    public final DeepSeek4LayerWeights<FloatTensor> layers;
    public final FloatTensor hcHeadFn;
    public final FloatTensor hcHeadBase;
    public final FloatTensor hcHeadScale;
    public final FloatTensor outputNorm;
    public final FloatTensor output;

    private final DataType weightType;

    public DeepSeek4StandardWeights(
            FloatTensor tokenEmbeddingTable,
            DeepSeek4LayerWeights<FloatTensor> layers,
            FloatTensor hcHeadFn,
            FloatTensor hcHeadBase,
            FloatTensor hcHeadScale,
            FloatTensor outputNorm,
            FloatTensor output,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.layers = layers;
        this.hcHeadFn = hcHeadFn;
        this.hcHeadBase = hcHeadBase;
        this.hcHeadScale = hcHeadScale;
        this.outputNorm = outputNorm;
        this.output = output;
        this.weightType = weightType;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
