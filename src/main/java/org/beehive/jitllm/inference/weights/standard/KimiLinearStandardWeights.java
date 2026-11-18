package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.KimiLinearLayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Host weights of a {@code kimi-linear} model, every tensor in the representation the file gave it.
 */
public final class KimiLinearStandardWeights implements Weights {

    public final FloatTensor tokenEmbeddingTable;
    public final KimiLinearLayerWeights<FloatTensor> layers;
    public final FloatTensor outputNorm;
    public final FloatTensor output;

    private final DataType weightType;

    public KimiLinearStandardWeights(
            FloatTensor tokenEmbeddingTable,
            KimiLinearLayerWeights<FloatTensor> layers,
            FloatTensor outputNorm,
            FloatTensor output,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.layers = layers;
        this.outputNorm = outputNorm;
        this.output = output;
        this.weightType = weightType;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
