package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Host weights of a {@code deepseek2} model, every tensor in the representation the file gave it.
 */
public final class DeepSeek2StandardWeights implements Weights {

    public final FloatTensor tokenEmbeddingTable;
    public final DeepSeek2LayerWeights<FloatTensor> layers;
    public final FloatTensor outputNorm;
    public final FloatTensor output;

    /** RoPE tables over the rotated width, indexed {@code position * ropeDim / 2 + pair}. */
    public final FloatTensor freqCisReal;

    public final FloatTensor freqCisImag;

    private final DataType weightType;

    public DeepSeek2StandardWeights(
            FloatTensor tokenEmbeddingTable,
            DeepSeek2LayerWeights<FloatTensor> layers,
            FloatTensor outputNorm,
            FloatTensor output,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.layers = layers;
        this.outputNorm = outputNorm;
        this.output = output;
        this.freqCisReal = freqCisReal;
        this.freqCisImag = freqCisImag;
        this.weightType = weightType;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
