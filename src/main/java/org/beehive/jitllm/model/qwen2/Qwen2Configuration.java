package org.beehive.jllm.model.qwen2;

import org.beehive.jllm.model.Configuration;

public record Qwen2Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int numberOfHeadsKey,
        int numberOfHeadsValue,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        boolean sharedWeights,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {
    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }
}
