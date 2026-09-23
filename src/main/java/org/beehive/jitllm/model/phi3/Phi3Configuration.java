package org.beehive.jitllm.model.phi3;

import org.beehive.jitllm.model.Configuration;

// @formatter:off
public record Phi3Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int vocabularySize,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int numberOfHeadsKey() {
        // For Phi3, key heads are the same as key-value heads
        return numberOfKeyValueHeads;
    }

    @Override
    public int headSize() {
        // Calculate head size from dim and numberOfHeads
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        // Calculate key-value dimension
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    @Override
    public int kvMul() {
        // Calculate key-value multiplier for multi-query attention
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }
}
