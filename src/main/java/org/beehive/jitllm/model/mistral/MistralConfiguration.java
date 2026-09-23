package org.beehive.jitllm.model.mistral;

import org.beehive.jitllm.model.Configuration;

// @formatter:off
public record MistralConfiguration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int vocabularySize,
        int contextLength,
        boolean sharedWeights,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsKey() {
        throw new UnsupportedOperationException("Not supported for Mistral.");
    }

    @Override
    public int contextLengthModel() {
        throw new UnsupportedOperationException("Not supported for Mistral.");
    }

    public int headSize() {
        return dim / numberOfHeads;
    }
}
