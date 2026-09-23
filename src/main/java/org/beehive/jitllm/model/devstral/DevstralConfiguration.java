package org.beehive.jitllm.model.devstral;

import org.beehive.jitllm.model.Configuration;

/**
 * Configuration for Devstral 2 models (Mistral 3 architecture). Unlike standard Mistral, Devstral 2
 * has an independent head dimension (head_dim != dim / num_heads), requiring explicit
 * key_length/value_length.
 */
// @formatter:off
public record DevstralConfiguration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int headDim,
        int vocabularySize,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    /**
     * Q projection output dimension = numberOfHeads * headDim. This differs from dim when headDim
     * != dim/numberOfHeads.
     */
    public int qDim() {
        return numberOfHeads * headDim;
    }

    public int kvDim() {
        return numberOfKeyValueHeads * headDim;
    }

    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsKey() {
        throw new UnsupportedOperationException("Not supported for Devstral.");
    }

    @Override
    public int contextLengthModel() {
        throw new UnsupportedOperationException("Not supported for Devstral.");
    }

    public int headSize() {
        return headDim;
    }
}
