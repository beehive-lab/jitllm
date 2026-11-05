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

    /**
     * One: the decode graphs of a batched plan consume the weights the prefill graphs uploaded, so
     * the plan holds them once. Measured: Phi-3-mini F16 (7.1 GiB file) peaks at 7.9 GiB of GPU
     * memory batched on CUDA.
     */
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

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
