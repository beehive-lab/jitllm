package org.beehive.jitllm.model.qwen2;

import org.beehive.jitllm.model.Configuration;

public record Qwen2MoEConfiguration(
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
        int numberOfExperts,
        int numberOfExpertsUsed,
        int moeHiddenDim,
        int sharedExpertHiddenDim,
        boolean sharedWeights,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    /**
     * One: the decode graphs of a batched plan consume the weights the prefill graphs uploaded, so
     * the plan holds them once. Measured: Qwen1.5-MoE-A2.7B Q8_0 (14.2 GiB file) peaks at 14.5 GiB
     * of GPU memory batched on CUDA, where counting the weights twice predicted 33.8 GiB.
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
