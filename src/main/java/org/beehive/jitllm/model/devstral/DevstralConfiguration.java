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

    /**
     * One: batched prefill lays out a prefill and a decode family of per-layer graphs, but the
     * decode graphs consume the weights the prefill graphs uploaded (their {@code
     * weightSourceGraphName}), so the plan holds the weights once, as Llama's does.
     */
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

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

    /** Its layer classes bind the paged kernels and chain their buffers by predecessor name. */
    @Override
    public boolean supportsSharedKvStorage() {
        return true;
    }
}
