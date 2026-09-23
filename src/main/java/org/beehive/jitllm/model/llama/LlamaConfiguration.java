package org.beehive.jitllm.model.llama;

import org.beehive.jitllm.model.Configuration;

// @formatter:off
public record LlamaConfiguration(
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
     * One: batched prefill lays out a prefill and a decode family of per-layer graphs, but the
     * decode graphs consume the weights the prefill graphs uploaded (their {@code
     * weightSourceGraphName}), so the plan holds the weights once. Measured: Llama-3.2-1B F16
     * batched prefill runs in a 2600 MB budget and Qwen3-8B F16 in 17 GB, where counting the
     * weights twice predicted 4.2 GB and 29.2 GB and refused the second. The native prefill path's
     * stacked projection copies are a separate component, not this count.
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
        throw new UnsupportedOperationException("Not supported for Llama.");
    }

    @Override
    public int contextLengthModel() {
        throw new UnsupportedOperationException("Not supported for Llama.");
    }

    /** Size of each attention head (derived from dim / numberOfHeads) */
    public int headSize() {
        return dim / numberOfHeads;
    }

    /** Key/value dimension (derived from dim * numberOfKeyValueHeads / numberOfHeads) */
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** Multiplier for key/value sharing in multi-query attention */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length, or the current instance if
     *     newContextLength is negative
     */
    // @formatter:off
    public LlamaConfiguration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new LlamaConfiguration(
                this.quantization,
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta);
    }
    // @formatter:on
}
