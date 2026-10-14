package org.beehive.jitllm.model.lfm2;

import org.beehive.jitllm.model.Configuration;

/**
 * Configuration for the {@code lfm2moe} architecture — Liquid AI's LFM2 / LFM2.5 mixture of
 * experts, verified here against {@code LFM2.5-24B-A2B-UD-Q4_K_XL.gguf}.
 *
 * <h2>Two kinds of mixer, two kinds of feed-forward</h2>
 *
 * <p>A layer mixes either with grouped-query attention or with a gated short convolution. The file
 * says which per layer, through {@code attention.head_count_kv}, an array whose entry is zero at
 * every convolution layer — ten of the 24B's forty layers attend. {@link #isAttentionLayer(int)} is
 * the single place that decision is made.
 *
 * <p>Independently, the first {@link #leadingDenseBlockCount()} layers feed forward through a dense
 * SwiGLU of {@link #hiddenDim()}, and every later layer through a routed mixture of {@link
 * #expertCount()} experts of {@link #expertHiddenDim()}, {@link #expertUsedCount()} chosen per
 * token. There is no shared expert.
 *
 * <h2>Dimensions that are derived</h2>
 *
 * <p>The head width is not stated by the file: llama.cpp takes it as {@code dim / heads}, 64 here,
 * and so does {@link #headSize()}. The per-head query and key norms are that wide, which the loader
 * checks.
 */
// @formatter:off
public record Lfm2Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int[] headCountKv,
        int shortConvCache,
        int expertCount,
        int expertUsedCount,
        int expertHiddenDim,
        int leadingDenseBlockCount,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    /** Key/value heads of the attention layers, which all share one count. */
    @Override
    public int numberOfKeyValueHeads() {
        for (int heads : headCountKv) {
            if (heads > 0) {
                return heads;
            }
        }
        return 0;
    }

    /** The attention head width: {@code dim / heads}, as llama.cpp derives it for this family. */
    @Override
    public int numberOfHeadsKey() {
        return dim / numberOfHeads;
    }

    @Override
    public int headSize() {
        return numberOfHeadsKey();
    }

    @Override
    public int kvDim() {
        return numberOfKeyValueHeads() * headSize();
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads();
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }

    // ---- layer kinds -------------------------------------------------------

    /** Whether layer {@code l} mixes with attention rather than with the short convolution. */
    public boolean isAttentionLayer(int l) {
        return headCountKv[l] > 0;
    }

    /** Whether layer {@code l} feeds forward through the routed experts rather than densely. */
    public boolean isMoeLayer(int l) {
        return l >= leadingDenseBlockCount;
    }

    /** How many layers attend, and so hold a key/value cache. */
    public int numberOfAttentionLayers() {
        int count = 0;
        for (int l = 0; l < numberOfLayers; l++) {
            if (isAttentionLayer(l)) {
                count++;
            }
        }
        return count;
    }

    @Override
    public int keyValueLayerCount() {
        return numberOfAttentionLayers();
    }

    /**
     * Where layer {@code l} addresses its key/value entries, or -1 if it holds none.
     *
     * <p>A dense index over the layers that attend: one in four does, so a store indexed by layer
     * would be four times larger than the model uses.
     */
    public int keyValueLayerIndex(int l) {
        if (!isAttentionLayer(l)) {
            return -1;
        }
        int index = 0;
        for (int layer = 0; layer < l; layer++) {
            if (isAttentionLayer(layer)) {
                index++;
            }
        }
        return index;
    }

    /** Where layer {@code l} addresses its convolution window, or -1 if it holds none. */
    public int convLayerIndex(int l) {
        if (isAttentionLayer(l)) {
            return -1;
        }
        int index = 0;
        for (int layer = 0; layer < l; layer++) {
            if (!isAttentionLayer(layer)) {
                index++;
            }
        }
        return index;
    }

    /** How many layers hold a convolution window. */
    public int convLayerCount() {
        return numberOfLayers - numberOfAttentionLayers();
    }

    // ---- short convolution -------------------------------------------------

    /**
     * Width of {@code shortconv.in_proj}'s output: the {@code B}, {@code C} and {@code x} chunks,
     * each {@code dim} wide, in that order.
     */
    public int shortConvProjectionDim() {
        return 3 * dim;
    }

    /** Elements of convolution history one layer holds: the kernel minus this step, per channel. */
    public int convStateSize() {
        return (shortConvCache - 1) * dim;
    }

    /** The convolution windows, one per convolution layer. */
    @Override
    public long recurrentStateBytes() {
        return (long) convLayerCount() * convStateSize() * Float.BYTES;
    }
}
