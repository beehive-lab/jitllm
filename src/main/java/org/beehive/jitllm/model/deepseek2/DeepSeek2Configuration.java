package org.beehive.jitllm.model.deepseek2;

import org.beehive.jitllm.model.Configuration;

/**
 * The geometry of a {@code deepseek2} model: multi-head latent attention over a compressed
 * key/value stream, and a mixture-of-experts feed-forward after a few leading dense blocks.
 * GLM-4.7-Flash is one such file.
 *
 * <p>Attention runs in the absorbed form llama.cpp uses. Every position stores one latent of width
 * {@link #kvLoraRank()} plus a rotated key of width {@link #ropeDim()}; that pair is the only key
 * and its latent half is the only value, shared by every head. A head's query is its no-rope part
 * pushed through {@code attn_k_b} into the latent space, followed by its rotated part, and a head's
 * output is the attended latent pushed back out through {@code attn_v_b}.
 *
 * @param numberOfLayers the trunk; any MTP block the file appends is not counted
 * @param leadingDenseBlocks the blocks before the first mixture of experts
 * @param qLoraRank the width of the compressed query; zero for files that project queries directly
 * @param nopeDim per head, the query/key width that is not rotated
 * @param ropeDim per head, the width that is rotated; also the shared rotated key's width
 * @param valueHeadDim per head, the width {@code attn_v_b} produces
 * @param sigmoidGating whether router scores are sigmoids (otherwise a softmax over all experts)
 */
// @formatter:off
public record DeepSeek2Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int leadingDenseBlocks,
        int numberOfHeads,
        int qLoraRank,
        int kvLoraRank,
        int nopeDim,
        int ropeDim,
        int valueHeadDim,
        int expertCount,
        int expertsUsed,
        int expertHiddenDim,
        int sharedExperts,
        float expertWeightsScale,
        boolean expertWeightsNorm,
        boolean sigmoidGating,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {
    // @formatter:on

    @Override
    public String quantization() {
        return quantization;
    }

    /** One shared latent head: the absorbed form is multi-query attention. */
    @Override
    public int numberOfKeyValueHeads() {
        return 1;
    }

    /** The width of a stored key: the latent and the rotated key. */
    @Override
    public int numberOfHeadsKey() {
        return keyWidth();
    }

    @Override
    public int headSize() {
        return keyWidth();
    }

    @Override
    public int kvDim() {
        return keyWidth();
    }

    @Override
    public int kvMul() {
        return numberOfHeads;
    }

    public int keyWidth() {
        return kvLoraRank + ropeDim;
    }

    /** A head's query before absorption: {@link #nopeDim()} then {@link #ropeDim()}. */
    public int queryHeadDim() {
        return nopeDim + ropeDim;
    }

    public int queryDim() {
        return numberOfHeads * queryHeadDim();
    }

    /** The absorbed query of every head, each {@link #keyWidth()} wide. */
    public int absorbedQueryDim() {
        return numberOfHeads * keyWidth();
    }

    /** The attended latent of every head. */
    public int latentOutputDim() {
        return numberOfHeads * kvLoraRank;
    }

    /** What {@code attn_output} reads. */
    public int attentionOutputInputDim() {
        return numberOfHeads * valueHeadDim;
    }

    /** {@code attn_kv_a_mqa}'s output: the latent, then the rotated key. */
    public int compressedKeyValueDim() {
        return kvLoraRank + ropeDim;
    }

    /**
     * The score scale. The un-absorbed head is {@code nopeDim + ropeDim} wide and that is what the
     * model was trained to divide by, whatever width the absorbed product is formed over.
     */
    public float attentionScale() {
        return (float) (1.0 / Math.sqrt(queryHeadDim()));
    }

    /** Slices the decode attention cuts each head's window into. */
    public static final int DECODE_ATTENTION_SPLITS = 16;

    public boolean isDenseLayer(int l) {
        return l < leadingDenseBlocks;
    }

    public int sharedHiddenDim() {
        return sharedExperts * expertHiddenDim;
    }

    public int routedHiddenDim() {
        return expertsUsed * expertHiddenDim;
    }

    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }
}
