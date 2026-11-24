package org.beehive.jitllm.model.kimilinear;

import org.beehive.jitllm.model.Configuration;

/**
 * The geometry of a {@code kimi-linear} model (Kimi-Linear-48B-A3B).
 *
 * <p>A plain pre-norm residual trunk. Each block's token mixer is either Kimi delta attention (KDA:
 * a delta-rule recurrence with a per-channel decay) or multi-head latent attention (MLA) without
 * RoPE; its feed-forward is a dense SwiGLU in the leading blocks and a mixture of experts with
 * shared experts after them.
 *
 * <p>MLA runs in the absorbed form llama.cpp uses. Every position stores one latent of width {@link
 * #kvLoraRank()} and a shared key tail of width {@link #ropeDim()}; the model applies no rotation
 * to that tail, so the device and host paths rotate it by an identity table. A head's query is
 * projected directly from the input (there is no compressed query).
 *
 * @param attentionTypes per layer, {@link #KDA} or {@link #MLA}
 * @param leadingDenseBlocks the blocks before the first mixture of experts
 * @param kdaHeads KDA heads; their query, key and value width is {@link #kdaHeadDim()}
 * @param nopeDim per MLA head, the query/key width that goes through {@code attn_k_b}
 * @param ropeDim per MLA head, the query/key tail compared directly with the shared key tail
 */
// @formatter:off
public record KimiLinearConfiguration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int[] attentionTypes,
        int leadingDenseBlocks,
        int kdaHeads,
        int kdaHeadDim,
        int kdaConvKernel,
        int numberOfHeads,
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
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps)
        implements Configuration {
    // @formatter:on

    public static final int KDA = 0;
    public static final int MLA = 1;

    /** Slices one head's positions are split over in decode attention. */
    public static final int DECODE_ATTENTION_SPLITS = 16;

    @Override
    public String quantization() {
        return quantization;
    }

    public boolean isKda(int l) {
        return attentionTypes[l] == KDA;
    }

    public boolean isDenseLayer(int l) {
        return l < leadingDenseBlocks;
    }

    /** Among the KDA layers, the index of layer {@code l}. */
    public int kdaIndex(int l) {
        int index = 0;
        for (int i = 0; i < l; i++) {
            index += isKda(i) ? 1 : 0;
        }
        return index;
    }

    /** Among the MLA layers, the index of layer {@code l}. */
    public int mlaIndex(int l) {
        return l - kdaIndex(l);
    }

    public int kdaLayerCount() {
        return kdaIndex(numberOfLayers);
    }

    public int mlaLayerCount() {
        return numberOfLayers - kdaLayerCount();
    }

    /** The width of every KDA projection: query, key, value, decay and output gate. */
    public int kdaDim() {
        return kdaHeads * kdaHeadDim;
    }

    public int queryHeadDim() {
        return nopeDim + ropeDim;
    }

    public int queryDim() {
        return numberOfHeads * queryHeadDim();
    }

    /** A stored MLA key: the normalized latent, then the shared tail. */
    public int keyWidth() {
        return kvLoraRank + ropeDim;
    }

    /** What {@code attn_kv_a_mqa} produces: the latent and the tail. */
    public int compressedKeyValueDim() {
        return kvLoraRank + ropeDim;
    }

    public int absorbedQueryDim() {
        return numberOfHeads * keyWidth();
    }

    public int attentionOutputInputDim() {
        return numberOfHeads * valueHeadDim;
    }

    /** The trained softmax scale: one over the root of the un-absorbed query head width. */
    public float attentionScale() {
        return (float) (1.0 / Math.sqrt(queryHeadDim()));
    }

    public int sharedHiddenDim() {
        return sharedExperts * expertHiddenDim;
    }

    @Override
    public int numberOfKeyValueHeads() {
        return 1;
    }

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

    /** No RoPE anywhere: the MLA key tail is compared unrotated. */
    @Override
    public float ropeTheta() {
        return 0f;
    }

    @Override
    public long recurrentStateBytes() {
        long perLayer =
                (long) kdaHeads * kdaHeadDim * kdaHeadDim + 3L * kdaDim() * (kdaConvKernel - 1);
        return kdaLayerCount() * perLayer * Float.BYTES;
    }
}
