package org.beehive.jitllm.model.glm5next;

import org.beehive.jitllm.model.Configuration;

/**
 * The geometry of a {@code glm5next} model, the text backbone of GLM-5.3-Flash.
 *
 * <p>Every block reads and writes {@link #hcCount()} residual streams through manifold-constrained
 * hyper-connections (mHC): before each sublayer the streams are collapsed into one input, and its
 * output is expanded back and mixed with the streams through a doubly-stochastic matrix. A block's
 * token mixer is either Kimi delta attention (KDA: a delta-rule recurrence with a per-channel
 * decay) or sparse latent attention without RoPE, whose indexer picks pools of {@link
 * #indexerPoolSize()} positions to attend to. Its feed-forward is a dense SwiGLU in the leading
 * blocks and a mixture of experts with one shared expert after them; both clamp their gate and up
 * projections at {@link #swigluLimit()}.
 *
 * <p>Sparse attention runs in the absorbed form: each position stores only its normalized latent of
 * width {@link #kvLoraRank()}, which is both the key and the value of every head.
 *
 * @param numberOfLayers the trunk; the MTP layers the file may append are counted in {@link
 *     #nextnLayers()}
 * @param attentionTypes per trunk layer, {@link #KDA} or {@link #SPARSE_ATTENTION}
 * @param ffnTypes per trunk layer, {@link #DENSE_FFN} or {@link #MOE_FFN}
 * @param keyHeadDim per head, the query and key width of sparse attention
 * @param expertChunkSize how many experts each loaded expert stack holds: the file's expert tensors
 *     are split so that every stack stays under 2³¹ bytes
 */
// @formatter:off
public record Glm5NextConfiguration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int nextnLayers,
        int[] attentionTypes,
        int[] ffnTypes,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        int hcCount,
        int hcSinkhornIterations,
        float hcEps,
        float hcPostScale,
        int kdaHeads,
        int kdaHeadDim,
        int kdaConvKernel,
        float kdaGateLowerBound,
        int numberOfHeads,
        int qLoraRank,
        int kvLoraRank,
        int keyHeadDim,
        int valueHeadDim,
        int indexerHeads,
        int indexerHeadDim,
        int indexerTopK,
        int indexerPoolSize,
        boolean indexerSelectTail,
        int expertCount,
        int expertsUsed,
        int expertHiddenDim,
        int sharedExperts,
        int expertChunkSize,
        float expertWeightsScale,
        boolean expertWeightsNorm,
        float swigluLimit)
        implements Configuration {
    // @formatter:on

    public static final int KDA = 0;
    public static final int SPARSE_ATTENTION = 1;
    public static final int DENSE_FFN = 0;
    public static final int MOE_FFN = 1;

    /** The epsilon of the indexer's key layer norm, fixed by the model. */
    public static final float INDEXER_NORM_EPS = 1e-6f;

    /** The epsilon of the KDA query and key L2 norms, fixed by the model. */
    public static final float KDA_L2_NORM_EPS = 1e-6f;

    @Override
    public String quantization() {
        return quantization;
    }

    /** Whether trunk layer {@code l} mixes tokens with KDA; an MTP layer never does. */
    public boolean isKda(int l) {
        return l < numberOfLayers && attentionTypes[l] == KDA;
    }

    /** Whether layer {@code l} (trunk or MTP) runs a mixture of experts. */
    public boolean isMoe(int l) {
        return l >= numberOfLayers || ffnTypes[l] == MOE_FFN;
    }

    /** Among the trunk's KDA layers, the index of layer {@code l}. */
    public int kdaIndex(int l) {
        int index = 0;
        for (int i = 0; i < l; i++) {
            index += isKda(i) ? 1 : 0;
        }
        return index;
    }

    /** Among the trunk's sparse-attention layers, the index of layer {@code l}. */
    public int sparseIndex(int l) {
        return l - kdaIndex(l);
    }

    public int kdaLayerCount() {
        return kdaIndex(numberOfLayers);
    }

    public int sparseLayerCount() {
        return numberOfLayers - kdaLayerCount();
    }

    /** Pools a cache of {@link #contextLength()} positions can complete, plus the open one. */
    public int maxPools() {
        return contextLength / indexerPoolSize + 1;
    }

    /**
     * Whether the batched prefill's sparse attention runs on FP16 tensor cores: latent and value
     * heads in whole 128-column tiles, query heads in whole 16-element steps.
     */
    public boolean tensorCoreAttention() {
        return kvLoraRank % 128 == 0 && valueHeadDim % 128 == 0 && keyHeadDim % 16 == 0;
    }

    /** Trunk and MTP layers together: the index range of the weights. */
    public int allLayers() {
        return numberOfLayers + nextnLayers;
    }

    /** The width of every KDA projection: query, key, value, decay and output gate. */
    public int kdaDim() {
        return kdaHeads * kdaHeadDim;
    }

    /** {@code (2 + n) * n} mixing coefficients: collapse, expand, then the n-by-n mix. */
    public int hcMixCount() {
        return (2 + hcCount) * hcCount;
    }

    public int queryDim() {
        return numberOfHeads * keyHeadDim;
    }

    public int absorbedQueryDim() {
        return numberOfHeads * kvLoraRank;
    }

    public int attentionOutputInputDim() {
        return numberOfHeads * valueHeadDim;
    }

    /** The trained softmax scale: one over the root of the un-absorbed head width. */
    public float attentionScale() {
        return (float) (1.0 / Math.sqrt(keyHeadDim));
    }

    /** The most pools one query attends to besides its incomplete tail. */
    public int indexerPoolBudget() {
        return indexerTopK / indexerPoolSize;
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
        return kvLoraRank;
    }

    @Override
    public int headSize() {
        return kvLoraRank;
    }

    @Override
    public int kvDim() {
        return kvLoraRank;
    }

    @Override
    public int kvMul() {
        return numberOfHeads;
    }

    /** There is no RoPE anywhere in the text backbone. */
    @Override
    public float ropeTheta() {
        return 0f;
    }

    @Override
    public long recurrentStateBytes() {
        long perLayer =
                (long) kdaHeads * kdaHeadDim * kdaHeadDim + 3L * kdaDim() * (kdaConvKernel - 1);
        long layers = 0;
        for (int l = 0; l < numberOfLayers; l++) {
            if (isKda(l)) {
                layers++;
            }
        }
        return layers * perLayer * Float.BYTES;
    }
}
