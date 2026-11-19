package org.beehive.jitllm.model.gptoss;

import org.beehive.jitllm.model.Configuration;

/**
 * The {@code gpt-oss} configuration (gpt-oss-20b, gpt-oss-120b).
 *
 * <p>The head width is stated by the file and is not {@code dim / heads}: 64 query heads of 64 make
 * a 4096-wide query over a 2880-wide residual stream. Every feed-forward is a mixture of experts,
 * so {@link #hiddenDim()} is one expert's width.
 *
 * <p>Layers alternate between a sliding window and full attention: layer {@code l} slides when
 * {@code l % slidingWindowPattern < slidingWindowPattern - 1}, which for the shipped pattern of 2
 * is every even layer.
 *
 * @param slidingWindow the number of most recent positions a sliding layer attends to, the current
 *     one included
 */
public record GptOssConfiguration(
        String quantization,
        int dim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int headSize,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        int numberOfExperts,
        int numberOfExpertsUsed,
        int expertHiddenDim,
        float rmsNormEps,
        float ropeTheta,
        float ropeScalingFactor,
        int ropeOriginalContextLength,
        float yarnBetaFast,
        float yarnBetaSlow,
        int slidingWindow,
        int slidingWindowPattern)
        implements Configuration {

    /** The SwiGLU clamp: the gate is capped above and the linear part clamped on both sides. */
    public static final float SWIGLU_LIMIT = 7.0f;

    /** The SwiGLU sigmoid slope: {@code gate * sigmoid(alpha * gate)}. */
    public static final float SWIGLU_ALPHA = 1.702f;

    /** Whether layer {@code layer} attends over a sliding window rather than the whole context. */
    public boolean isSlidingWindowLayer(int layer) {
        return slidingWindowPattern == 0 || layer % slidingWindowPattern < slidingWindowPattern - 1;
    }

    /** The first position layer {@code layer} attends to from {@code position}. */
    public int windowStart(int layer, int position) {
        return isSlidingWindowLayer(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
    }

    /** One: decode and prefill graphs share one upload of the weights. */
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int hiddenDim() {
        return expertHiddenDim;
    }

    @Override
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    @Override
    public int kvDim() {
        return numberOfKeyValueHeads * headSize;
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /** The attention projections and every expert: what the file holds per layer, times layers. */
    @Override
    public long projectionWeights() {
        long d = dim;
        long q = queryDim();
        long kv = kvDim();
        long experts = (long) numberOfExperts * 3L * d * expertHiddenDim;
        return (d * (q + 2 * kv) + q * d + experts) * numberOfLayers;
    }
}
