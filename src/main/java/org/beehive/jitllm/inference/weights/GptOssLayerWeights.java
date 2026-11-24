package org.beehive.jitllm.inference.weights;

/**
 * The per-layer tensors {@code gpt-oss} has beyond a plain decoder: the attention biases, the
 * output bias, one learned attention sink per query head, and the mixture of experts with a bias on
 * the router and on every expert projection.
 *
 * <p>The expert tensors are stacked, one {@code [experts x rows x columns]} tensor per role; the
 * expert biases likewise, {@code [experts x rows]}.
 */
public record GptOssLayerWeights<T>(
        T[] queryBias,
        T[] keyBias,
        T[] valueBias,
        T[] outputBias,
        T[] sinks,
        T[] router,
        T[] routerBias,
        T[] gateExperts,
        T[] gateExpertsBias,
        T[] upExperts,
        T[] upExpertsBias,
        T[] downExperts,
        T[] downExpertsBias) {}
