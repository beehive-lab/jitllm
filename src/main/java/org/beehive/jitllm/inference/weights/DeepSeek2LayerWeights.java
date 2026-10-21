package org.beehive.jitllm.inference.weights;

/**
 * The per-block tensors of a {@code deepseek2} model, for either backend's tensor type.
 *
 * <p>Attention tensors are present at every block. The dense feed-forward ({@code ffnGate}, {@code
 * ffnUp}, {@code ffnDown}) is present only at the leading dense blocks and the mixture of experts
 * only after them; the slots a block does not carry are null.
 */
public record DeepSeek2LayerWeights<T>(
        T[] attnNorm,
        T[] qA,
        T[] qANorm,
        T[] qB,
        T[] kvAMqa,
        T[] kvANorm,
        T[] kB,
        T[] vB,
        T[] wo,
        T[] ffnNorm,
        T[] ffnGate,
        T[] ffnUp,
        T[] ffnDown,
        T[] router,
        T[] routerBias,
        T[] gateExperts,
        T[] upExperts,
        T[] downExperts,
        T[] sharedGate,
        T[] sharedUp,
        T[] sharedDown) {}
