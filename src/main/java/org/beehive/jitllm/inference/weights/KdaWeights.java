package org.beehive.jitllm.inference.weights;

/**
 * Per-layer tensors of Kimi delta attention (KDA), indexed by layer; null on layers that mix tokens
 * otherwise. Shared by every family with KDA layers (GLM-5.3-Flash, Kimi-Linear).
 *
 * <p>{@code a} holds {@code -exp(A_log)} per head, as llama.cpp's converter writes it.
 */
// @formatter:off
public record KdaWeights<T>(
        T[] q,
        T[] k,
        T[] v,
        T[] convQ,
        T[] convK,
        T[] convV,
        T[] decayA,
        T[] decayB,
        T[] dtBias,
        T[] a,
        T[] beta,
        T[] gateA,
        T[] gateB,
        T[] outNorm,
        T[] output) {}
// @formatter:on
