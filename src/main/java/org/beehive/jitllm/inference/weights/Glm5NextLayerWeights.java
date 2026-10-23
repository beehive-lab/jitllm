package org.beehive.jitllm.inference.weights;

import java.util.List;

/**
 * Per-layer tensors of a {@code glm5next} model, indexed by layer: the trunk first, then any MTP
 * layer. A slot a layer does not carry is {@code null} (or an empty list for the experts).
 *
 * <p>The routed experts of one projection are stored as a few stacked tensors of {@code
 * expertChunkSize} experts each, so no single tensor exceeds what one host tensor can address.
 */
// @formatter:off
public record Glm5NextLayerWeights<T>(
        Mhc<T> mhc,
        T[] attnNorm,
        T[] ffnNorm,
        Kda<T> kda,
        SparseAttention<T> sparse,
        FeedForward<T> ffn,
        Nextn<T> nextn) {

    /** The collapse/expand/mix projections around the attention and feed-forward sublayers. */
    public record Mhc<T>(
            T[] attnFn, T[] attnBase, T[] attnScale, T[] ffnFn, T[] ffnBase, T[] ffnScale) {}

    /** Kimi delta attention: projections, short convolutions, decay, beta and the output gate. */
    public record Kda<T>(
            T[] q,
            T[] k,
            T[] v,
            T[] convQ,
            T[] convK,
            T[] convV,
            T[] decayA,
            T[] decayB,
            T[] dtBias,
            T[] aLog,
            T[] beta,
            T[] gateA,
            T[] gateB,
            T[] outNorm,
            T[] output) {}

    /** Latent attention without RoPE, its {@code kv_b} split llama.cpp's way, and the indexer. */
    public record SparseAttention<T>(
            T[] qA,
            T[] qANorm,
            T[] qB,
            T[] kvA,
            T[] kvANorm,
            T[] kB,
            T[] vB,
            T[] output,
            T[] indexQ,
            T[] indexK,
            T[] indexKNorm,
            T[] indexKNormBias,
            T[] indexWeights,
            T[] poolApe,
            T[] poolGate) {}

    /** The dense SwiGLU, or the router, routed experts and shared expert. */
    public record FeedForward<T>(
            T[] gate,
            T[] up,
            T[] down,
            T[] router,
            T[] routerBias,
            List<List<T>> gateExperts,
            List<List<T>> upExperts,
            List<List<T>> downExperts,
            T[] sharedGate,
            T[] sharedUp,
            T[] sharedDown) {}

    /** The MTP layer's input projection and norms; null slots on trunk layers. */
    public record Nextn<T>(T[] embedNorm, T[] hiddenNorm, T[] inputProjection, T[] headNorm) {}
}
// @formatter:on
