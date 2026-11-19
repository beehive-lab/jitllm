package org.beehive.jitllm.inference.weights;

/**
 * Per-layer tensors of a {@code kimi-linear} model, indexed by layer, for either backend's tensor
 * type. A slot a layer does not carry is {@code null}: the KDA tensors on MLA layers, the MLA
 * tensors on KDA layers, the dense feed-forward after the leading blocks and the experts in them.
 */
// @formatter:off
public record KimiLinearLayerWeights<T>(
        T[] attnNorm, T[] ffnNorm, KdaWeights<T> kda, Mla<T> mla, FeedForward<T> ffn) {

    /**
     * Latent attention with its {@code kv_b} split llama.cpp's way: {@code attn_k_b} takes a head's
     * no-rope query into the latent space, {@code attn_v_b} takes the attended latent out. The
     * query comes straight from {@code attn_q}.
     */
    public record Mla<T>(T[] q, T[] kvAMqa, T[] kvANorm, T[] kB, T[] vB, T[] output) {}

    /** The dense SwiGLU, or the router, the routed experts (one tensor each) and shared experts. */
    public record FeedForward<T>(
            T[] gate,
            T[] up,
            T[] down,
            T[] router,
            T[] routerBias,
            T[] gateExperts,
            T[] upExperts,
            T[] downExperts,
            T[] sharedGate,
            T[] sharedUp,
            T[] sharedDown) {}
}
// @formatter:on
