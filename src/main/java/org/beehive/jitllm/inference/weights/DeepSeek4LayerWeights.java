package org.beehive.jitllm.inference.weights;

import java.util.List;

/**
 * Per-layer tensors of a {@code deepseek4} model, indexed by layer: the trunk first, then any MTP
 * layer. A slot a layer does not carry is {@code null} (or an empty list for the experts).
 *
 * <p>The routed experts of one projection are stored as a few stacked tensors of {@code
 * expertChunkSize} experts each, so no single tensor exceeds what one tensor can address.
 */
// @formatter:off
public record DeepSeek4LayerWeights<T>(
        Mhc<T> mhc,
        T[] attnNorm,
        T[] ffnNorm,
        Attention<T> attention,
        Compressor<T> compressor,
        Indexer<T> indexer,
        FeedForward<T> ffn,
        Nextn<T> nextn) {

    /** The collapse/expand/mix projections around the attention and feed-forward sublayers. */
    public record Mhc<T>(
            T[] attnFn, T[] attnBase, T[] attnScale, T[] ffnFn, T[] ffnBase, T[] ffnScale) {}

    /**
     * The query path ({@code q_a}, its norm, {@code q_b}), the single key/value projection and its
     * norm, the grouped output ({@code output_a} per group, then {@code output_b}) and the sinks.
     */
    public record Attention<T>(
            T[] qA, T[] qANorm, T[] qB, T[] kv, T[] kvNorm, T[] outputA, T[] outputB, T[] sinks) {}

    /** The key/value compressor of a layer with a compression ratio. */
    public record Compressor<T>(T[] kv, T[] gate, T[] ape, T[] norm) {}

    /** The indexer of a ratio-4 layer: its query projection, head weights and own compressor. */
    public record Indexer<T>(
            T[] qB,
            T[] headWeights,
            T[] compressorKv,
            T[] compressorGate,
            T[] compressorApe,
            T[] compressorNorm) {}

    /**
     * The router, its selection bias (score-routed layers) or token-to-expert table (hash-routed
     * layers, {@code [vocabulary][expertsUsed]}), the routed experts and the shared expert.
     */
    public record FeedForward<T>(
            T[] router,
            T[] routerBias,
            int[][] tokenExperts,
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
