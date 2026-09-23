package org.beehive.jllm.backend.tornado.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the N+2 prefill/decode forward plan.
 *
 * <pre>
 *   [0]      decodeActivation
 *   [1.N]   layer_0. layer_{N-1}
 *   [N+1]    logits
 * </pre>
 */
public record PrefillDecodeForwardTaskGraphLayout(int N) {
    public int activationIdx() {
        return 0;
    }

    public int layerIdx(int i) {
        return 1 + i;
    }

    public int logitsIdx() {
        return N + 1;
    }

    /**
     * How many distinct <b>layer graph families</b> this topology builds.
     *
     * <p>Not the graph count. A family is a set of per-layer graphs that each bind the layer's
     * weights, and the Tornado runtime allocates a device buffer per graph — so this, not {@link
     * #totalGraphs()}, is the multiplier on per-layer weight memory. This layout has 1.
     *
     * <p>{@code TornadoGraphTopology} asserts that {@code totalGraphs()} equals the sum of {@link
     * #layerFamilyGraphCounts()} plus {@link #nonLayerGraphs()}, with one term per declared family,
     * so adding a family here without stating its graphs fails that check rather than silently
     * under-predicting memory.
     */
    public int layerGraphFamilies() {
        return 1;
    }

    /** Graphs each family contributes, in layout order. One family, one graph per layer. */
    public int[] layerFamilyGraphCounts() {
        return new int[] {N};
    }

    /**
     * Graphs that are not per-layer: activation and logits; prefill reuses the decode layer graphs.
     */
    public int nonLayerGraphs() {
        return 2;
    }

    public int totalGraphs() {
        return N + 2;
    }
}
// @formatter:on
