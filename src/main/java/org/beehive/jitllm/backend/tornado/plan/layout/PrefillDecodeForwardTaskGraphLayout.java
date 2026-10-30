package org.beehive.jitllm.backend.tornado.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the prefill/decode forward plan: P prepacking graphs, initialization,
 * then the N+2 inference graphs.
 *
 * <pre>
 *   [0 .. P-1]      prepacking graphs (P = prepackingGraphs, 0 if none)
 *   [P]             decodeActivation
 *   [P+1 .. P+N]    layer_0 .. layer_{N-1}
 *   [P+N+1]         logits
 * </pre>
 *
 * <p>Every index is computed here from P, so no caller adds an offset of its own.
 */
public record PrefillDecodeForwardTaskGraphLayout(int N, int prepackingGraphs) {

    public PrefillDecodeForwardTaskGraphLayout {
        if (prepackingGraphs < 0) {
            throw new IllegalArgumentException(
                    "prepacking graphs must not be negative, got " + prepackingGraphs);
        }
    }

    /** A layout with no prepacking graphs. */
    public PrefillDecodeForwardTaskGraphLayout(int N) {
        this(N, 0);
    }

    /**
     * The index of the {@code i}-th prepacking graph. They come first and run once, before the
     * warm-up; no inference step selects them.
     */
    public int prepackingIdx(int i) {
        return i;
    }

    /** The index of the first inference graph, right after the prepacking graphs. */
    public int firstInferenceIdx() {
        return prepackingGraphs;
    }

    public int activationIdx() {
        return firstInferenceIdx();
    }

    public int layerIdx(int i) {
        return firstInferenceIdx() + 1 + i;
    }

    public int logitsIdx() {
        return firstInferenceIdx() + N + 1;
    }

    /**
     * How many distinct <b>layer graph families</b> this topology builds.
     *
     * <p>Not the graph count. A family is a set of per-layer graphs that each bind the layer's
     * weights, and the Tornado runtime allocates a device buffer per graph — so this, not {@link
     * #inferenceGraphCount()}, is the multiplier on per-layer weight memory. This layout has 1.
     *
     * <p>{@code TornadoGraphTopology} asserts that {@code inferenceGraphCount()} equals the sum of
     * {@link #layerFamilyGraphCounts()} plus {@link #nonLayerGraphs()}, with one term per declared
     * family, so adding a family here without stating its graphs fails that check rather than
     * silently under-predicting memory.
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

    /** The inference graphs: activation, layers and logits, without the prepacking graphs. */
    public int inferenceGraphCount() {
        return N + 2;
    }

    /** Every graph of the plan: the prepacking graphs and the inference graphs. */
    public int totalGraphs() {
        return prepackingGraphs + inferenceGraphCount();
    }
}
// @formatter:on
