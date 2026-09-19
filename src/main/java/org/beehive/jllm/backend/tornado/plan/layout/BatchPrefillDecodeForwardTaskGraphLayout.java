package org.beehive.jllm.backend.tornado.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the batch-prefill/decode forward plan.
 *
 * <pre>
 *   [0]                         batchPrefillActivation
 *   [1..N]                      batchPrefillLayer_0 .. batchPrefillLayer_{N-1}
 *   [N+1]                       decodeActivation    (consumes + re-persists KV cache)
 *   [N+2..N+1+D]                decode layer graphs, D of them
 *   [N+2+D]                     logits
 * </pre>
 *
 * <p><b>{@code D} is not always {@code N}.</b> A family may put more than one transformer layer in
 * a decode graph, which costs one graph submission instead of several. The logical layer count and
 * the decode <i>graph</i> count are therefore separate quantities, and only the second one moves
 * the indices below. Nothing here says how layers are distributed among those graphs; that is the
 * layer builder's business.
 */
// @formatter:on
public record BatchPrefillDecodeForwardTaskGraphLayout(
        int N, int batchLayerGraphs, int fallbackLayerGraphs, int decodeLayerGraphs) {

    /** The ungrouped layout: one decode graph per layer, which is what every family built. */
    public BatchPrefillDecodeForwardTaskGraphLayout(int N) {
        this(N, N, 0, N);
    }

    /** The batch-prefill side ungrouped, the decode side as given. */
    public BatchPrefillDecodeForwardTaskGraphLayout(int N, int decodeLayerGraphs) {
        this(N, N, 0, decodeLayerGraphs);
    }

    public BatchPrefillDecodeForwardTaskGraphLayout {
        if (batchLayerGraphs < 1 || batchLayerGraphs > N) {
            throw new IllegalArgumentException(
                    "batch-prefill layer graphs must be between 1 and "
                            + N
                            + ", got "
                            + batchLayerGraphs);
        }
        if (fallbackLayerGraphs < 0 || fallbackLayerGraphs > N) {
            throw new IllegalArgumentException(
                    "fallback layer graphs must be between 0 and "
                            + N
                            + ", got "
                            + fallbackLayerGraphs);
        }
        if (decodeLayerGraphs < 1 || decodeLayerGraphs > N) {
            throw new IllegalArgumentException(
                    "decode layer graphs must be between 1 and "
                            + N
                            + ", got "
                            + decodeLayerGraphs);
        }
    }

    public int batchActivationIdx() {
        return 0;
    }

    public int batchLayerIdx(int i) {
        return 1 + i;
    }

    /** The index of the {@code g}-th FALLBACK batch-prefill layer graph. */
    public int fallbackLayerIdx(int g) {
        return 1 + batchLayerGraphs + g;
    }

    public int decodeActivationIdx() {
        return batchLayerGraphs + fallbackLayerGraphs + 1;
    }

    /**
     * The index of the {@code g}-th decode layer <b>graph</b>.
     *
     * <p>This is what the forward loop iterates. It is indexed by graph, not by layer, because a
     * graph may hold more than one layer.
     */
    public int decodeLayerGraphIdx(int g) {
        return batchLayerGraphs + fallbackLayerGraphs + 2 + g;
    }

    public int logitsIdx() {
        return batchLayerGraphs + fallbackLayerGraphs + 2 + decodeLayerGraphs;
    }

    // @formatter:off
    /**
     * How many distinct <b>layer graph families</b> this topology builds.
     *
     * <p>A family is a set of graphs that bind the layers' weights. This layout has 2: the
     * batch-prefill layers and the decode layers. It is <b>not</b> a graph count and <b>not</b> a
     * weight multiplier — a family that consumes another's upload costs graphs, not gigabytes,
     * which is what {@code Configuration.weightBindingFamilies} decides.
     */
    // @formatter:on
    public int layerGraphFamilies() {
        return fallbackLayerGraphs > 0 ? 3 : 2;
    }

    /**
     * Graphs each family contributes, in layout order.
     *
     * <p>{@code TornadoGraphTopology} checks this against both the family count and the total, so a
     * family added without a term here fails rather than under-predicting silently.
     */
    public int[] layerFamilyGraphCounts() {
        return fallbackLayerGraphs > 0
                ? new int[] {batchLayerGraphs, fallbackLayerGraphs, decodeLayerGraphs}
                : new int[] {batchLayerGraphs, decodeLayerGraphs};
    }

    /** Graphs that are not per-layer: batch activation, decode activation and logits. */
    public int nonLayerGraphs() {
        return 3;
    }

    public int totalGraphs() {
        return batchLayerGraphs + fallbackLayerGraphs + decodeLayerGraphs + nonLayerGraphs();
    }
}
