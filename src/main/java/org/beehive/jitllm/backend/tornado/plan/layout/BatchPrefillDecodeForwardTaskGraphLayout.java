package org.beehive.jitllm.backend.tornado.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the batch-prefill/decode forward plan: P prepacking graphs,
 * initialization, then the inference graphs.
 *
 * <pre>
 *   [0 .. P-1]                  prepacking graphs (P = prepackingGraphs, 0 if none)
 *   [P]                         batchPrefillActivation
 *   [P+1 .. P+N]                batchPrefillLayer_0 .. batchPrefillLayer_{N-1}
 *   [P+N+1]                     decodeActivation    (consumes + re-persists KV cache)
 *   [P+N+2 .. P+N+1+D]          decode layer graphs, D of them
 *   [P+N+2+D]                   logits
 * </pre>
 *
 * <p>With a fallback family its F graphs follow the batch-prefill layers and move every later index
 * by F; a narrow family's R graphs follow those and move every later index by R. Every index is
 * computed here from P, so no caller adds an offset of its own.
 *
 * <p><b>{@code D} is not always {@code N}.</b> A family may put more than one transformer layer in
 * a decode graph, which costs one graph submission instead of several. The logical layer count and
 * the decode <i>graph</i> count are therefore separate quantities, and only the second one moves
 * the indices below. Nothing here says how layers are distributed among those graphs; that is the
 * layer builder's business.
 */
// @formatter:on
public record BatchPrefillDecodeForwardTaskGraphLayout(
        int N,
        int batchLayerGraphs,
        int fallbackLayerGraphs,
        int decodeLayerGraphs,
        int prepackingGraphs,
        int narrowLayerGraphs) {

    /** The layout as given, with no narrow family. */
    public BatchPrefillDecodeForwardTaskGraphLayout(
            int N,
            int batchLayerGraphs,
            int fallbackLayerGraphs,
            int decodeLayerGraphs,
            int prepackingGraphs) {
        this(N, batchLayerGraphs, fallbackLayerGraphs, decodeLayerGraphs, prepackingGraphs, 0);
    }

    /** The layout as given, with no prepacking graphs. */
    public BatchPrefillDecodeForwardTaskGraphLayout(
            int N, int batchLayerGraphs, int fallbackLayerGraphs, int decodeLayerGraphs) {
        this(N, batchLayerGraphs, fallbackLayerGraphs, decodeLayerGraphs, 0, 0);
    }

    /** The fully ungrouped layout, no fallback family: one graph per layer on both sides. */
    public BatchPrefillDecodeForwardTaskGraphLayout(int N) {
        this(N, N, 0, N, 0, 0);
    }

    /** The batch-prefill side ungrouped, the decode side as given. */
    public BatchPrefillDecodeForwardTaskGraphLayout(int N, int decodeLayerGraphs) {
        this(N, N, 0, decodeLayerGraphs, 0, 0);
    }

    public BatchPrefillDecodeForwardTaskGraphLayout {
        if (prepackingGraphs < 0) {
            throw new IllegalArgumentException(
                    "prepacking graphs must not be negative, got " + prepackingGraphs);
        }
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
        if (narrowLayerGraphs < 0 || narrowLayerGraphs > N) {
            throw new IllegalArgumentException(
                    "narrow layer graphs must be between 0 and "
                            + N
                            + ", got "
                            + narrowLayerGraphs);
        }
        if (decodeLayerGraphs < 1 || decodeLayerGraphs > N) {
            throw new IllegalArgumentException(
                    "decode layer graphs must be between 1 and "
                            + N
                            + ", got "
                            + decodeLayerGraphs);
        }
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

    public int batchActivationIdx() {
        return firstInferenceIdx();
    }

    public int batchLayerIdx(int i) {
        return firstInferenceIdx() + 1 + i;
    }

    /** The index of the {@code g}-th FALLBACK batch-prefill layer graph. */
    public int fallbackLayerIdx(int g) {
        return firstInferenceIdx() + 1 + batchLayerGraphs + g;
    }

    /** The index of the {@code g}-th NARROW batch-prefill layer graph. */
    public int narrowLayerIdx(int g) {
        return firstInferenceIdx() + 1 + batchLayerGraphs + fallbackLayerGraphs + g;
    }

    /** The batch-prefill layer graphs of every family. */
    private int prefillLayerGraphs() {
        return batchLayerGraphs + fallbackLayerGraphs + narrowLayerGraphs;
    }

    public int decodeActivationIdx() {
        return firstInferenceIdx() + prefillLayerGraphs() + 1;
    }

    /**
     * The index of the {@code g}-th decode layer <b>graph</b>.
     *
     * <p>This is what the forward loop iterates. It is indexed by graph, not by layer, because a
     * graph may hold more than one layer.
     */
    public int decodeLayerGraphIdx(int g) {
        return firstInferenceIdx() + prefillLayerGraphs() + 2 + g;
    }

    public int logitsIdx() {
        return firstInferenceIdx() + prefillLayerGraphs() + 2 + decodeLayerGraphs;
    }

    // @formatter:off
    /**
     * How many distinct <b>layer graph families</b> this topology builds.
     *
     * <p>A family is a set of graphs that bind the layers' weights. This layout has 2: the
     * batch-prefill layers and the decode layers, plus one for a fallback family and one for a
     * narrow family when the plan builds them. It is <b>not</b> a graph count and <b>not</b> a
     * weight multiplier — a family that consumes another's upload costs graphs, not gigabytes,
     * which is what {@code Configuration.weightBindingFamilies} decides.
     */
    // @formatter:on
    public int layerGraphFamilies() {
        return layerFamilyGraphCounts().length;
    }

    /**
     * Graphs each family contributes, in layout order.
     *
     * <p>{@code TornadoGraphTopology} checks this against both the family count and the total, so a
     * family added without a term here fails rather than under-predicting silently.
     */
    public int[] layerFamilyGraphCounts() {
        return java.util.stream.IntStream.of(
                        batchLayerGraphs, fallbackLayerGraphs, narrowLayerGraphs, decodeLayerGraphs)
                .filter(graphs -> graphs > 0)
                .toArray();
    }

    /** Graphs that are not per-layer: batch activation, decode activation and logits. */
    public int nonLayerGraphs() {
        return 3;
    }

    /** The inference graphs, without the prepacking graphs. */
    public int inferenceGraphCount() {
        return prefillLayerGraphs() + decodeLayerGraphs + nonLayerGraphs();
    }

    /** Every graph of the plan: the prepacking graphs and the inference graphs. */
    public int totalGraphs() {
        return prepackingGraphs + inferenceGraphCount();
    }
}
