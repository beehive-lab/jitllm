package org.beehive.jllm.backend.tornado.plan.layout;

import org.beehive.jllm.backend.tornado.plan.ExecutionMode;

/**
 * How many layer graph families each execution mode builds — read from the layouts themselves.
 *
 * <p><b>Derived, not declared.</b> The count comes from the layout records that already describe
 * each topology, so the answer lives beside the graph indices rather than in a table someone must
 * remember to update. {@link #verify} then checks each layout's own arithmetic against the graphs
 * it says each family contributes, which is what makes an added family a test failure instead of a
 * silent under-prediction.
 *
 * <p><b>Families are not graphs.</b> A family may build one graph per layer or fewer, so the check
 * is {@code totalGraphs == sum(layerFamilyGraphCounts) + nonLayerGraphs} together with {@code
 * layerFamilyGraphCounts.length == layerGraphFamilies}. The older form, {@code families × N +
 * nonLayerGraphs}, is the special case where every family builds a graph per layer, and it is still
 * asserted for the layouts that do. Neither form says anything about weight memory: a family that
 * consumes another's upload costs graphs and not gigabytes, which {@code
 * Configuration.weightBindingFamilies} decides.
 *
 * <p><b>Exhaustive by construction.</b> The switch has no {@code default}, so a new {@link
 * ExecutionMode} does not compile until it states its answer here.
 */
public final class TornadoGraphTopology {

    /** An arbitrary layer count for structural checks; the identity holds for every N. */
    private static final int PROBE_LAYERS = 16;

    private TornadoGraphTopology() {}

    /** Layer graph families for {@code mode}. */
    public static int layerGraphFamilies(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).layerGraphFamilies();
            case PREFILL_DECODE ->
                    new PrefillDecodeForwardTaskGraphLayout(layers).layerGraphFamilies();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).layerGraphFamilies();
        };
    }

    /** Graphs that are not per-layer, for {@code mode}. */
    public static int nonLayerGraphs(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).nonLayerGraphs();
            case PREFILL_DECODE -> new PrefillDecodeForwardTaskGraphLayout(layers).nonLayerGraphs();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).nonLayerGraphs();
        };
    }

    /** Graphs each layer family contributes, in layout order, for {@code mode}. */
    public static int[] layerFamilyGraphCounts(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).layerFamilyGraphCounts();
            case PREFILL_DECODE ->
                    new PrefillDecodeForwardTaskGraphLayout(layers).layerFamilyGraphCounts();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).layerFamilyGraphCounts();
        };
    }

    /** Total graphs for {@code mode}. */
    public static int totalGraphs(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).totalGraphs();
            case PREFILL_DECODE -> new PrefillDecodeForwardTaskGraphLayout(layers).totalGraphs();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).totalGraphs();
        };
    }

    /** Whether a mode's declared family count agrees with the graphs it actually lays out. */
    public static boolean verify(ExecutionMode mode, int layers) {
        int[] counts = layerFamilyGraphCounts(mode, layers);
        if (counts.length != layerGraphFamilies(mode, layers)) {
            return false;
        }
        int layerGraphs = 0;
        for (int count : counts) {
            if (count < 1 || count > layers) {
                return false;
            }
            layerGraphs += count;
        }
        return totalGraphs(mode, layers) == layerGraphs + nonLayerGraphs(mode, layers);
    }

    /**
     * Whether every family of {@code mode} builds one graph per layer.
     *
     * <p>True of every layout as laid out by default. A layout that groups layers into fewer graphs
     * answers false, and {@link #verify} still holds for it.
     */
    public static boolean isUngrouped(ExecutionMode mode, int layers) {
        for (int count : layerFamilyGraphCounts(mode, layers)) {
            if (count != layers) {
                return false;
            }
        }
        return true;
    }

    /** Every selectable mode agrees with its own layout arithmetic. */
    public static boolean verifyAll() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            if (!verify(mode, PROBE_LAYERS)) {
                return false;
            }
        }
        return true;
    }
}
