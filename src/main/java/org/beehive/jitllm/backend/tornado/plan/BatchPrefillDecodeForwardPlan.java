package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.PrefillFamily;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.model.Model;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

// @formatter:off
/**
 * Topology plan for the 2N+3 batch-prefill/decode forward pass.
 *
 * <p>Graph layout:
 *
 * <pre>
 *   [0 .. P-1]           prepacking          ← the session's prepacker, only with weight prepacking
 *   [P]                  batch activation    ← batchPrefillActivation(int)
 *   [P+1 .. P+N]         batch layers        ← batchPrefillTransformerLayers(int)
 *   then                 fallback, narrow    ← the same layers again, when built
 *   [P+N+1]              decode activation   ← batchDecodeActivation(String)
 *   [P+N+2 .. P+2N+1]    decode layers       ← batchDecodeTransformerLayers()
 *   [P+2N+2]             logits              ← decodeLogits(String)
 * </pre>
 *
 * <p>The prepacking graphs initialize the plan, once. During batch prefill, the master plan
 * executes the batch activation and layers; during decode, the decode activation, layers and
 * logits. The indices, including those of a fallback family, come from {@link
 * BatchPrefillDecodeForwardTaskGraphLayout}.
 */
// @formatter:on
public class BatchPrefillDecodeForwardPlan extends ForwardPlan {

    private final BatchPrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    /** The batched prefill layers the plan was built from, for diagnostics of what they chose. */
    private final BatchPrefillTransformerLayerTaskGraphs batchPrefillLayers;

    /** The narrow family's rows, or 0 when the plan has none. */
    private final int narrowPrefillRows;

    /** What the narrow family's build bound and allocated, or why there is none. */
    private final String narrowPrefillDescription;

    /** The per-chunk arrays the narrow family reads, by the primary layer graph that holds them. */
    private final Map<Integer, Object[]> narrowRefreshes;

    public BatchPrefillDecodeForwardPlan(
            Model model,
            BatchPrefillDecodeForwardPlanComponents components,
            int batchSize,
            TornadoWorkspace workspace) {
        super(workspace.prepacker);
        int N = model.configuration().numberOfLayers();

        List<ImmutableTaskGraph> inference = new ArrayList<>(2 * N + 3);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph batchAct = components.batchPrefillActivation(batchSize);
        inference.add(batchAct.getImmutableTaskGraph());
        batchAct.updateGridScheduler(scheduler);

        BatchPrefillTransformerLayerTaskGraphs batchLayers;
        List<ImmutableTaskGraph> batchLayerGraphs;
        List<ImmutableTaskGraph> fallbackLayerGraphs;
        List<ImmutableTaskGraph> narrowLayerGraphs = List.of();
        Map<Integer, Object[]> refreshes = Map.of();
        String off =
                System.getProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
        String narrowDescription =
                "false".equalsIgnoreCase(off)
                        ? "none, turned off"
                        : "none, the plan is one tile wide";
        int narrowRows =
                BatchPrefillTransformerLayerTaskGraphs.narrowRows((batchSize + 127) & ~127);
        PrefillFamily primary = PrefillFamily.primary();
        batchLayers =
                primary.build(workspace, () -> components.batchPrefillTransformerLayers(batchSize));
        batchLayerGraphs = batchLayers.getLayerImmutableTaskGraphs();
        inference.addAll(batchLayerGraphs);
        batchLayers.updateGridScheduler(scheduler);

        // The fallback family, when the primary builds one: the same layers with an attention
        // implementation that handles a chunk starting past position 0. Its graphs bind their
        // buffers from the primary's, so they cost graphs and no memory. Placed immediately after
        // the primary so the decode side's producer names are unaffected.
        fallbackLayerGraphs = batchLayers.getFallbackLayerImmutableTaskGraphs();
        inference.addAll(fallbackLayerGraphs);
        batchLayers.updateFallbackGridScheduler(scheduler);

        // The narrow family: the same layers built again at fewer rows, for a chunk with no more
        // tokens than that. Its graphs bind the arrays of the primary's.
        if (narrowRows > 0) {
            long start = System.nanoTime();
            PrefillFamily family = primary.narrow(narrowRows, batchLayers);
            try {
                BatchPrefillTransformerLayerTaskGraphs narrow =
                        family.build(
                                workspace,
                                () -> components.batchPrefillTransformerLayers(narrowRows));
                narrow.updateGridScheduler(scheduler);
                narrowLayerGraphs = narrow.getLayerImmutableTaskGraphs();
                inference.addAll(narrowLayerGraphs);
                // Its twins are the primary's graphs, or the fallback's under cuDNN, which follow.
                List<ImmutableTaskGraph> twins = new ArrayList<>(batchLayerGraphs);
                twins.addAll(fallbackLayerGraphs);
                refreshes = family.refreshes(twins);
                narrowDescription =
                        String.format(
                                Locale.ROOT,
                                "%d graphs, %d rows, %d arrays bound from the primary, %d of its"
                                        + " own (%d KiB), built in %.2f s",
                                narrowLayerGraphs.size(),
                                narrowRows,
                                family.boundArrays(),
                                family.ownArrays(),
                                family.ownBytes() / 1024,
                                (System.nanoTime() - start) / 1e9);
            } catch (PrefillFamily.Declined e) {
                narrowDescription = "none, " + e.getMessage();
            }
        }
        this.batchPrefillLayers = batchLayers;
        this.narrowPrefillRows = narrowLayerGraphs.isEmpty() ? 0 : narrowRows;
        this.narrowPrefillDescription = narrowDescription;
        this.narrowRefreshes = refreshes;

        ActivationTaskGraph decodeAct =
                components.batchDecodeActivation(batchLayers.getLastLayerTaskGraphID());
        inference.add(decodeAct.getImmutableTaskGraph());
        decodeAct.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs decodeLayers = components.batchDecodeTransformerLayers();
        List<ImmutableTaskGraph> decodeLayerGraphs = decodeLayers.getFFNLayerImmutableTaskGraphs();
        inference.addAll(decodeLayerGraphs);
        decodeLayers.updateGridScheduler(scheduler);
        AbstractLogitsTaskGraph logits =
                components.decodeLogits(decodeLayers.getLastFFNLayerTaskGraphID());
        inference.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        // The prepacking graphs, when the plan uses weight prepacking, come first: built from the
        // weights the layer graphs above registered. The layout puts every inference index after.
        int prepackingGraphs = assembleGraphs(inference, scheduler);
        // Read from the graphs each family actually built rather than assumed to be one per
        // layer: either side may hold several layers in one graph, and every index after them
        // depends on how many there are.
        this.taskGraphLayout =
                new BatchPrefillDecodeForwardTaskGraphLayout(
                        N,
                        batchLayerGraphs.size(),
                        fallbackLayerGraphs.size(),
                        decodeLayerGraphs.size(),
                        prepackingGraphs,
                        narrowLayerGraphs.size());
    }

    public BatchPrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }

    public BatchPrefillTransformerLayerTaskGraphs getBatchPrefillLayers() {
        return batchPrefillLayers;
    }

    /** The narrow family's rows, or 0 when the plan has none. */
    public int narrowPrefillRows() {
        return narrowPrefillRows;
    }

    /**
     * The per-chunk arrays the narrow family reads from the primary's buffers, by the index of the
     * graph holding them among the primary's layer graphs and the fallback's after them: a narrow
     * chunk uploads them there before it runs.
     */
    public Map<Integer, Object[]> narrowRefreshes() {
        return narrowRefreshes;
    }

    /**
     * What the narrow family's build bound and allocated, or why there is none.
     *
     * <p>Example: {@code 16 graphs, 128 rows, 157 arrays bound from the primary, 2 of its own (0
     * KiB), built in 0.02 s}.
     */
    public String describeNarrowPrefill() {
        return narrowPrefillDescription;
    }
}
