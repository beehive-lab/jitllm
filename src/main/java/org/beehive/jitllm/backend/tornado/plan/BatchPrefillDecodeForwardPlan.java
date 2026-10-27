package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
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

    public BatchPrefillDecodeForwardPlan(
            Model model,
            BatchPrefillDecodeForwardPlanComponents components,
            int batchSize,
            WeightPrepacker prepacker) {
        super(prepacker);
        int N = model.configuration().numberOfLayers();

        List<ImmutableTaskGraph> inference = new ArrayList<>(2 * N + 3);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph batchAct = components.batchPrefillActivation(batchSize);
        inference.add(batchAct.getImmutableTaskGraph());
        batchAct.updateGridScheduler(scheduler);

        BatchPrefillTransformerLayerTaskGraphs batchLayers =
                components.batchPrefillTransformerLayers(batchSize);
        this.batchPrefillLayers = batchLayers;
        List<ImmutableTaskGraph> batchLayerGraphs = batchLayers.getLayerImmutableTaskGraphs();
        inference.addAll(batchLayerGraphs);
        batchLayers.updateGridScheduler(scheduler);

        // The fallback family, when the primary builds one: the same layers with an attention
        // implementation that handles a chunk starting past position 0. Its graphs bind their
        // buffers from the primary's, so they cost graphs and no memory. Placed immediately after
        // the primary so the decode side's producer names are unaffected.
        List<ImmutableTaskGraph> fallbackLayerGraphs =
                batchLayers.getFallbackLayerImmutableTaskGraphs();
        inference.addAll(fallbackLayerGraphs);
        batchLayers.updateFallbackGridScheduler(scheduler);

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
                        prepackingGraphs);
    }

    public BatchPrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }

    public BatchPrefillTransformerLayerTaskGraphs getBatchPrefillLayers() {
        return batchPrefillLayers;
    }
}
