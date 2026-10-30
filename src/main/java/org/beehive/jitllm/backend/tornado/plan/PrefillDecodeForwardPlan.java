package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.model.Model;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

// @formatter:off
/**
 * Topology plan for the N+2 prefill/decode forward pass.
 *
 * <p>Graph layout:
 *
 * <pre>
 *   [0 .. P-1]     prepacking   ← the session's prepacker, only with weight prepacking
 *   [P]            activation   ← prefillDecodeActivation()
 *   [P+1 .. P+N]   layers       ← prefillDecodeTransformerLayers()
 *   [P+N+1]        logits       ← decodeLogits(String)
 * </pre>
 *
 * <p>The prepacking graphs initialize the plan, once. During prefill, the master plan executes the
 * activation and the layers (skipping logits); during decode, all N+2 inference graphs. The indices
 * come from {@link PrefillDecodeForwardTaskGraphLayout}.
 */
// @formatter:on
public class PrefillDecodeForwardPlan extends ForwardPlan {

    private final PrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    public PrefillDecodeForwardPlan(
            Model model, PrefillDecodeForwardPlanComponents components, WeightPrepacker prepacker) {
        super(prepacker);
        int N = model.configuration().numberOfLayers();
        List<ImmutableTaskGraph> inference = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.prefillDecodeActivation();
        inference.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs layers = components.prefillDecodeTransformerLayers();
        inference.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits =
                components.decodeLogits(layers.getLastFFNLayerTaskGraphID());
        inference.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        // The prepacking graphs, when the plan uses weight prepacking, come first: built from the
        // weights the layer graphs above registered. The layout puts every inference index after.
        int prepackingGraphs = assembleGraphs(inference, scheduler);
        this.taskGraphLayout = new PrefillDecodeForwardTaskGraphLayout(N, prepackingGraphs);
    }

    public PrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
