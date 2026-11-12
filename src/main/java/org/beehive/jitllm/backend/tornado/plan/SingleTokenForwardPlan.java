package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.model.Model;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

// @formatter:off
/**
 * Topology plan for the N+2 single-token forward pass.
 *
 * <p>Graph layout:
 *
 * <pre>
 *   [0 .. P-1]     prepacking   ← the session's prepacker, only with weight prepacking
 *   [P]            activation   ← singleTokenActivation()
 *   [P+1 .. P+N]   layers       ← singleTokenTransformerLayers()
 *   [P+N+1]        logits       ← singleTokenLogits(String)
 * </pre>
 *
 * <p>The prepacking graphs initialize the plan, once; the others are its inference graphs. The
 * indices come from {@link SingleTokenForwardTaskGraphLayout}.
 */
// @formatter:on
public class SingleTokenForwardPlan extends ForwardPlan {

    private final SingleTokenForwardTaskGraphLayout taskGraphLayout;

    public SingleTokenForwardPlan(
            Model model, SingleTokenForwardPlanComponents components, WeightPrepacker prepacker) {
        super(prepacker);
        // The layer graphs are built first because N is how many of them there are, which is not
        // always the layer count: a family may put several adjacent layers in one graph to cut
        // submissions. For every family that does not, the two are equal and this is the number it
        // always was.
        TransformerLayerTaskGraphs layers = components.singleTokenTransformerLayers();
        List<ImmutableTaskGraph> layerGraphs = layers.getFFNLayerImmutableTaskGraphs();
        int N = layerGraphs.size();

        List<ImmutableTaskGraph> inference = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.singleTokenActivation();
        inference.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        inference.addAll(layerGraphs);
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits =
                components.singleTokenLogits(layers.getLastFFNLayerTaskGraphID());
        inference.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        // The prepacking graphs, when the plan uses weight prepacking, come first: built from the
        // weights the layer graphs above registered. The layout puts every inference index after.
        int prepackingGraphs = assembleGraphs(inference, scheduler);
        this.taskGraphLayout = new SingleTokenForwardTaskGraphLayout(N, prepackingGraphs);
    }

    public SingleTokenForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
