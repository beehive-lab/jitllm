package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
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
 *   [0]      activation   ← singleTokenActivation()
 *   [1.N]   layers       ← singleTokenTransformerLayers()
 *   [N+1]    logits       ← singleTokenLogits(String)
 * </pre>
 */
// @formatter:on
public class SingleTokenForwardPlan extends ForwardPlan {

    private final SingleTokenForwardTaskGraphLayout taskGraphLayout;

    public SingleTokenForwardPlan(Model model, SingleTokenForwardPlanComponents components) {
        // The layer graphs are built first because N is how many of them there are, which is not
        // always the layer count: a family may put several adjacent layers in one graph to cut
        // submissions. For every family that does not, the two are equal and this is the number it
        // always was.
        TransformerLayerTaskGraphs layers = components.singleTokenTransformerLayers();
        List<ImmutableTaskGraph> layerGraphs = layers.getFFNLayerImmutableTaskGraphs();
        int N = layerGraphs.size();
        this.taskGraphLayout = new SingleTokenForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.singleTokenActivation();
        all.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        all.addAll(layerGraphs);
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits =
                components.singleTokenLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public SingleTokenForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
