package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout;
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
 *   [0]      activation   ← prefillDecodeActivation()
 *   [1.N]   layers       ← prefillDecodeTransformerLayers()
 *   [N+1]    logits       ← decodeLogits(String)
 * </pre>
 *
 * <p>During prefill, the master plan executes graphs 0.N (skipping logits). During decode, all N+2
 * graphs run.
 */
// @formatter:on
public class PrefillDecodeForwardPlan extends ForwardPlan {

    private final PrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    public PrefillDecodeForwardPlan(Model model, PrefillDecodeForwardPlanComponents components) {
        int N = model.configuration().numberOfLayers();
        this.taskGraphLayout = new PrefillDecodeForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.prefillDecodeActivation();
        all.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs layers = components.prefillDecodeTransformerLayers();
        all.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits =
                components.decodeLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public PrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
