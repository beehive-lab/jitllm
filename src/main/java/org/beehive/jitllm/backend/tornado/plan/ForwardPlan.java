package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepackingTaskGraphs;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

/**
 * Abstract base for GPU forward-pass execution plans.
 *
 * <p>Subclasses build their inference graphs and grids in their constructors, then hand them to
 * {@link #assembleGraphs}, which completes the plan: the prepacking graphs first, when the plan
 * uses weight prepacking, its initialization, then the inference graphs. The results are exposed
 * via {@link #getImmutableTaskGraphs}, {@link #getGridScheduler} and {@link #getWeightPrepacking};
 * each subclass's layout records every position.
 */
public abstract class ForwardPlan {

    /**
     * The session's prepacker: the same one its layer builders register the weights to prepack
     * with, so the prepacking graphs it builds are this plan's.
     */
    protected final WeightPrepacker prepacker;

    private List<ImmutableTaskGraph> immutableTaskGraphs;
    private GridScheduler gridScheduler;
    private WeightPrepackingTaskGraphs weightPrepacking = WeightPrepackingTaskGraphs.NONE;

    protected ForwardPlan(WeightPrepacker prepacker) {
        this.prepacker = prepacker;
    }

    /**
     * Completes the plan from its {@code inferenceGraphs}, whose grids are in {@code scheduler}.
     * Builds the prepacking graphs for the weights those graphs registered with {@link #prepacker},
     * lists them first and the inference graphs after them in a new list, and merges their grids
     * into {@code scheduler}. Leaves {@code inferenceGraphs} unchanged.
     *
     * @return how many prepacking graphs come first, for the layout: 0, with nothing allocated,
     *     when the plan does not use weight prepacking
     */
    protected final int assembleGraphs(
            List<ImmutableTaskGraph> inferenceGraphs, GridScheduler scheduler) {
        WeightPrepackingTaskGraphs prepacking = prepacker.buildTaskGraphs();
        prepacking.updateGridScheduler(scheduler);
        List<ImmutableTaskGraph> all =
                new ArrayList<>(prepacking.graphCount() + inferenceGraphs.size());
        all.addAll(prepacking.getImmutableTaskGraphs());
        all.addAll(inferenceGraphs);
        this.immutableTaskGraphs = all;
        this.gridScheduler = scheduler;
        this.weightPrepacking = prepacking;
        return prepacking.graphCount();
    }

    public final List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return immutableTaskGraphs;
    }

    public final GridScheduler getGridScheduler() {
        return gridScheduler;
    }

    /** The plan's prepacking graphs; {@link WeightPrepackingTaskGraphs#NONE} without prepacking. */
    public final WeightPrepackingTaskGraphs getWeightPrepacking() {
        return weightPrepacking;
    }

    /**
     * Runs the plan's prepacking graphs in {@code plan}, built from this plan's graphs: the
     * initialization step, which the master plan takes once, before its warm-up. Nothing when the
     * plan does not use weight prepacking.
     */
    public final void runWeightPrepacking(TornadoExecutionPlan plan) {
        weightPrepacking.run(plan, gridScheduler);
    }
}
