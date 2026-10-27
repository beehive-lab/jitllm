package org.beehive.jitllm.backend.tornado.plan.prepacking;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;

/**
 * The prepacking {@link TaskGraphs} of one plan, built by {@link WeightPrepacker#buildTaskGraphs}: the plan's
 * initialization, which the plan lists ahead of its inference graphs.
 *
 * <p>They are created after the layer graphs, which register the weights to prepack, but they come
 * first in the list and run first: the plan runs them once, before its warm-up ({@link #run}), and
 * no inference step selects them. Should they execute again, their kernels skip the work, because
 * the last of them set a completion flag on the device.
 */
public final class WeightPrepackingTaskGraphs {

    /**
     * No prepacking graphs, for a plan that does not use weight prepacking. Running it does
     * nothing.
     */
    public static final WeightPrepackingTaskGraphs NONE =
            new WeightPrepackingTaskGraphs(List.of(), Map.of());

    private static final System.Logger LOGGER =
            System.getLogger(WeightPrepackingTaskGraphs.class.getName());

    private final List<ImmutableTaskGraph> graphs;

    /** The work grid of each of their tasks, by {@code graph.task} name, in creation order. */
    private final Map<String, WorkerGrid> grids;

    WeightPrepackingTaskGraphs(List<ImmutableTaskGraph> graphs, Map<String, WorkerGrid> grids) {
        this.graphs = List.copyOf(graphs);
        this.grids = Collections.unmodifiableMap(new LinkedHashMap<>(grids));
    }

    /** The prepacking graphs, one per layer graph that reads prepacked weights. */
    public List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return graphs;
    }

    /** How many prepacking graphs there are: the length of the plan's initialization prefix. */
    public int graphCount() {
        return graphs.size();
    }

    /** Adds the work grids of the prepacking graphs' tasks to {@code scheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
    }

    /**
     * Executes the prepacking graphs, the first {@link #graphCount} graphs of {@code plan}, with
     * the grids of {@code scheduler}: each uploads its weights and rearranges them into the
     * prepacked layout. Leaves the last of them selected; callers select the graphs they run next.
     * Does nothing, and logs nothing, when there are none.
     */
    public void run(TornadoExecutionPlan plan, GridScheduler scheduler) {
        if (graphs.isEmpty()) {
            return;
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < graphs.size(); i++) {
            plan.withGraph(i).withGridScheduler(scheduler).execute();
        }
        LOGGER.log(
                System.Logger.Level.INFO,
                String.format(
                        "weights prepacked on the GPU in %.1f ms (%d graphs)",
                        (System.nanoTime() - t0) / 1e6, graphs.size()));
    }
}
