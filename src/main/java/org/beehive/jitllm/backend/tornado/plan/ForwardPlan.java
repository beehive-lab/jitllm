package org.beehive.jllm.backend.tornado.plan;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Abstract base for GPU forward-pass execution plans.
 *
 * <p>Subclasses assemble the {@link ImmutableTaskGraph} list and {@link GridScheduler} in their
 * constructors by calling {@link #setGraphs}, then expose the results via {@link
 * #getImmutableTaskGraphs} and {@link #getGridScheduler}.
 */
public abstract class ForwardPlan {

    private List<ImmutableTaskGraph> immutableTaskGraphs;
    private GridScheduler gridScheduler;

    protected final void setGraphs(List<ImmutableTaskGraph> itgs, GridScheduler scheduler) {
        this.immutableTaskGraphs = itgs;
        this.gridScheduler = scheduler;
    }

    public final List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return immutableTaskGraphs;
    }

    public final GridScheduler getGridScheduler() {
        return gridScheduler;
    }
}
