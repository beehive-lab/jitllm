package org.beehive.jitllm.backend.tornado.ops;

import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;

/** An {@link Implementation} after allocation: it adds the tasks of its steps to task graphs. */
public interface Instance {

    /** Buffers that all layer graphs share: the first graph uploads them, the others reuse them. */
    default Object[] buffers() {
        return new Object[0];
    }

    /**
     * Adds the tasks of {@code step} to {@code graph}, and records in {@code bindings} the weights
     * those tasks read.
     */
    void emit(TaskGraph graph, Step step, Operands operands, WeightBindings bindings);

    /** Adds the worker grids of the tasks {@link #emit} added. */
    void addGrids(GridScheduler scheduler);
}
