package org.beehive.jitllm.backend.tornado.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Common interface for a single activation task graph (standard, decode, and batch-prefill
 * variants).
 *
 * <p>Implemented by {@link Activation} and custom activation wrappers used by {@link
 * org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents}.
 */
public interface ActivationTaskGraph {
    ImmutableTaskGraph getImmutableTaskGraph();

    GridScheduler updateGridScheduler(GridScheduler scheduler);
}
