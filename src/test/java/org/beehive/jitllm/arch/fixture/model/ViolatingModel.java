package org.beehive.jitllm.arch.fixture.model;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Deliberate violator, used only by {@link org.beehive.jitllm.arch.DependencyRulesSelfTest} to prove
 * the rules actually fail on bad code. Never imported by production code and never part of {@link
 * org.beehive.jitllm.arch.ProductionClasses}.
 *
 * <p>Breaks Rule 1 and Rule 2 (imports TornadoVM outside the backend, from a model package), Rule 5
 * (non-final field) and Rule 11 (references TaskGraph).
 */
public class ViolatingModel {

    /** Rule 5 — mutable field on a model type. */
    public FloatArray logits;

    /** Rules 1, 2 and 11 — a plan type in a signature outside the backend. */
    public TaskGraph leak(TaskGraph graph) {
        return graph;
    }
}
