package org.beehive.jllm.arch.fixture.model;

import org.beehive.jllm.engine.RequestState;

/**
 * Deliberate violator, used only by {@link org.beehive.jllm.arch.DependencyRulesSelfTest} to prove
 * Rule 18 actually fails on bad code. Never imported by production code and never part of {@link
 * org.beehive.jllm.arch.ProductionClasses}.
 *
 * <p>Breaks Rule 18: a type in a model package reaching up into {@code.engine.}. If this were
 * allowed, the simple single-sequence path — the one defined by not having an engine — would
 * acquire a dependency on a scheduler it does not need.
 */
public class ViolatingEngineUser {

    /** Rule 18 — an engine type reached from below the engine. */
    public RequestState leak() {
        return RequestState.QUEUED;
    }
}
