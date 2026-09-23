package org.beehive.jllm.arch.fixture.metrics;

import org.beehive.jllm.model.AbstractModel;

/**
 * Deliberate Rule 17 violation: a type inside the metrics seam that depends on a layer above it.
 * Exists so {@code DependencyRulesSelfTest} can prove the rule reports something — a rule only ever
 * seen to pass is not a guardrail.
 */
public class ViolatingSeam {

    private final AbstractModel model;

    public ViolatingSeam(AbstractModel model) {
        this.model = model;
    }

    public AbstractModel model() {
        return model;
    }
}
