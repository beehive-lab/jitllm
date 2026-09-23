package org.beehive.jllm.backend.tornado.plan;

import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.runtime.model.ArchitectureId;

/**
 * The cast the central factory used to do, in one family-neutral place.
 *
 * <p>It names no architecture: the identity is a parameter. What it replaces is a raw {@code
 * ClassCastException} naming two internal classes and neither the architecture nor the expectation.
 */
public final class PlanStates {

    private PlanStates() {}

    /**
     * @throws IllegalArgumentException naming the architecture, what was needed and what arrived
     */
    public static <T extends State> T expect(Class<T> type, State state, ArchitectureId id) {
        if (!type.isInstance(state)) {
            throw new IllegalArgumentException(
                    "'"
                            + id
                            + "' needs a "
                            + type.getSimpleName()
                            + ", got "
                            + state.getClass().getSimpleName()
                            + "; the model and its state disagree about which family this is");
        }
        return type.cast(state);
    }
}
