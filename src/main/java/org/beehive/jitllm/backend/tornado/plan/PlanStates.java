/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.backend.tornado.plan;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.runtime.model.ArchitectureId;

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
