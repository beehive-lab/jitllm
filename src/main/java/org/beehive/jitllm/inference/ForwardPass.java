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
package org.beehive.jitllm.inference;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;

/**
 * One forward pass, without saying what performs it.
 *
 * <p><b>As small as the loop requires.</b> One method, returning nothing, because that is exactly
 * what {@code Model.forward} was. No result type, no options, no lifecycle: this is the shape of a
 * call that already exists, not a backend framework.
 *
 * <p><b>Where it is not.</b> Not in {@code runtime.backend}, which stays small and must not name
 * {@link Model} or {@link State}; not in {@code program}, which describes computation and has no
 * business naming a mutable state.
 */
public interface ForwardPass {

    /**
     * Runs one token through the model, writing its results into {@code state}.
     *
     * @param position the token's position in the sequence
     */
    void forward(Model model, State state, int token, int position);
}
