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

/**
 * Where a generation loop reads its continuation seed and records the token it produced.
 *
 * <p>Deliberately <b>not</b> routed through the compiled program's invocation boundary. A cursor is
 * not a per-kernel input and belongs to neither the program nor the binding domain.
 */
public interface GenerationCursor {

    /** The token a continued conversation resumes from; the family's initial seed for a new one. */
    int seed();

    /** Records the token just produced. */
    void advance(int token);

    /**
     * The legacy adapter: the cursor is the {@code State}'s own field, exactly as before.
     *
     * <p>Every path but the lowered one uses this, so their behaviour is unchanged — including the
     * families whose initial seed is not repeated in the prompt, which depend on the seed being
     * whatever {@code createNewState} put there.
     */
    static GenerationCursor forState(State state) {
        return new GenerationCursor() {
            @Override
            public int seed() {
                return state.latestToken;
            }

            @Override
            public void advance(int token) {
                state.latestToken = token;
            }
        };
    }
}
