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
package org.beehive.jitllm.backend.tornado.lowering;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * One family's lowering: it validates the programs it claims and builds their compiled form.
 *
 * <p>Extracted when the second family arrived, which is what {@code LlamaLowering}'s own comment
 * said to wait for — an abstraction drawn from one example would have been drawn from a guess. What
 * it shares is deliberately thin: what a lowering <i>is</i>, not how any family is shaped.
 *
 * <p><b>Each implementation states its own layer sequence, in order, as explicit checks.</b> There
 * is no rule table, no pattern language and no fusion-rule engine. Two families that look similar
 * are still written out twice, because the similarity is not a fact about the architecture — it is
 * a fact about these two families today.
 */
interface FamilyLowering {

    /** The architecture identity this lowering's programs carry. */
    ArchitectureId architecture();

    /** Whether this lowering can handle {@code program}, without throwing. */
    default boolean supports(InferenceProgram program) {
        try {
            validate(program);
            return true;
        } catch (UnsupportedProgramException e) {
            return false;
        }
    }

    /**
     * @throws UnsupportedProgramException naming the first thing that did not match
     */
    void validate(InferenceProgram program);

    /** Validates, then builds the compiled program. */
    TornadoVMMasterPlan lower(InferenceProgram program, State state, Model model, MetricsSink sink);
}
