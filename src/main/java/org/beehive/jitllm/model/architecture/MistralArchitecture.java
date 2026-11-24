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
package org.beehive.jitllm.model.architecture;

import java.util.Set;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.mistral.MistralConfiguration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Mistral's computation, which <b>is</b> Llama's.
 *
 * <p>{@code Mistral.forward} calls {@code InferenceCore.forwardJava} — Llama's own method — so this
 * delegates rather than restating fifteen operations that would then have to be kept in step.
 *
 * <p><b>Explicit delegation, distinct identity.</b> It calls Llama's description with its own
 * {@link ArchitectureId}, so the two signatures differ and neither can borrow the other's compiled
 * program. Sharing a computation is not sharing an identity, and the difference is the whole reason
 * the alias is written this way rather than by registering `LlamaArchitecture` twice — which the
 * duplicate check would refuse anyway.
 */
public final class MistralArchitecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("mistral");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof MistralConfiguration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a MistralConfiguration, got "
                            + configuration.getClass().getSimpleName());
        }
    }

    @Override
    public Set<PhaseId> logicalPhases() {
        // The computation expresses both. Whether a backend runs them is registered, not described:
        // Tornado supports only STANDARD for Mistral today, and that is not this class's business.
        return new LlamaArchitecture().logicalPhases();
    }

    @Override
    public InferenceProgram describe(ArchitectureInputs inputs) {
        validateConfiguration(inputs.configuration());
        return LlamaArchitecture.describeAs(ID, inputs);
    }
}
