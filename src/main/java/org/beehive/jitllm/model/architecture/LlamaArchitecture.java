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

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Llama's computation.
 *
 * <p>It expresses <b>both logical phases</b>, and says nothing about whether a backend can run them
 * — the Tornado backend happens to support all three execution modes for Llama and only {@code
 * STANDARD} for Mistral, which shares this exact computation. That difference lives in the
 * backend's registration, not here.
 */
public final class LlamaArchitecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("llama");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof LlamaConfiguration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a LlamaConfiguration, got "
                            + configuration.getClass().getSimpleName());
        }
    }

    @Override
    public Set<PhaseId> logicalPhases() {
        return EnumSet.allOf(PhaseId.class);
    }

    @Override
    public InferenceProgram describe(ArchitectureInputs inputs) {
        validateConfiguration(inputs.configuration());
        return describeAs(ID, inputs);
    }

    /**
     * The same computation under another identity, for an alias that delegates here.
     *
     * <p>Package-private and explicit: {@code MistralArchitecture} calls it rather than extending
     * this class or being handed a string. The identity stays its own, so the two produce distinct
     * signatures and can never share a compiled program by accident.
     */
    static InferenceProgram describeAs(ArchitectureId id, ArchitectureInputs inputs) {
        return LlamaProgramDescription.build(
                id,
                inputs.configuration(),
                inputs.weights(),
                inputs.keyValue(),
                inputs.deviceSample(),
                inputs.splitKvAttention());
    }
}
