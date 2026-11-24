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
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * DeepSeek-R1-Distill-Qwen's computation, which <b>is</b> Qwen2's.
 *
 * <p>{@code DeepSeekR1Qwen extends Qwen2} and carries a {@code Qwen2Configuration}; it shares
 * Qwen2's state, plan components and description. So this delegates, for the same reason {@code
 * MistralArchitecture} does — and, like it, <b>keeps its own identity</b>, so the two signatures
 * differ and neither borrows the other's compiled program.
 */
public final class DeepSeekR1DistillQwenArchitecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("deepseek-r1-distill-qwen");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof Qwen2Configuration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a Qwen2Configuration, got "
                            + configuration.getClass().getSimpleName());
        }
    }

    @Override
    public Set<PhaseId> logicalPhases() {
        return new Qwen2Architecture().logicalPhases();
    }

    @Override
    public InferenceProgram describe(ArchitectureInputs inputs) {
        validateConfiguration(inputs.configuration());
        return Qwen2Architecture.describeAs(ID, inputs);
    }
}
