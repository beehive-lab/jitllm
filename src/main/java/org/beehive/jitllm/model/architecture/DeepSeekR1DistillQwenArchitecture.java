package org.beehive.jllm.model.architecture;

import java.util.Set;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.qwen2.Qwen2Configuration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.runtime.model.ArchitectureId;

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
