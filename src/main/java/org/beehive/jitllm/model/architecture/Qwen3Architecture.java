package org.beehive.jllm.model.architecture;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.runtime.model.ArchitectureId;

/** Qwen3's computation. Grouped per-head query and key norms, and three head dimensions. */
public final class Qwen3Architecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("qwen3");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof Qwen3Configuration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a Qwen3Configuration, got "
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

    /** The same computation under another identity, for an alias that delegates here. */
    static InferenceProgram describeAs(ArchitectureId id, ArchitectureInputs inputs) {
        return Qwen3ProgramDescription.build(
                id,
                (Qwen3Configuration) inputs.configuration(),
                inputs.weights(),
                inputs.keyValue(),
                inputs.deviceSample(),
                inputs.splitKvAttention());
    }
}
