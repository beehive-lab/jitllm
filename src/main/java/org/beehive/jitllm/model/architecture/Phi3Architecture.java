package org.beehive.jllm.model.architecture;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.phi3.Phi3Configuration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.runtime.model.ArchitectureId;

/** Phi3's computation. Two fused projections, each followed by its split. */
public final class Phi3Architecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("phi3");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof Phi3Configuration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a Phi3Configuration, got "
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
        return Phi3ProgramDescription.build(
                id,
                (Phi3Configuration) inputs.configuration(),
                inputs.weights(),
                inputs.keyValue(),
                inputs.deviceSample(),
                inputs.splitKvAttention());
    }
}
