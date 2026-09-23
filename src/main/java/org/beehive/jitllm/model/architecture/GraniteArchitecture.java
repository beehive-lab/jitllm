package org.beehive.jitllm.model.architecture;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Granite's computation. Llama with four {@code Scale} components and an attention scale that
 * multiplies.
 */
public final class GraniteArchitecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("granite");

    @Override
    public ArchitectureId id() {
        return ID;
    }

    @Override
    public void validateConfiguration(Configuration configuration) {
        if (!(configuration instanceof GraniteConfiguration)) {
            throw new IllegalArgumentException(
                    ID
                            + " needs a GraniteConfiguration, got "
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
        return GraniteProgramDescription.build(
                id,
                (GraniteConfiguration) inputs.configuration(),
                inputs.weights(),
                inputs.keyValue(),
                inputs.deviceSample(),
                inputs.splitKvAttention());
    }
}
