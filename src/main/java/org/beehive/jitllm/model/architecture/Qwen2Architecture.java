package org.beehive.jitllm.model.architecture;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Qwen2's computation. Three {@code BiasAdd} between the projections and the rotation, and the
 * NEOX_HALF rotary layout.
 */
public final class Qwen2Architecture implements ModelArchitecture {

    static final ArchitectureId ID = ArchitectureId.of("qwen2");

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
        return EnumSet.allOf(PhaseId.class);
    }

    @Override
    public InferenceProgram describe(ArchitectureInputs inputs) {
        validateConfiguration(inputs.configuration());
        return describeAs(ID, inputs);
    }

    /** The same computation under another identity, for an alias that delegates here. */
    static InferenceProgram describeAs(ArchitectureId id, ArchitectureInputs inputs) {
        return Qwen2ProgramDescription.build(
                id,
                (Qwen2Configuration) inputs.configuration(),
                inputs.weights(),
                inputs.keyValue(),
                inputs.deviceSample(),
                inputs.splitKvAttention());
    }
}
