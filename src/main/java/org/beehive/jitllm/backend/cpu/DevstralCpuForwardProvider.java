package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * The host forward pass for {@code devstral}.
 *
 * <p>Devstral is fixture-blocked and its legacy components stand. This moves who calls its routine;
 * it removes nothing.
 */
public final class DevstralCpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("devstral");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaDevstral(model, state, token, position);
    }
}
