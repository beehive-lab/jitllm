package org.beehive.jllm.backend.cpu;

import org.beehive.jllm.inference.ForwardPass;
import org.beehive.jllm.runtime.model.ArchitectureId;

/** The host forward pass for {@code gemma4}. */
public final class Gemma4CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("gemma4");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaGemma4(model, state, token, position);
    }
}
