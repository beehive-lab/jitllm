package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/** The host forward pass for {@code qwen3}. */
public final class Qwen3CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen3");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaQwen3(model, state, token, position);
    }
}
