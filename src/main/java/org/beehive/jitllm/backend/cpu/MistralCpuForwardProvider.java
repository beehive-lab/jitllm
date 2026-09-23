package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * The host forward pass for {@code mistral}.
 *
 * <p>Mistral's host pass <b>is</b> Llama's — the same method, not a copy of it. It gets its own
 * provider because which routine an architecture runs is a provider's answer to give.
 */
public final class MistralCpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("mistral");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJava(model, state, token, position);
    }
}
