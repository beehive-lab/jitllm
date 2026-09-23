package org.beehive.jllm.backend.cpu;

import org.beehive.jllm.inference.ForwardPass;
import org.beehive.jllm.runtime.model.ArchitectureId;

/**
 * The host forward pass for {@code phi3}.
 *
 * <p>The cast is the one the family class did, kept rather than smoothed away: the routine needs a
 * Phi3State, and a state of another kind here is a wiring error worth failing on.
 */
public final class Phi3CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("phi3");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaPhi3(
                        model, (org.beehive.jllm.inference.state.Phi3State) state, token, position);
    }
}
