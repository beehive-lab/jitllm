package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * The host forward pass for {@code deepseek-r1-distill-qwen}.
 *
 * <p>DeepSeek-R1-Distill-Qwen is a Qwen2 by configuration and runs Qwen2's host pass, exactly as it
 * did when it inherited the method.
 */
public final class DeepSeekR1DistillQwenCpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE =
            ArchitectureId.of("deepseek-r1-distill-qwen");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaQwen2(model, state, token, position);
    }
}
