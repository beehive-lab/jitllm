package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/** The host forward pass of {@code deepseek4}. */
public final class DeepSeek4CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("deepseek4");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return DeepSeek4Forward::forward;
    }
}
