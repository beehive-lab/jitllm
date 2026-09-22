package org.beehive.jllm.integration.cli;

import java.nio.file.Path;
import java.util.Objects;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.backend.DeviceResolvers;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/** Common model-loading settings for terminal and HTTP integrations. */
public record ModelRunConfig(Path model, int contextLength, boolean gpu) {
    public ModelRunConfig {
        Objects.requireNonNull(model, "model");
        if (contextLength <= 0) throw new IllegalArgumentException("--ctx-size must be positive");
    }

    public ModelOptions modelOptions() {
        BackendId backend = BackendId.CPU;
        if (gpu) {
            backend =
                    DeviceResolvers.discovered()
                            .orElseThrow(
                                    () ->
                                            new IllegalArgumentException(
                                                    "--gpu requested but no accelerator backend is installed"))
                            .resolve()
                            .backend();
            if (backend.equals(BackendId.CPU))
                throw new IllegalArgumentException(
                        "--gpu requested but no accelerator device is available");
        }
        return ModelOptions.builder()
                .contextLength(contextLength)
                .backend(backend)
                .executionPolicy(ExecutionPolicy.fromSystemProperties())
                .build();
    }
}
