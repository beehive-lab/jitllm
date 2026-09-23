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
        if (contextLength < 0)
            throw new IllegalArgumentException(
                    "--ctx-size must be non-negative (0 means the model's own)");
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
        ExecutionPolicy policy = ExecutionPolicy.fromSystemProperties();
        ExperimentalWarnings.nativeLibraries(policy);
        StartupDiagnostics.installTaskGraphChainOutput();
        return ModelOptions.builder()
                .contextLength(contextLength)
                // run, chat and the serial server each hold exactly one session per model.
                .maxConcurrentSessions(1)
                .backend(backend)
                .executionPolicy(policy)
                .build();
    }
}
