package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * One architecture's lowering, as this backend offers it.
 *
 * <p>What a provider declares is <b>backend capability</b>: which materialized weight
 * representations it lowers and which execution modes it has plans for. That is not something the
 * architecture can know — it depends on which kernels were written — which is why it is declared
 * here and not on {@code ModelArchitecture}.
 *
 * <h2>Aliases</h2>
 *
 * <p>Two identities may share a lowering implementation — Mistral shares Llama's, DeepSeek shares
 * Qwen2's — but they are <b>separate providers with separate identities</b>. Sharing an
 * implementation is not sharing an identity: the lowering a provider creates validates programs
 * under its own {@link #architecture()}, so a delegation can never quietly make two architectures
 * one program.
 */
public interface TornadoLoweringProvider {

    /** The identity this provider lowers. Two providers claiming one identity is an error. */
    ArchitectureId architecture();

    /** The materialized weight representations this lowering handles. */
    Set<DataType> supportedDataTypes();

    /** The execution modes this backend has plans for, for this architecture. */
    Set<ExecutionMode> supportedModes();

    /** Builds the lowering. Called once per compiled program, never per token. */
    FamilyLowering create(CompileOptions options, DeviceCapabilities capabilities);
}
