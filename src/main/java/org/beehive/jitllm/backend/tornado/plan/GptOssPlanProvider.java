package org.beehive.jitllm.backend.tornado.plan;

import java.util.EnumSet;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.gptoss.GptOssPlanComponents;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * gpt-oss on the device: Q8_0 attention and vocabulary with MXFP4 experts, which is what
 * llama.cpp's {@code ggml-org/gpt-oss-*-GGUF} files hold. The model reports Q8_0, the type of its
 * attention projections; the layers read the experts as MXFP4.
 */
public final class GptOssPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("gpt-oss");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return EnumSet.of(DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return EnumSet.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        return new GptOssPlanComponents(PlanStates.expect(GptOssState.class, state, ID), model);
    }

    /** The batched prefill runs its projections on CUDA int8 tensor cores. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.CUDA.equals(c.backend()) && c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the gpt-oss batched prefill runs its projections on CUDA int8 tensor"
                                + " cores, which this device does not have");
    }

    /** The gpt-oss kernels reduce with warp shuffles, which only the CUDA backend lowers. */
    @Override
    public Optional<String> deviceUnsupported(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                ? Optional.empty()
                : Optional.of("the gpt-oss layers are written for the CUDA backend");
    }

    /** The decode and batched-prefill kernels read and write either cache representation. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }
}
