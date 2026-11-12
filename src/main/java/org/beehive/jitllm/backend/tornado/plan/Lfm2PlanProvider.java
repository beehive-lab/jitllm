package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.layers.Lfm2BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.Lfm2PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * The {@code lfm2moe} plan components — single-token decode over a mixed model.
 *
 * <ul>
 *   <li>{@link #supportedDataTypes()} is admission: the representation the model reports, that of
 *       its short-convolution projections, which is Q4_K in the 24B.
 *   <li>Every tensor is kept as the file stores it, and each task is selected by the representation
 *       of the tensor it reads: K-quant experts, Q8_0 attention, dense layers and vocabulary, F32
 *       norms, routers and convolution taps.
 * </ul>
 *
 * <p>All three modes. Batched prefill groups each chunk's rows by expert and runs one GEMM per
 * expert, for Q4_K, Q5_K and Q6_K experts.
 */
// @formatter:on
public final class Lfm2PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("lfm2moe");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q4_K, DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Lfm2State typed = PlanStates.expect(Lfm2State.class, state, ID);
        return new Lfm2PlanComponents(typed, model);
    }

    /**
     * The batched prefill on Metal (SIMD-group GEMMs) and CUDA (tiled batch kernels), for K-quant
     * experts. OpenCL miscompiles the shuffles its router and selection reduce with, and a Q8_0
     * file's experts are Q8_0, which has no grouped GEMM.
     */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        if (BackendId.OPENCL.equals(c.backend())) {
            return Optional.of(
                    "the lfm2moe batched prefill reduces with 32-lane shuffles, which the"
                            + " OpenCL backend miscompiles");
        }
        return c.weights() == DataType.Q8_0
                ? Optional.of(
                        "the lfm2moe batched prefill groups Q4_K, Q5_K and Q6_K experts only, and a"
                                + " Q8_0 file's experts are Q8_0")
                : Optional.empty();
    }

    /** The batched prefill is the default on Metal, where its GEMMs are tuned. */
    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return true;
    }

    /** The routing, the sorted assignments and their rows, and the convolution's chunk. */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        return config instanceof Lfm2Configuration lfm2
                ? Optional.of(
                        BatchPrefillWorkspace.withoutResidencyChoice(
                                Lfm2BatchPrefillLayers.workspaceBytes(lfm2, batchSize)))
                : Optional.empty();
    }

    /**
     * Greedy decode can take its token from the device: the logits are always {@code
     * LogitsQ8_0Layer}'s, whichever representation the tied output projection holds.
     */
    @Override
    public boolean deviceSampling(DataType weights) {
        return true;
    }

    /**
     * On Metal the layers write and read the FP16 cache in every mode, held to the host's own FP16
     * cache by {@code Lfm2Fp16KvParityAccelTest}.
     */
    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }

    /**
     * On CUDA and on OpenCL over an NVIDIA device, measured as on Metal ({@code
     * Lfm2Fp16KvParityAccelTest}).
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }
}
