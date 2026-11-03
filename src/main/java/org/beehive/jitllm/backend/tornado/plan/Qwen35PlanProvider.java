package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.Qwen35PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * The {@code qwen35} plan components — single-token decode, and a mixed model underneath.
 *
 * <p>{@link #supportedDataTypes()} is admission: the representation the model <b>reports</b> for
 * its trunk projections. The layer, activation and logits graphs read each tensor <b>by its own
 * type</b>, as the file stores it, so a K-quant mix (Q4_K_M) runs as a uniform file does.
 *
 * <p>All three modes. Sequential prefill is the same layer computation with the logits graph
 * skipped; batched prefill has its own layer graphs, because a chunk of prompt tokens is not a
 * token and this family's recurrence has to be scanned in order inside the kernel.
 */
// @formatter:on
public final class Qwen35PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen35");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.F16_AND_NATIVE_QUANTIZED;
    }

    /** Every qwen35 path reads its Q8_0 and Q4_0 projections prepacked. */
    @Override
    public Set<DataType> prepackableTypes(ExecutionPolicy policy) {
        return Set.of(DataType.Q8_0, DataType.Q4_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen35State typed = PlanStates.expect(Qwen35State.class, state, ID);
        return new Qwen35PlanComponents(typed, model);
    }

    /** The layers index the caches and recurrent state by absolute layer. */
    @Override
    public boolean stageCacheHoldsOnlyItsLayers() {
        return false;
    }

    /** Every stage's layers are built from a range, single-token and batched alike. */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.empty();
    }

    // @formatter:off
    /**
     * Refused on OpenCL: the scalar batched projections ({@code matrixVectorTiledBatchQ4_0} and its
     * Q4_1/Q5_K siblings) do not compile on TornadoVM's OpenCL backend. A guard the compiler can
     * prove true after the kernel's early return becomes a {@code LogicConstantNode} once a
     * constant-trip loop is unrolled, and the OpenCL LIR builder has no rule for it ({@code
     * OCLNodeLIRBuilder.emitLogicNode}). That is a compiler defect, reproduced by a ten-line
     * kernel, and it is refused here until TornadoVM fixes it rather than worked around in kernels
     * CUDA shares.
     */
    // @formatter:on
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return batchPrefillOn(c);
    }

    static Optional<String> batchPrefillOn(Combination c) {
        return BackendId.OPENCL.equals(c.backend())
                ? Optional.of(
                        "the qwen35 batched prefill projections do not compile on the"
                                + " OpenCL backend (TornadoVM OpenCL code generation"
                                + " fails with 'logic node (LogicConstantNode)')")
                : Optional.empty();
    }

    /**
     * FP16 writers and readers in every mode, including batched prefill. On CUDA the split-KV and
     * tensor-core attention read it; on OpenCL (an NVIDIA-class device) the single-workgroup FP16
     * kernel does.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return true;
    }

    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        return BackendId.CUDA.equals(backend) ? Set.of(DataType.Q8_0) : Set.of();
    }
}
