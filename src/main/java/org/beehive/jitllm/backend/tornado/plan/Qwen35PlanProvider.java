package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
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
 * <p>The two declarations below answer different questions, and for this family they differ.
 *
 * <ul>
 *   <li>{@link #supportedDataTypes()} is admission: the representation the model <b>reports</b>,
 *       which for this family is the one its trunk projections share. Q4_0 is what the 27B holds,
 *       and it is what the loader insists the projections agree on.
 *   <li>{@link #nativeTensorTypes()} is what the layer, activation and logits graphs decode <b>per
 *       tensor</b>, without materializing anything. The 27B needs five of them at once, and the
 *       memory prediction has to count each tensor at its own size or it refuses a model that fits:
 *       14.944 GiB retained against roughly 27 GiB converted, on a 24 GiB device.
 * </ul>
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
        // Q8_0: the decode layers read it directly, and the batched prefill decodes it for cuBLAS.
        return Set.of(DataType.Q4_0, DataType.Q8_0);
    }

    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(
                DataType.F32,
                DataType.F16,
                DataType.Q4_0,
                DataType.Q4_1,
                DataType.Q4_K,
                DataType.Q5_K,
                DataType.Q6_K,
                DataType.Q8_0);
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
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill projections and attention on Apple MLX (Qwen35MlxPrefill).
            return Set.of(DataType.Q8_0, DataType.Q4_0);
        }
        return BackendId.CUDA.equals(backend) ? Set.of(DataType.Q8_0) : Set.of();
    }

    /** The MLX prefill's projections in MLX's form, its staging and its gathered cache. */
    @Override
    public Optional<org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace>
            batchPrefillWorkspace(
                    DataType weights,
                    org.beehive.jitllm.model.Configuration config,
                    int batchSize,
                    ExecutionPolicy policy,
                    boolean fp16Cache) {
        if (!(config instanceof org.beehive.jitllm.model.qwen35.Qwen35Configuration qwen35)) {
            return Optional.empty();
        }
        return org.beehive.jitllm.backend.tornado.layers.Qwen35MlxPrefill.resolve(
                        policy, weights, qwen35)
                .map(
                        m ->
                                org.beehive.jitllm.backend.tornado.layers.Qwen35MlxPrefill
                                        .workspace(weights, qwen35, batchSize));
    }
}
