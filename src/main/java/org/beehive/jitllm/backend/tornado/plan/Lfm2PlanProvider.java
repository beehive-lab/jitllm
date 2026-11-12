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
 * <p>{@code STANDARD} and sequential {@code PREFILL_DECODE}, which is the same layer graphs with
 * the logits graph skipped. Batched prefill has no graphs for this family and is not declared.
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

    /** The batched prefill's grouped expert GEMMs are SIMD-group matrices: Metal for now. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.METAL.equals(c.backend())
                ? Optional.empty()
                : Optional.of(
                        "the lfm2moe batched prefill runs its expert GEMMs on Metal SIMD-group"
                                + " matrices, which this backend does not have");
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
     * On Metal the layers write and read the FP16 cache in both modes, held to the host's own FP16
     * cache by {@code Lfm2Fp16KvParityAccelTest}. Elsewhere they keep the FP32 cache until measured
     * there.
     */
    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }
}
