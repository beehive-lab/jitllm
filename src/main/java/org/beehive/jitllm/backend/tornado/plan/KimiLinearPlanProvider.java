package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.KimiLinearPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.KimiLinearState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code kimi-linear} plan (Kimi-Linear-48B-A3B): Q8_0 projections, the routed experts in Q8_0
 * or Q4_0, on CUDA, one token at a time or as a sequential prefill followed by decode, split across
 * devices when asked.
 */
public final class KimiLinearPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("kimi-linear");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        KimiLinearState typed = PlanStates.expect(KimiLinearState.class, state, ID);
        return new KimiLinearPlanComponents(typed, model);
    }

    /** The batched prefill runs its projections and attention on CUDA tensor cores. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.CUDA.equals(c.backend()) && c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the kimi-linear batched prefill runs its projections and attention on"
                                + " CUDA tensor cores, which this device does not have");
    }

    /** The reductions use warp shuffles, which are CUDA's here. */
    @Override
    public Optional<String> deviceUnsupported(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                ? Optional.empty()
                : Optional.of("the kimi-linear layers are written for the CUDA backend");
    }

    /**
     * The layers address the latent cache and the KDA state by absolute layer, so each stage keeps
     * the whole model's layout and uses its own layers' part.
     */
    @Override
    public boolean stageCacheHoldsOnlyItsLayers() {
        return false;
    }

    /** Every stage's single-token layers are built from a range. */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.empty();
    }

    /** The latent cache is written and read in FP16 by the decode kernels as well. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return c.weights() == DataType.Q8_0
                ? Optional.empty()
                : Optional.of("the kimi-linear " + c.weights() + " layers keep an FP32 cache");
    }

    /** The layers bind their weights directly, on CUDA only ({@link #deviceUnsupported}). */
    @Override
    public boolean unifiedMemoryWeights() {
        return false;
    }
}
