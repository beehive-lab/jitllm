package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.Qwen35MoePlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Qwen35MoeState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code qwen35moe} plan components: the {@code qwen35} layer graphs, whose feed-forward is the
 * routed and shared experts. Q8_0 only — the expert kernels read Q8_0 blocks.
 */
public final class Qwen35MoePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen35moe");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q8_0);
    }

    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(DataType.F32, DataType.F16, DataType.Q8_0);
    }

    /** The trunk projections, which load as Qwen3.5's do ({@code Qwen35MoeModelLoader}). */
    @Override
    public Set<DataType> packedProjectionTypes() {
        return Set.of(DataType.Q8_0, DataType.Q4_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen35MoeState typed = PlanStates.expect(Qwen35MoeState.class, state, ID);
        return new Qwen35MoePlanComponents(typed, model);
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

    /** The qwen35 layers' batched prefill, with the experts in place of the dense feed-forward. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return Qwen35PlanProvider.batchPrefillOn(c);
    }

    /** The qwen35 layers' FP16 key/value cache. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.empty();
    }
}
