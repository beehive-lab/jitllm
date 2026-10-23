package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.Glm5NextPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code glm5next} plan (GLM-5.3-Flash): token-by-token decode, Q8_0 only, on CUDA. There is no
 * batched prefill and no split across devices yet, and the latent cache is FP32.
 */
public final class Glm5NextPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("glm5next");

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
        return Set.of(DataType.F32, DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.STANDARD, ExecutionMode.PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Glm5NextState typed = PlanStates.expect(Glm5NextState.class, state, ID);
        return new Glm5NextPlanComponents(typed, model);
    }

    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return Optional.of("glm5next has no batched prefill yet; it prefills token by token");
    }

    /** The reductions use warp shuffles, which are CUDA's here. */
    @Override
    public Optional<String> deviceUnsupported(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                ? Optional.empty()
                : Optional.of("the glm5next layers are written for the CUDA backend");
    }

    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return stages > 1
                ? Optional.of("glm5next builds its whole trunk as one graph on one device")
                : Optional.empty();
    }

    @Override
    public Optional<String> packedWeightsUnsupported(ExecutionPolicy policy) {
        return Optional.of("glm5next reads its projections unpacked");
    }
}
