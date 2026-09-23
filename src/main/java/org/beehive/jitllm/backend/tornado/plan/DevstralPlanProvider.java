package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.DevstralFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.DevstralQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Devstral's plan components. **No fixture exists on this machine**, so this migration is
 * structural only: the components it builds are the ones the factory built, unchanged and unproven
 * by a real run.
 */
public final class DevstralPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("devstral");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        // BOTH_REPRESENTATIONS plus Q4_K, which this family retains rather than materializing.
        return java.util.Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_K);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_ONLY;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        DevstralState typed = PlanStates.expect(DevstralState.class, state, ID);
        if (weights == DataType.F16) {
            return new DevstralFP16PlanComponents(typed, model);
        }
        // Q4_K reaches here as itself rather than as a Q8_0 materialization: Devstral is the family
        // that has Q4_K kernels.
        if (weights == DataType.Q4_K) {
            return new org.beehive.jitllm.backend.tornado.plan.components.q4_k
                    .DevstralQ4_KPlanComponents(typed, model);
        }
        return new DevstralQ8_0PlanComponents(typed, model);
    }
}
