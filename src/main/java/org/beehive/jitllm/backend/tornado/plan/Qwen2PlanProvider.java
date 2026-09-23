package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.Qwen2FP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Qwen2Q8_0PlanComponents;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Qwen2's plan components. */
public final class Qwen2PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen2");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_ONLY;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen2State typed = PlanStates.expect(Qwen2State.class, state, ID);
        return weights == DataType.F16
                ? new Qwen2FP16PlanComponents(typed, model)
                : new Qwen2Q8_0PlanComponents(typed, model);
    }
}
