package org.beehive.jllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.fp16.Phi3FP16PlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.q8_0.Phi3Q8_0PlanComponents;
import org.beehive.jllm.inference.state.Phi3State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/** Phi3's plan components. */
public final class Phi3PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("phi3");

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
        Phi3State typed = PlanStates.expect(Phi3State.class, state, ID);
        return weights == DataType.F16
                ? new Phi3FP16PlanComponents(typed, model)
                : new Phi3Q8_0PlanComponents(typed, model);
    }
}
