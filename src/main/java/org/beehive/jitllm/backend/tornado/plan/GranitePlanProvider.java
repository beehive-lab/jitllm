package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.GraniteFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.GraniteQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Granite's plan components. */
public final class GranitePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("granite");

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
        GraniteState typed = PlanStates.expect(GraniteState.class, state, ID);
        return weights == DataType.F16
                ? new GraniteFP16PlanComponents(typed, model)
                : new GraniteQ8_0PlanComponents(typed, model);
    }
}
