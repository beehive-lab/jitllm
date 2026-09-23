package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.MistralFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.MistralQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Mistral's plan components. It shares {@code LlamaState} and has its own components, which is why
 * the alias needs its own provider rather than a reference to Llama's.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families.
 */
public final class MistralPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("mistral");

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
        LlamaState typed = PlanStates.expect(LlamaState.class, state, ID);
        return weights == DataType.F16
                ? new MistralFP16PlanComponents(typed, model)
                : new MistralQ8_0PlanComponents(typed, model);
    }
}
