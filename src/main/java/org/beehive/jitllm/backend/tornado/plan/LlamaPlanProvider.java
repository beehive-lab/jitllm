package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.LlamaFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.LlamaQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Llama's plan components — all three plan shapes, both representations.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families.
 */
public final class LlamaPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("llama");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        // BOTH_REPRESENTATIONS plus Q4_0, which this family retains rather than materializing.
        // Q4_0 is single-token only; the registry refuses the other modes by name.
        return java.util.Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.EVERY_MODE;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        LlamaState typed = PlanStates.expect(LlamaState.class, state, ID);
        if (weights == DataType.F16) {
            return new LlamaFP16PlanComponents(typed, model);
        }
        // Q4_0 reaches here as itself rather than as a Q8_0 materialization: the loader retained it
        // because every per-layer weight in the file is Q4_0 and these layers have kernels for it.
        if (weights == DataType.Q4_0) {
            return new org.beehive.jitllm.backend.tornado.plan.components.q4_0
                    .LlamaQ4_0PlanComponents(typed, model);
        }
        return new LlamaQ8_0PlanComponents(typed, model);
    }
}
