package org.beehive.jllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.q8_0.Qwen2MoEQ8_0PlanComponents;
import org.beehive.jllm.inference.state.Qwen2MoEState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Qwen2-MoE's plan components — the family the inventory flagged.
 *
 * <p><b>Q8_0 only, and two of the three modes.</b> It is the shape that made the registry's
 * protocol matter: a registered provider that refuses {@code F16} must produce the named error, not
 * an empty result, or a deliberate refusal would be indistinguishable from a family nobody had
 * migrated — and the factory switch would then answer for a family that had already moved.
 *
 * <p>{@code PREFILL_DECODE} is likewise a refusal, matching what the factory threw.
 */
public final class Qwen2MoePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen2-moe");

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
        return java.util.EnumSet.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen2MoEState typed = PlanStates.expect(Qwen2MoEState.class, state, ID);
        return new Qwen2MoEQ8_0PlanComponents(typed, model);
    }
}
