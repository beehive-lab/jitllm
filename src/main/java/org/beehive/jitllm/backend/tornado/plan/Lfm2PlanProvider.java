package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.components.Lfm2PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * The {@code lfm2moe} plan components — single-token decode over a mixed model.
 *
 * <ul>
 *   <li>{@link #supportedDataTypes()} is admission: the representation the model reports, that of
 *       its short-convolution projections, which is Q4_K in the 24B.
 *   <li>{@link #nativeTensorTypes()} is what the graphs decode per tensor without materializing:
 *       K-quant experts, Q8_0 attention, dense layers and vocabulary, F32 norms, routers and
 *       convolution taps.
 * </ul>
 *
 * <p>{@code STANDARD} and sequential {@code PREFILL_DECODE}, which is the same layer graphs with
 * the logits graph skipped. Batched prefill has no graphs for this family and is not declared.
 */
// @formatter:on
public final class Lfm2PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("lfm2moe");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q4_K, DataType.Q8_0);
    }

    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(DataType.F32, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K, DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.STANDARD, ExecutionMode.PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Lfm2State typed = PlanStates.expect(Lfm2State.class, state, ID);
        return new Lfm2PlanComponents(typed, model);
    }
}
