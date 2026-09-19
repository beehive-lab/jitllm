package org.beehive.jllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.fp16.Gemma4FP16PlanComponents;
import org.beehive.jllm.backend.tornado.plan.components.q8_0.Gemma4Q8_0PlanComponents;
import org.beehive.jllm.inference.state.Gemma4State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Gemma4's plan components. No program description exists for it yet, which is a separate fact: the
 * legacy plan is what it has always run.
 */
public final class Gemma4PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("gemma4");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0);
    }

    /**
     * Every representation this family's tasks decode per tensor, which is a different question
     * from the one {@link #supportedDataTypes()} answers.
     *
     * <p>That one is admission: the single representation a model reports and a plan is selected
     * on. This one is what the memory preflight must predict against, and a Q4_0 file is mixed —
     * Q4_0 projections, Q4_1 {@code ffn_down} on its first blocks, a Q4_K {@code token_embd} that
     * is also the output projection, and F32 norms. Answering admission here would predict every
     * tensor at the model's representation and mispredict all of those.
     */
    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(
                DataType.F32,
                DataType.F16,
                DataType.BF16,
                DataType.Q4_0,
                DataType.Q4_1,
                DataType.Q4_K,
                DataType.Q8_0);
    }

    // @formatter:off
    /**
     * Single-token everywhere, and the batched prefill/decode plan as well.
     *
     * <p>The batched plan is declared for the family, not for a representation — {@code
     * supportedModes()} has no dtype to answer for. What decides per representation is whether that
     * representation's components implement the batched interface; the registry refuses the rest by
     * name. Today Q8_0's do and Q4_0's do not, because the batched projections are tensor-core
     * GEMMs over 34-byte blocks.
     */
    // @formatter:on
    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Gemma4State typed = PlanStates.expect(Gemma4State.class, state, ID);
        // Named branches, not a fallthrough: the quantized components read a tensor by its own
        // representation, and letting an unexpected dtype land on them would read one block layout
        // as another rather than fail.
        return switch (weights) {
            case F16 -> new Gemma4FP16PlanComponents(typed, model);
            case Q8_0, Q4_0 -> new Gemma4Q8_0PlanComponents(typed, model);
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 has no plan components for " + weights);
        };
    }
}
