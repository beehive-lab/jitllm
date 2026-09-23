package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.components.Qwen35PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * The {@code qwen35} plan components — single-token decode, and a mixed model underneath.
 *
 * <p>The two declarations below answer different questions, and for this family they differ.
 *
 * <ul>
 *   <li>{@link #supportedDataTypes()} is admission: the representation the model <b>reports</b>,
 *       which for this family is the one its trunk projections share. Q4_0 is what the 27B holds,
 *       and it is what the loader insists the projections agree on.
 *   <li>{@link #nativeTensorTypes()} is what the layer, activation and logits graphs decode <b>per
 *       tensor</b>, without materializing anything. The 27B needs five of them at once, and the
 *       memory prediction has to count each tensor at its own size or it refuses a model that fits:
 *       14.944 GiB retained against roughly 27 GiB converted, on a 24 GiB device.
 * </ul>
 *
 * <p>All three modes. Sequential prefill is the same layer computation with the logits graph
 * skipped; batched prefill has its own layer graphs, because a chunk of prompt tokens is not a
 * token and this family's recurrence has to be scanned in order inside the kernel.
 */
// @formatter:on
public final class Qwen35PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen35");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q4_0);
    }

    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(
                DataType.F32,
                DataType.F16,
                DataType.Q4_0,
                DataType.Q4_1,
                DataType.Q4_K,
                DataType.Q5_K,
                DataType.Q6_K,
                DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen35State typed = PlanStates.expect(Qwen35State.class, state, ID);
        return new Qwen35PlanComponents(typed, model);
    }
}
