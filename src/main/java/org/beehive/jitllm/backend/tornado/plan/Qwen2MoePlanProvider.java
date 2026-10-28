package org.beehive.jitllm.backend.tornado.plan;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Qwen2MoEQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Qwen2-MoE's plan components — the family the inventory flagged.
 *
 * <p><b>Quantized files only, and two of the three modes.</b> It is the shape that made the
 * registry's protocol matter: a registered provider that refuses {@code F16} must produce the named
 * error, not an empty result, or a deliberate refusal would be indistinguishable from a family
 * nobody had migrated — and the factory switch would then answer for a family that had already
 * moved.
 *
 * <p>{@code PREFILL_DECODE} is likewise a refusal, matching what the factory threw.
 */
public final class Qwen2MoePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen2-moe");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    /** Every quantized representation a kept load holds. */
    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.NATIVE_QUANTIZED;
    }

    /** What a load keeps whatever the backend: F16 and Q8_0. The rest depends on the backend. */
    @Override
    public Set<DataType> nativeTensorTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    /**
     * Every quantized type kept exactly where the loader keeps it: {@link
     * TornadoPlanProvider#retainsNative}.
     */
    @Override
    public DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        return TornadoPlanProvider.nativeRetention(nativeTensorTypes(), backend);
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

    /**
     * The NVIDIA-class layers write the FP16 cache single-token, and after the tensor-core batched
     * prefill, which writes it too; the scalar batched prefill keeps FP32. A kept file answers as
     * the Q8_0 layers it runs.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination combination) {
        Combination c = combination.asQ8_0Layers();
        if (!c.nvidiaScheduler()) {
            return Optional.of("the non-NVIDIA decode layers keep an FP32 cache");
        }
        if (c.weights() != DataType.Q8_0) {
            return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
        }
        return switch (c.mode()) {
            case STANDARD -> Optional.empty();
            case BATCH_PREFILL_DECODE ->
                    c.tensorCores()
                            ? Optional.empty()
                            : Optional.of("the scalar batched prefill keeps an FP32 cache");
            case PREFILL_DECODE -> Optional.of("qwen2-moe has no sequential prefill/decode plan");
        };
    }
}
