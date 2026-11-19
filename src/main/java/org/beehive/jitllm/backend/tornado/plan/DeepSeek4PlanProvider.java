package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.DeepSeek4PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code deepseek4} plan (DeepSeek-V4): Q8_0 only, on CUDA, in all three modes. The batched
 * prefill's projections are int8 tensor-core GEMMs where the shapes allow. The attention window and
 * the compressed keys are FP32. The trunk splits by layers across devices ({@code --devices}),
 * single-token and batched alike, with the residual streams crossing between stages.
 */
public final class DeepSeek4PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("deepseek4");

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
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        DeepSeek4State typed = PlanStates.expect(DeepSeek4State.class, state, ID);
        return new DeepSeek4PlanComponents(typed, model);
    }

    /** The batched prefill's projections run on CUDA int8 tensor cores. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.CUDA.equals(c.backend()) && c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the deepseek4 batched prefill runs its projections on CUDA int8 tensor"
                                + " cores, which this device does not have");
    }

    /** The reductions use warp shuffles, which are CUDA's here. */
    @Override
    public Optional<String> deviceUnsupported(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                ? Optional.empty()
                : Optional.of("the deepseek4 layers are written for the CUDA backend");
    }

    /**
     * The layers address the window, the compressors' entries and the compressed keys by absolute
     * layer, so each stage keeps the whole model's layout and uses its own layers' part.
     */
    @Override
    public boolean stageCacheHoldsOnlyItsLayers() {
        return false;
    }

    /**
     * A stage's state, which reads the hash-routed experts the host stages in the session's: the
     * token loop stages them there, once for every stage.
     */
    @Override
    public State stageState(
            Model model, State session, int firstLayer, int endLayer, int prefillBatchSize) {
        State stage =
                TornadoPlanProvider.super.stageState(
                        model, session, firstLayer, endLayer, prefillBatchSize);
        PlanStates.expect(DeepSeek4State.class, stage, ID)
                .readHashExpertsFrom(PlanStates.expect(DeepSeek4State.class, session, ID));
        return stage;
    }

    /**
     * Every stage's layers are built from a range, single-token and batched alike; the residual
     * streams cross between stages ({@code DeepSeek4PlanComponents.stageCarrier}).
     */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.empty();
    }

    /** The layers bind their weights directly, on CUDA only ({@link #deviceUnsupported}). */
    @Override
    public boolean unifiedMemoryWeights() {
        return false;
    }
}
