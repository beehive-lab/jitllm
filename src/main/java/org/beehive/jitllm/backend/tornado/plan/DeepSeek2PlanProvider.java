package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.components.DeepSeek2PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code deepseek2} plan (GLM-4.7-Flash): the embedding, the latent-attention and
 * mixture-of-experts layers, and the logits. Q8_0 only — every projection is a packed-integer
 * kernel over Q8_0 blocks — in all three modes; the batched prefill runs on CUDA tensor cores.
 */
public final class DeepSeek2PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("deepseek2");

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
        DeepSeek2State typed = PlanStates.expect(DeepSeek2State.class, state, ID);
        return new DeepSeek2PlanComponents(typed, model);
    }

    /** A stage's latent cache holds its own layers; the layers address it from the first. */
    @Override
    public State stageState(
            Model model, State session, int firstLayer, int endLayer, int prefillBatchSize) {
        DeepSeek2State stage =
                (DeepSeek2State)
                        TornadoPlanProvider.super.stageState(
                                model, session, firstLayer, endLayer, prefillBatchSize);
        stage.cacheFirstLayer = firstLayer;
        return stage;
    }

    /** The batched prefill runs its projections and attention on CUDA tensor cores. */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.CUDA.equals(c.backend()) && c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the deepseek2 batched prefill runs its projections and"
                                + " attention on CUDA tensor cores, which this device"
                                + " does not have");
    }

    /** Every deepseek2 kernel is CUDA's: they reduce with warp shuffles, in every mode. */
    @Override
    public Optional<String> deviceUnsupported(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                ? Optional.empty()
                : Optional.of("the deepseek2 layers are written for the CUDA backend");
    }

    /** Every stage's layers are built from a range, single-token and batched alike. */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.empty();
    }

    /**
     * The latent cache is written and read in FP16 by the decode and batched-prefill kernels; the
     * backend is {@link #deviceUnsupported}'s to refuse.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        if (c.weights() != DataType.Q8_0) {
            return Optional.of("the deepseek2 " + c.weights() + " layers keep an FP32 cache");
        }
        return Optional.empty();
    }
}
