package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.Qwen2FP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Qwen2Q8_0PlanComponents;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/** DeepSeek-R1-Distill-Qwen's plan components: Qwen2's, under its own identity. */
public final class DeepSeekR1DistillQwenPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("deepseek-r1-distill-qwen");

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
        return TornadoSupportSets.STANDARD_AND_BATCH;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen2State typed = PlanStates.expect(Qwen2State.class, state, ID);
        return weights == DataType.F16
                ? new Qwen2FP16PlanComponents(typed, model)
                : new Qwen2Q8_0PlanComponents(typed, model);
    }

    /** Qwen2's: single-token and batched-prefill plans, on the NVIDIA-class decode layers. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.singleTokenAndBatchedNvidiaLayers(c);
    }

    /**
     * The batched prefill's projections through cuBLAS: Q8_0 weights decoded to FP16 first; F16
     * weights' output and down projections.
     */
    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill attention and projections on Apple MLX (MlxPrefill).
            return Set.of(DataType.F16, DataType.Q8_0);
        }
        return BackendId.CUDA.equals(backend) ? Set.of(DataType.Q8_0, DataType.F16) : Set.of();
    }

    /** On Metal under native libraries, what the MLX prefill allocates ({@code MlxPrefill}). */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        return MlxPrefill.resolve(policy, weights, config)
                .map(mlx -> MlxPrefill.workspace(weights, config, batchSize, false, false));
    }
}
