package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.DevstralFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.DevstralQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Devstral's plan components. **No fixture exists on this machine**, so this migration is
 * structural only: the components it builds are the ones the factory built, unchanged and unproven
 * by a real run.
 */
public final class DevstralPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("devstral");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        // BOTH_REPRESENTATIONS plus Q4_K, which this family retains rather than materializing.
        return java.util.Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_K);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_AND_BATCH;
    }

    /**
     * The layers write the FP16 cache on the NVIDIA path, single-token and after the tensor-core
     * batched prefill, which writes it too.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        if (!c.nvidiaScheduler()) {
            return Optional.of("the non-NVIDIA decode layers keep an FP32 cache");
        }
        return c.mode() == ExecutionMode.PREFILL_DECODE
                ? Optional.of("devstral has no sequential prefill/decode plan")
                : Optional.empty();
    }

    /**
     * The batched prefill runs Llama's tensor-core layers: Q8_0 and F16 directly, Q4_K with each
     * projection decoded into FP16 first.
     */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        if (c.tensorCores()) {
            return Optional.empty();
        }
        if (BackendId.METAL.equals(c.backend()) && c.weights() == DataType.Q4_K) {
            // Apple MLX's GEMMs and attention (DevstralLayersBatchPrefillMlx).
            return org.beehive.jitllm.backend.tornado.MlxPrefill.available()
                    ? Optional.empty()
                    : Optional.of(
                            "the devstral batched prefill on Metal runs on Apple MLX, whose"
                                    + " mlx.metallib could not be loaded (brew install mlx)");
        }
        return Optional.of(
                "the devstral batched prefill runs on the CUDA tensor cores, and on Metal for Q4_K"
                        + " weights through Apple MLX");
    }

    /** On Metal the batched prefill exists only on MLX, so it needs native libraries asked for. */
    @Override
    public Optional<String> batchPrefillNeeds(
            Combination c, org.beehive.jitllm.runtime.policy.ExecutionPolicy policy) {
        if (BackendId.METAL.equals(c.backend()) && !policy.nativeLibraries()) {
            return Optional.of(
                    "the devstral batched prefill on Metal runs on Apple MLX; add"
                            + " --with-native-libraries");
        }
        return Optional.empty();
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        DevstralState typed = PlanStates.expect(DevstralState.class, state, ID);
        if (weights == DataType.F16) {
            return new DevstralFP16PlanComponents(typed, model);
        }
        // Q4_K reaches here as itself rather than as a Q8_0 materialization: Devstral is the family
        // that has Q4_K kernels.
        if (weights == DataType.Q4_K) {
            return new org.beehive.jitllm.backend.tornado.plan.components.q4_k
                    .DevstralQ4_KPlanComponents(typed, model);
        }
        return new DevstralQ8_0PlanComponents(typed, model);
    }

    /**
     * The batched prefill's projections through cuBLAS, quantized weights decoded to FP16 first.
     */
    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill projections and attention on Apple MLX.
            return Set.of(DataType.Q4_K);
        }
        return BackendId.CUDA.equals(backend)
                ? Set.of(DataType.Q8_0, DataType.Q4_K, DataType.F16)
                : Set.of();
    }

    /** Llama's tensor-core layers' attention, split-K and decoded-projection buffers. */
    @Override
    public Optional<org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace>
            batchPrefillWorkspace(
                    DataType weights,
                    org.beehive.jitllm.model.Configuration config,
                    int batchSize,
                    org.beehive.jitllm.runtime.policy.ExecutionPolicy policy,
                    boolean fp16Cache) {
        var mlx =
                org.beehive.jitllm.backend.tornado.MlxPrefill.resolve(policy, weights, config)
                        .map(
                                m ->
                                        org.beehive.jitllm.backend.tornado.MlxPrefill.workspace(
                                                weights, config, batchSize, false, false));
        if (mlx.isPresent()) {
            return mlx;
        }
        if (weights == DataType.F16) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill
                            .LlamaFP16LayersBatchPrefillMMA.workspace(
                            config, batchSize, policy, fp16Cache));
        }
        if (weights == DataType.Q8_0) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill
                            .LlamaQ8_0LayersBatchPrefillMMA.workspace(
                            config, batchSize, policy, fp16Cache));
        }
        return Optional.empty();
    }
}
