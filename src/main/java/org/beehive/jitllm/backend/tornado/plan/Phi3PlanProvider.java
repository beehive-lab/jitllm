package org.beehive.jitllm.backend.tornado.plan;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.Phi3FP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Phi3Q8_0PlanComponents;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Phi3's plan components. */
public final class Phi3PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("phi3");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    /** F16, and every quantized representation a kept load holds. */
    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.F16_AND_NATIVE_QUANTIZED;
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
        return TornadoSupportSets.STANDARD_AND_BATCH;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Phi3State typed = PlanStates.expect(Phi3State.class, state, ID);
        return weights == DataType.F16
                ? new Phi3FP16PlanComponents(typed, model)
                : new Phi3Q8_0PlanComponents(typed, model);
    }

    /** Single-token and batched-prefill plans, on the NVIDIA-class decode layers. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.singleTokenAndBatchedNvidiaLayers(c.asQ8_0Layers());
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalSingleTokenAndBatchedLayers(c.asQ8_0Layers());
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return TornadoSupportSets.F16_AND_NATIVE_QUANTIZED.contains(weights);
    }

    /**
     * The batched prefill runs the Metal SIMD-group kernels or the CUDA tensor cores, which read
     * the fused QKV and gate/up weights in one GEMM each; there is no scalar path.
     */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return BackendId.METAL.equals(c.backend()) || c.tensorCores()
                ? Optional.empty()
                : Optional.of(
                        "the phi3 batched prefill runs on the Metal SIMD-group kernels or the CUDA"
                                + " tensor cores, for its fused QKV and gate/up weights");
    }

    /**
     * The batched prefill's projections through cuBLAS: quantized weights decoded to FP16 first;
     * F16 weights' output and down projections. MLX on Metal for F16 and Q8_0, and no other
     * quantized type until part 5 (decision 2).
     */
    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill attention and projections on Apple MLX (MlxPrefill).
            return Set.of(DataType.F16, DataType.Q8_0);
        }
        return BackendId.CUDA.equals(backend)
                ? TornadoSupportSets.F16_AND_NATIVE_QUANTIZED
                : Set.of();
    }

    /** On Metal under native libraries, what the MLX prefill allocates ({@code MlxPrefill}). */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        // MLX only for the types Metal's native libraries take here (nativeLibraryWeights).
        return nativeLibraryWeights(BackendId.METAL).contains(weights)
                ? MlxPrefill.resolve(policy, weights, config)
                        .map(mlx -> MlxPrefill.workspace(weights, config, batchSize, true, true))
                : Optional.empty();
    }
}
