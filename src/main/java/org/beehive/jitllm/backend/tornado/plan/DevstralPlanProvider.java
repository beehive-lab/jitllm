package org.beehive.jitllm.backend.tornado.plan;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.DevstralFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.DevstralQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.DevstralState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
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

    /**
     * F16, Q8_0 and Q4_K as before (Q4_K for a file whose projections are all K-quants, {@link
     * #kQuantLayers}), and the types a kept file reports when it runs the Q8_0 layers one weight at
     * a time.
     */
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
     * Every quantized type kept where the loader keeps it ({@link
     * TornadoPlanProvider#retainsNative}). Elsewhere a {@link #kQuantLayers} file keeps its Q4_K
     * and Q6_K projections and converts its embedding and output projection, and any other file is
     * converted to Q8_0, as the loader does.
     */
    @Override
    public DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        if (TornadoPlanProvider.retainsNative(backend)) {
            return TornadoPlanProvider.nativeRetention(nativeTensorTypes(), backend);
        }
        DeviceRetention converted = DeviceRetention.retaining(nativeTensorTypes());
        if (!kQuantLayers(sourceTypes, layers)) {
            return converted;
        }
        return (name, source) ->
                name.startsWith("blk.") && (source == DataType.Q4_K || source == DataType.Q6_K)
                        ? source
                        : converted.deviceType(name, source);
    }

    private static final String[] PROJECTIONS = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
    };

    /**
     * Whether a Devstral load runs the K-quant layers: every per-layer weight they read is Q4_K or
     * Q6_K, as in a Q4_K_M file. Asked by the loader and by the memory preflight, so the two cannot
     * disagree about what a tensor becomes. The norms are F32 and read by dtype-independent
     * kernels.
     */
    public static boolean kQuantLayers(Map<String, DataType> sourceTypes, int layers) {
        if (layers <= 0) {
            return false;
        }
        for (int i = 0; i < layers; i++) {
            for (String kind : PROJECTIONS) {
                DataType type = sourceTypes.get("blk." + i + "." + kind + ".weight");
                if (type != DataType.Q4_K && type != DataType.Q6_K) {
                    return false;
                }
            }
        }
        return true;
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
        return c.tensorCores()
                ? Optional.empty()
                : Optional.of("the devstral batched prefill runs on the CUDA tensor cores");
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
        return BackendId.CUDA.equals(backend)
                ? TornadoSupportSets.F16_AND_NATIVE_QUANTIZED
                : Set.of();
    }

    /** Llama's tensor-core layers' attention, split-K and decoded-projection buffers. */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        if (weights == DataType.F16) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill
                            .LlamaFP16LayersBatchPrefillMMA.workspace(
                            config, batchSize, policy, fp16Cache));
        }
        if (TornadoSupportSets.NATIVE_QUANTIZED.contains(weights) && weights != DataType.Q4_K) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill
                            .LlamaQ8_0LayersBatchPrefillMMA.workspace(
                            config, batchSize, policy, fp16Cache));
        }
        return Optional.empty();
    }
}
