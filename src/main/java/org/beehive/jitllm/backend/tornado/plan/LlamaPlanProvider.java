package org.beehive.jitllm.backend.tornado.plan;

import java.util.EnumSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.BatchDecodeLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LlamaFP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.LlamaFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.LlamaQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Llama's plan components — all three plan shapes, both representations.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families.
 */
public final class LlamaPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("llama");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    /** FP16 weights: the tensor-core decode layers, contiguous or paged. */
    @Override
    public Optional<BatchDecodeLayers> batchDecodeLayers(
            State state, Model model, BatchDecode decode) {
        if (model.weights().dataType() != DataType.F16) {
            return Optional.empty();
        }
        return Optional.of(
                new LlamaFP16LayersBatchDecodeMMA(
                        PlanStates.expect(LlamaState.class, state, ID),
                        (LlamaTornadoWeights) model.weights(),
                        (LlamaConfiguration) model.configuration(),
                        decode.batchSize(),
                        decode.decodeContext(),
                        decode.keyCache(),
                        decode.valueCache(),
                        decode.seqPositions(),
                        decode.blockTable(),
                        decode.blockSize(),
                        decode.maxBlocksPerSlot()));
    }

    /** The FP16 and Q8_0 logits sample on the device; a Q4_0 file's logits are the Q8_0 layer. */
    @Override
    public boolean deviceSampling(DataType weights) {
        return weights == DataType.F16 || weights == DataType.Q8_0 || weights == DataType.Q4_0;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        // BOTH_REPRESENTATIONS plus Q4_0, which this family retains rather than materializing.
        // Q4_0 is single-token only; the registry refuses the other modes by name.
        return java.util.Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0);
    }

    /** Its Q4_0 layers read Q4_0 projections prepacked. */
    @Override
    public Set<DataType> prepackableTypes(ExecutionPolicy policy) {
        return Set.of(DataType.Q4_0);
    }

    /**
     * Q4_0 is kept only when {@link #retainsQ4_0} says so; otherwise every tensor is materialized
     * as Q8_0, as the loader does.
     *
     * <p>When Q4_0 is kept, the loader also keeps a Q4_1 {@code ffn_down} and a Q6_K output
     * projection as they are, which this still predicts at Q8_0: an over-estimate, the safe
     * direction.
     */
    @Override
    public DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        Set<DataType> retained = EnumSet.copyOf(nativeTensorTypes());
        if (!retainsQ4_0(sourceTypes, layers, backend)) {
            retained.remove(DataType.Q4_0);
        }
        return DeviceRetention.retaining(retained);
    }

    /**
     * Whether a Q4_0 file's weights are kept as they are, rather than materialized as Q8_0.
     *
     * <p>On by default — retaining is the point. It is switchable so the two can be measured
     * against each other on <b>one file</b>, which is the only comparison that isolates the
     * representation: comparing a Q4_0 model against a separately quantized Q8_0 one measures the
     * quantization as well. It is also the fallback if a device turns out to miscompile the Q4_0
     * kernels, the way the warp path did on OpenCL.
     */
    private static boolean retainQ4_0Enabled() {
        return !"false".equalsIgnoreCase(System.getProperty("jitllm.q4_0.retain", "true"));
    }

    private static final String[] PROJECTIONS = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"
    };

    /**
     * Whether a Llama load keeps its Q4_0 projections as they are: {@code jitllm.q4_0.retain} is
     * not {@code false}, and every per-layer weight the Q4_0 layer graph reads is Q4_0, except
     * {@code ffn_down}, which may also be Q4_1. The one rule, asked by the loader and by the memory
     * preflight, so the two cannot disagree about what a tensor becomes.
     *
     * <p>The norms are F32 and are read by dtype-independent kernels, so they are not consulted;
     * these seven are the ones a Q4_0 kernel would decode. {@code llama-quantize ... Q4_0} gives
     * some layers' {@code ffn_down} more bits as Q4_1 (10 of 80 in Llama-3.3-70B, 2 of 16 in
     * Llama-3.2-1B), and the layer graph has a Q4_1 kernel for exactly that projection.
     *
     * <p>Not on Metal: there a Q4_0 model's prompt is read by the SIMD-group batched prefill, which
     * has Q4_0 kernels only, so a file with Q4_1 layers is materialized as Q8_0 instead.
     *
     * @param sourceTypes each tensor's representation in the file, by tensor name
     * @param layers the number of transformer blocks
     * @param backend the device's backend
     */
    public static boolean retainsQ4_0(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        if (!retainQ4_0Enabled() || layers <= 0) {
            return false;
        }
        boolean q4_1DownAllowed = !BackendId.METAL.equals(backend);
        for (int layer = 0; layer < layers; layer++) {
            for (String kind : PROJECTIONS) {
                DataType type = sourceTypes.get("blk." + layer + "." + kind + ".weight");
                boolean q4_1Down =
                        q4_1DownAllowed && kind.equals("ffn_down") && type == DataType.Q4_1;
                if (type != DataType.Q4_0 && !q4_1Down) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.EVERY_MODE;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        LlamaState typed = PlanStates.expect(LlamaState.class, state, ID);
        if (weights == DataType.F16) {
            return new LlamaFP16PlanComponents(typed, model);
        }
        // Q4_0 reaches here as itself rather than as a Q8_0 materialization: the loader retained it
        // because every per-layer weight in the file is Q4_0 and these layers have kernels for it.
        if (weights == DataType.Q4_0) {
            return new org.beehive.jitllm.backend.tornado.plan.components.q4_0
                    .LlamaQ4_0PlanComponents(typed, model);
        }
        return new LlamaQ8_0PlanComponents(typed, model);
    }

    /**
     * Llama Q4_0's batched prefill runs its projections in cuBLAS, through the JIT tensor-core GEMM
     * on FP16-decoded weights, or on the Metal SIMD-group kernels: elsewhere there is nothing to
     * build.
     */
    @Override
    public Optional<String> batchPrefillNeeds(Combination c, ExecutionPolicy policy) {
        if (c.weights() == DataType.Q4_0
                && !BackendId.METAL.equals(c.backend())
                && !c.tensorCores()
                && !NativePrefillSupport.nativeProjections(policy)) {
            return Optional.of(
                    "the llama Q4_0 batched prefill needs tensor cores, or cuBLAS through"
                            + " --with-native-libraries");
        }
        return Optional.empty();
    }

    /**
     * F16 has no stage layers. Q8_0 splits its single-token plan only. Q4_0 splits its batched
     * prefill too, but a stage holding part of the model runs its projections through cuBLAS: only
     * the whole model's layers decode them for the JIT tensor-core GEMM.
     */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        boolean batched = ExecutionMode.of(policy) == ExecutionMode.BATCH_PREFILL_DECODE;
        if (weights == DataType.F16) {
            return TornadoPlanProvider.super.splitUnsupported(weights, policy, stages);
        }
        if (weights == DataType.Q4_0) {
            return batched && stages > 1 && !NativePrefillSupport.nativeProjections(policy)
                    ? Optional.of(
                            "the llama Q4_0 batched prefill of a pipeline stage runs its"
                                    + " projections through cuBLAS: add --with-native-libraries")
                    : Optional.empty();
        }
        return batched
                ? Optional.of(
                        "the llama "
                                + weights
                                + " batched prefill is not split across devices; drop"
                                + " --batch-prefill-size to split its single-token plan")
                : Optional.empty();
    }

    /** The NVIDIA-class decode layers, whose Q4_0 path writes FP16 too. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.nvidiaDecodeLayers(c, true);
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalDecodeLayers(c, true);
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return weights == DataType.F16 || weights == DataType.Q8_0 || weights == DataType.Q4_0;
    }

    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill attention and projections on Apple MLX (MlxPrefill).
            return Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0);
        }
        return BackendId.CUDA.equals(backend)
                ? Set.of(DataType.Q4_0, DataType.Q8_0, DataType.F16)
                : Set.of();
    }

    /** The tensor-core layers' attention, split-K and decoded-projection buffers. */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        var mlx =
                MlxPrefill.resolve(policy, weights, config)
                        .map(m -> MlxPrefill.workspace(weights, config, batchSize, false, false));
        if (mlx.isPresent()) {
            return mlx;
        }
        if (!TensorCoreSupport.isTensorCoreCapableBackend()) {
            return Optional.empty();
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
