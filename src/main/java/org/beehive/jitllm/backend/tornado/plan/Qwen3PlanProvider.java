package org.beehive.jitllm.backend.tornado.plan;

import java.util.EnumSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.BatchDecodeLayers;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.Qwen3FP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Qwen3Q8_0PlanComponents;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/** Qwen3's plan components — all three plan shapes. */
public final class Qwen3PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen3");

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
                new Qwen3FP16LayersBatchDecodeMMA(
                        PlanStates.expect(Qwen3State.class, state, ID),
                        (Qwen3TornadoWeights) model.weights(),
                        (Qwen3Configuration) model.configuration(),
                        decode.batchSize(),
                        decode.decodeContext(),
                        decode.keyCache(),
                        decode.valueCache(),
                        decode.seqPositions(),
                        decode.blockTable(),
                        decode.blockSize(),
                        decode.maxBlocksPerSlot()));
    }

    /**
     * The FP16 and Q8_0 logits sample on the device; a Q4_0 file's logits are the Q8_0 layer, and
     * so are a Q4_K_M file's, whichever representation its output projection holds.
     */
    @Override
    public boolean deviceSampling(DataType weights) {
        return weights == DataType.F16
                || weights == DataType.Q8_0
                || weights == DataType.Q4_0
                || weights == DataType.Q4_K;
    }

    /**
     * Whether a Qwen3 load keeps its K-quant tensors as the file stores them, rather than
     * converting them to Q8_0: the layout of a Q4_K_M file. Every quantized tensor is Q4_K or Q6_K,
     * {@code token_embd} (and {@code output}, where the file has one) is one of the two, and in
     * every layer {@code attn_q}, {@code attn_k}, {@code attn_output}, {@code ffn_gate} and {@code
     * ffn_up} are Q4_K while {@code attn_v} and {@code ffn_down} are either.
     *
     * <p>CUDA and Metal only: on any other backend the tiled batch projections these layers would
     * need are not known to compile (Qwen3.5's do not on OpenCL), so the file is converted as
     * before.
     *
     * <p>The one rule for the loader and the memory preflight, so the two cannot disagree about
     * what a tensor becomes; a model it keeps reports {@link DataType#Q4_K}, which is what
     * admission sees.
     *
     * @param sourceTypes each tensor's representation in the file, by tensor name
     * @param layers the number of transformer blocks
     * @param backend the device's backend
     */
    public static boolean retainsQ4_K(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        if (layers <= 0 || !(BackendId.CUDA.equals(backend) || BackendId.METAL.equals(backend))) {
            return false;
        }
        for (DataType type : sourceTypes.values()) {
            if (type.isQuantized() && !kQuant(type)) {
                return false;
            }
        }
        if (!kQuant(sourceTypes.get("token_embd.weight"))
                || (sourceTypes.containsKey("output.weight")
                        && !kQuant(sourceTypes.get("output.weight")))) {
            return false;
        }
        for (int layer = 0; layer < layers; layer++) {
            for (String kind : Q4_K_PROJECTIONS) {
                if (sourceTypes.get("blk." + layer + "." + kind + ".weight") != DataType.Q4_K) {
                    return false;
                }
            }
            for (String kind : K_QUANT_PROJECTIONS) {
                if (!kQuant(sourceTypes.get("blk." + layer + "." + kind + ".weight"))) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean kQuant(DataType type) {
        return type == DataType.Q4_K || type == DataType.Q6_K;
    }

    /** The projections a Q4_K_M layer holds as Q4_K. */
    private static final String[] Q4_K_PROJECTIONS = {
        "attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"
    };

    /** The projections a Q4_K_M layer holds as Q4_K or Q6_K, layer by layer. */
    private static final String[] K_QUANT_PROJECTIONS = {"attn_v", "ffn_down"};

    /** F16, Q8_0, and Q4_K for a Q4_K_M file kept as it is ({@link #retainsQ4_K}). */
    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_K);
    }

    /** What a load keeps whatever the file: F16 and Q8_0. The K-quants depend on the file. */
    @Override
    public Set<DataType> nativeTensorTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    /** Q4_K and Q6_K kept exactly when the loader keeps them: {@link #retainsQ4_K}. */
    @Override
    public DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        Set<DataType> retained = EnumSet.copyOf(nativeTensorTypes());
        if (retainsQ4_K(sourceTypes, layers, backend)) {
            retained.add(DataType.Q4_K);
            retained.add(DataType.Q6_K);
        }
        return DeviceRetention.retaining(retained);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.EVERY_MODE;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen3State typed = PlanStates.expect(Qwen3State.class, state, ID);
        return weights == DataType.F16
                ? new Qwen3FP16PlanComponents(typed, model)
                : new Qwen3Q8_0PlanComponents(typed, model);
    }

    /**
     * The NVIDIA-class decode layers, without a Q4_0 path. Q4_K runs the Q8_0 layers, which read a
     * K-quant layer's tensors and write the cache as they do Q8_0's, so it is answered as Q8_0.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.nvidiaDecodeLayers(asQ8_0Layers(c), false);
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalDecodeLayers(asQ8_0Layers(c), false);
    }

    /** {@code c} with Q4_K answered as the Q8_0 layers it runs. */
    private static Combination asQ8_0Layers(Combination c) {
        return c.weights() != DataType.Q4_K
                ? c
                : new Combination(
                        c.architecture(),
                        DataType.Q8_0,
                        c.mode(),
                        c.backend(),
                        c.nvidiaScheduler(),
                        c.tensorCores(),
                        c.nvidiaDevice());
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return weights == DataType.F16 || weights == DataType.Q8_0 || weights == DataType.Q4_K;
    }

    /**
     * F16: projections through cuBLAS, the first chunk's attention through cuDNN. Q8_0: projections
     * through cuBLAS over the weights decoded to FP16.
     */
    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            // Batched prefill attention and projections on Apple MLX (MlxPrefill).
            return Set.of(DataType.F16, DataType.Q8_0);
        }
        // Q4_K through cuBLAS over the same FP16-decoded weights; not on MLX yet.
        return BackendId.CUDA.equals(backend)
                ? Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_K)
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
        if (!TensorCoreSupport.isTensorCoreCapableBackend()
                || !(config instanceof Qwen3Configuration qwen3)) {
            return Optional.empty();
        }
        if (weights == DataType.F16) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill
                            .Qwen3FP16LayersBatchPrefillMMA.workspace(
                            qwen3, batchSize, policy, fp16Cache));
        }
        if (weights == DataType.Q8_0 || weights == DataType.Q4_K) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill
                            .Qwen3Q8_0LayersBatchPrefillMMA.workspace(
                            qwen3, batchSize, policy, fp16Cache));
        }
        return Optional.empty();
    }
}
