package org.beehive.jitllm.backend.tornado.plan;

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
