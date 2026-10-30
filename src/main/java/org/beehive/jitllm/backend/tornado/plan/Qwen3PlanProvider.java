package org.beehive.jitllm.backend.tornado.plan;

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

    /** The FP16 and Q8_0 logits sample on the device; a Q4_0 file's logits are the Q8_0 layer. */
    @Override
    public boolean deviceSampling(DataType weights) {
        return weights == DataType.F16 || weights == DataType.Q8_0 || weights == DataType.Q4_0;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
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

    /** The NVIDIA-class decode layers, without a Q4_0 path. */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.nvidiaDecodeLayers(c, false);
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalDecodeLayers(c, false);
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return weights == DataType.F16 || weights == DataType.Q8_0;
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
        return BackendId.CUDA.equals(backend) ? Set.of(DataType.F16, DataType.Q8_0) : Set.of();
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
        if (weights == DataType.Q8_0) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill
                            .Qwen3Q8_0LayersBatchPrefillMMA.workspace(
                            qwen3, batchSize, policy, fp16Cache));
        }
        return Optional.empty();
    }
}
