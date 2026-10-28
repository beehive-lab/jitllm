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
     * The FP16 and Q8_0 logits sample on the device, and so do every quantized file's: they are the
     * Q8_0 logits layer, whichever representation the output projection holds.
     */
    @Override
    public boolean deviceSampling(DataType weights) {
        return weights == DataType.F16 || NATIVE_QUANTIZED.contains(weights);
    }

    // @formatter:off
    /**
     * Whether a Qwen3 load keeps every tensor as the file stores it, rather than converting Q4_0,
     * Q4_1 and the K-quants to Q8_0: on CUDA and Metal, where every path reads each tensor by its
     * own type (the single-token projections, the batched prefill, the tensor-core prefill and the
     * embedding row).
     *
     * <p>On any other backend the tiled batch projections these layers would need are not known to
     * compile (Qwen3.5's do not on OpenCL), so the file is converted as before.
     *
     * <p>The one rule for the loader and the memory preflight, so the two cannot disagree about
     * what a tensor becomes. A kept model reports its trunk projection's type, which is what
     * admission sees.
     */
    // @formatter:on
    public static boolean retainsNative(BackendId backend) {
        return BackendId.CUDA.equals(backend) || BackendId.METAL.equals(backend);
    }

    /** The quantized representations a kept Qwen3 load holds, each read by its own kernels. */
    private static final Set<DataType> NATIVE_QUANTIZED =
            EnumSet.of(
                    DataType.Q8_0,
                    DataType.Q4_0,
                    DataType.Q4_1,
                    DataType.Q4_K,
                    DataType.Q5_K,
                    DataType.Q6_K);

    /** F16, and every quantized representation a kept load holds ({@link #retainsNative}). */
    @Override
    public Set<DataType> supportedDataTypes() {
        Set<DataType> supported = EnumSet.copyOf(NATIVE_QUANTIZED);
        supported.add(DataType.F16);
        return supported;
    }

    /** What a load keeps whatever the backend: F16 and Q8_0. The rest depends on the backend. */
    @Override
    public Set<DataType> nativeTensorTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    /** Every quantized type kept exactly where the loader keeps it: {@link #retainsNative}. */
    @Override
    public DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        Set<DataType> retained = EnumSet.copyOf(nativeTensorTypes());
        if (retainsNative(backend)) {
            retained.addAll(NATIVE_QUANTIZED);
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
     * The NVIDIA-class decode layers, without a Q4_0 path. A quantized file runs the Q8_0 layers,
     * which read each tensor by its type and write the cache as they do Q8_0's, so it is answered
     * as Q8_0.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.nvidiaDecodeLayers(asQ8_0Layers(c), false);
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalDecodeLayers(asQ8_0Layers(c), false);
    }

    /** {@code c} with a quantized file answered as the Q8_0 layers it runs. */
    private static Combination asQ8_0Layers(Combination c) {
        return !NATIVE_QUANTIZED.contains(c.weights()) || c.weights() == DataType.Q8_0
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
        return weights == DataType.F16 || NATIVE_QUANTIZED.contains(weights);
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
        // Every quantized type through cuBLAS over the same FP16-decoded weights; on MLX only
        // Q8_0 until part 5 (decision 2), so a Metal session with any other file runs the JIT
        // kernels.
        if (!BackendId.CUDA.equals(backend)) {
            return Set.of();
        }
        Set<DataType> weights = EnumSet.copyOf(NATIVE_QUANTIZED);
        weights.add(DataType.F16);
        return weights;
    }

    /** The tensor-core layers' attention, split-K and decoded-projection buffers. */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        // MLX only for the types Metal's native libraries take here (nativeLibraryWeights).
        var mlx =
                nativeLibraryWeights(BackendId.METAL).contains(weights)
                        ? MlxPrefill.resolve(policy, weights, config)
                                .map(
                                        m ->
                                                MlxPrefill.workspace(
                                                        weights, config, batchSize, false, false))
                        : Optional.<BatchPrefillWorkspace>empty();
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
        if (NATIVE_QUANTIZED.contains(weights)) {
            return Optional.of(
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill
                            .Qwen3Q8_0LayersBatchPrefillMMA.workspace(
                            qwen3, batchSize, policy, fp16Cache));
        }
        return Optional.empty();
    }
}
