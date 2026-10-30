package org.beehive.jitllm.backend.tornado.plan;

import static org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets.NATIVE_QUANTIZED;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.MistralFP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.MistralQ8_0PlanComponents;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Mistral's plan components. It shares {@code LlamaState} and has its own components, which is why
 * the alias needs its own provider rather than a reference to Llama's.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families.
 */
public final class MistralPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("mistral");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    /**
     * The FP16 and Q8_0 logits sample on the device, and so do every quantized file's: they are the
     * Q8_0 logits layer, whichever representation the output projection holds.
     */
    @Override
    public boolean deviceSampling(DataType weights) {
        return weights == DataType.F16 || NATIVE_QUANTIZED.contains(weights);
    }

    /** F16, and every quantized representation a kept load holds. */
    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.F16_AND_NATIVE_QUANTIZED;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_AND_BATCH;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        LlamaState typed = PlanStates.expect(LlamaState.class, state, ID);
        return weights == DataType.F16
                ? new MistralFP16PlanComponents(typed, model)
                : new MistralQ8_0PlanComponents(typed, model);
    }

    /**
     * Mistral keeps the non-NVIDIA scheduler for its FP32 attention; its FP16 cache takes the flash
     * kernels, which need an NVIDIA-class device, not that scheduler. A file that runs the Q8_0
     * layers one weight at a time is answered as Q8_0.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination q) {
        Combination c = q.asQ8_0Layers();
        if (!c.nvidiaDevice()) {
            return Optional.of("the FP16 flash kernels need an NVIDIA-class device");
        }
        if (c.weights() != DataType.F16 && c.weights() != DataType.Q8_0) {
            return Optional.of("the " + c.weights() + " layers keep an FP32 cache");
        }
        // Batched plans run Llama's layers, prefill and decode, which write the FP16 cache too.
        return c.mode() == ExecutionMode.PREFILL_DECODE
                ? Optional.of("mistral's sequential prefill/decode plan keeps an FP32 cache")
                : Optional.empty();
    }

    @Override
    public Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Fp16KeyValueSupport.metalSingleTokenAndBatchedLayers(c.asQ8_0Layers());
    }

    @Override
    public boolean metalBatchedPrefillDefault(DataType weights) {
        return weights == DataType.F16 || NATIVE_QUANTIZED.contains(weights);
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
                        .map(mlx -> MlxPrefill.workspace(weights, config, batchSize, false, false))
                : Optional.empty();
    }
}
