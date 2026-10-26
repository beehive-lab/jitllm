package org.beehive.jitllm.backend.tornado.plan;

import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.layers.Gemma4BatchPrefillLayers;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.fp16.Gemma4FP16PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.q8_0.Gemma4Q8_0PlanComponents;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.gemma4.Gemma4Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Gemma4's plan components. No program description exists for it yet, which is a separate fact: the
 * legacy plan is what it has always run.
 */
public final class Gemma4PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("gemma4");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0);
    }

    /**
     * Every representation this family's tasks decode per tensor, which is a different question
     * from the one {@link #supportedDataTypes()} answers.
     *
     * <p>That one is admission: the single representation a model reports and a plan is selected
     * on. This one is what the memory preflight must predict against, and a Q4_0 file is mixed —
     * Q4_0 projections, Q4_1 {@code ffn_down} on its first blocks, a Q4_K {@code token_embd} that
     * is also the output projection, and F32 norms. Answering admission here would predict every
     * tensor at the model's representation and mispredict all of those.
     */
    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(
                DataType.F32,
                DataType.F16,
                DataType.BF16,
                DataType.Q4_0,
                DataType.Q4_1,
                DataType.Q4_K,
                DataType.Q8_0);
    }

    // @formatter:off
    /**
     * Single-token everywhere, and the batched prefill/decode plan as well.
     *
     * <p>The batched plan is declared for the family, not for a representation — {@code
     * supportedModes()} has no dtype to answer for. What decides per representation is whether that
     * representation's components implement the batched interface; the registry refuses the rest by
     * name. Q8_0 and Q4_0 both route to {@code Gemma4Q8_0PlanComponents}, which does.
     */
    // @formatter:on
    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Gemma4State typed = PlanStates.expect(Gemma4State.class, state, ID);
        // Named branches, not a fallthrough: the quantized components read a tensor by its own
        // representation, and letting an unexpected dtype land on them would read one block layout
        // as another rather than fail.
        return switch (weights) {
            case F16 -> new Gemma4FP16PlanComponents(typed, model);
            case Q8_0, Q4_0 -> new Gemma4Q8_0PlanComponents(typed, model);
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 has no plan components for " + weights);
        };
    }

    /**
     * A stage's key/value cache holds its own layers, at offsets the layers read from the state.
     */
    @Override
    public State stageState(
            Model model, State session, int firstLayer, int endLayer, int prefillBatchSize) {
        State stage =
                State.withStorageOptions(
                        session.storageOptions(),
                        () ->
                                State.withPrefillBatchSize(
                                        prefillBatchSize,
                                        () ->
                                                Gemma4State.withLayerRange(
                                                        firstLayer,
                                                        endLayer,
                                                        model::createNewState)));
        stage.resolveExecutionPolicy(session.executionPolicy());
        return stage;
    }

    /**
     * The quantized layers split, single-token and batched; the F16 (BF16 file) layers have no
     * stage layers.
     */
    @Override
    public Optional<String> splitUnsupported(DataType weights, ExecutionPolicy policy, int stages) {
        return weights == DataType.F16
                ? TornadoPlanProvider.super.splitUnsupported(weights, policy, stages)
                : Optional.empty();
    }

    /**
     * The batched projections and attention are tensor-core kernels, with no scalar twin. Without
     * tensor cores the plan fails at "MMA instructions only supported for the CUDA backend".
     */
    @Override
    public Optional<String> batchPrefillUnsupported(Combination c) {
        return c.tensorCores()
                        || (BackendId.METAL.equals(c.backend()) && c.weights() == DataType.Q8_0)
                ? Optional.empty()
                : Optional.of(
                        "the gemma4 batched prefill is written for tensor cores, and for Q8_0"
                                + " weights on the Metal SIMD-group kernels, neither of which"
                                + " this combination has");
    }

    /**
     * The quantized layers' FP16 writers and grouped attention, in both of this family's modes: the
     * shuffle-reduced grouped kernel on CUDA, its shared-memory twin on OpenCL. The F16 (BF16 file)
     * layers write the FP16 cache too, attending with the split kernel.
     */
    @Override
    public Optional<String> fp16KeyValueUnsupported(Combination c) {
        if (c.weights() != DataType.Q8_0
                && c.weights() != DataType.Q4_0
                && c.weights() != DataType.F16) {
            return Optional.of("the gemma4 " + c.weights() + " layers keep an FP32 cache");
        }
        return Optional.empty();
    }

    /**
     * CUDA: the projections through cuBLAS over the weights decoded to FP16. Metal: the Q8_0
     * projections on Apple MLX, attention kept on the SIMD-group kernels.
     */
    @Override
    public Set<DataType> nativeLibraryWeights(BackendId backend) {
        if (BackendId.METAL.equals(backend)) {
            return Set.of(DataType.Q8_0);
        }
        return BackendId.CUDA.equals(backend) ? Set.of(DataType.Q8_0, DataType.Q4_0) : Set.of();
    }

    /** The MLX projections' FP16 copies (or affine repack) and staging, on Metal. */
    @Override
    public Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        if (!(config instanceof Gemma4Configuration gemma4)) {
            return Optional.empty();
        }
        return Gemma4BatchPrefillLayers.mlxPrefill(policy, weights, gemma4)
                .map(m -> Gemma4BatchPrefillLayers.mlxWorkspace(weights, gemma4, batchSize));
    }

    /** The native-library prefill decodes each projection for cuBLAS, from blocks, not tiles. */
    @Override
    public Optional<String> packedWeightsUnsupported(ExecutionPolicy policy) {
        return policy.nativeLibraries()
                ? Optional.of(
                        "its native-library prefill decodes the weights for cuBLAS and reads no"
                                + " packed projections")
                : Optional.empty();
    }
}
