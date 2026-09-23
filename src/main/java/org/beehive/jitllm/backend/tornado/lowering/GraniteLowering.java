package org.beehive.jitllm.backend.tornado.lowering;

import java.util.List;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanSingleToken;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.KvAppend;
import org.beehive.jitllm.program.op.MatVec;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.program.op.RoPE;
import org.beehive.jitllm.program.op.RopeLayout;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Lowers a Granite single-token {@link InferenceProgram} onto TornadoVM task graphs.
 *
 * <p>Llama's layer with two {@code Scale} components — one on each residual branch, before it is
 * added back — and an {@code Attention} whose scale <b>multiplies</b>. The multiplying mode is
 * checked rather than assumed: a Granite program that divided would be arithmetically a different
 * model and would still validate against every structural rule.
 *
 * <p>The remaining two µP factors are outside the layer: the embedding scale immediately after the
 * lookup, and the logit scale after the vocabulary projection. Both are checked in {@link
 * #validate}.
 */
public final class GraniteLowering implements FamilyLowering {

    static final String FAMILY = "Granite single-token at F16 or Q8_0";

    private final CompileOptions compileOptions;
    private final DeviceCapabilities capabilities;

    public GraniteLowering(CompileOptions compileOptions, DeviceCapabilities capabilities) {
        this.compileOptions = compileOptions;
        this.capabilities = capabilities;
    }

    @Override
    public ArchitectureId architecture() {
        return ArchitectureId.of("granite");
    }

    /**
     * Granite's skeleton is not the shared one: a {@code Scale} follows the embedding and another
     * follows the projection, so the shared checks are applied to the program either side of them.
     */
    @Override
    public void validate(InferenceProgram program) {
        ProgramShape.validateSkeleton(
                FAMILY,
                program,
                ArchitectureId.of("granite"),
                GraniteLowering::validateLayer,
                ProgramShape.TailShape.SCALED_LOGITS,
                true);
    }

    @Override
    public TornadoVMMasterPlan lower(
            InferenceProgram program, State state, Model model, MetricsSink sink) {
        validate(program);
        return new TornadoVMMasterPlanSingleToken(state, model, sink);
    }

    /** What this lowering was built for, for diagnostics and for the cache key. */
    public String describeContext() {
        return compileOptions.fingerprint() + ";" + capabilities.fingerprint();
    }

    private static void validateLayer(ProgramComponent.Composite layer, DataType weights) {
        OperationKind[] expected = {
            OperationKind.RMS_NORM, OperationKind.MAT_VEC, OperationKind.MAT_VEC,
            OperationKind.MAT_VEC, OperationKind.ROPE, OperationKind.KV_APPEND,
            OperationKind.ATTENTION, OperationKind.MAT_VEC, OperationKind.SCALE,
            OperationKind.RESIDUAL_ADD, OperationKind.RMS_NORM, OperationKind.MAT_VEC,
            OperationKind.MAT_VEC, OperationKind.SWIGLU, OperationKind.MAT_VEC,
            OperationKind.SCALE, OperationKind.RESIDUAL_ADD
        };
        ProgramShape.expectLayerSequence(FAMILY, layer, expected);

        List<ProgramComponent> inner = layer.children();
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == OperationKind.MAT_VEC) {
                ProgramShape.expectWeightRepresentation(
                        FAMILY,
                        layer.name() + " component " + i,
                        ((MatVec) ProgramShape.leafOperation(inner.get(i))).dataType(),
                        weights);
            }
        }

        RoPE rope = (RoPE) ProgramShape.leafOperation(inner.get(4));
        if (rope.layout() != RopeLayout.INTERLEAVED) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " rotary layout",
                    RopeLayout.INTERLEAVED.name(),
                    rope.layout().name());
        }

        Attention attention = (Attention) ProgramShape.leafOperation(inner.get(6));
        // The µP multiplier replaces the conventional division. A Granite program that divided
        // would compute a different model and would pass every structural check, so the mode is
        // checked here and not left to the description to get right.
        if (attention.scoreScaling() != Attention.ScoreScaling.MULTIPLY) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " attention score scaling",
                    Attention.ScoreScaling.MULTIPLY.name(),
                    attention.scoreScaling().name());
        }
        if (attention.window().isPresent()) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " attention window",
                    "full causal attention",
                    "a sliding window of " + attention.window().getAsInt());
        }

        KvAppend append = (KvAppend) ProgramShape.leafOperation(inner.get(5));
        if (!append.keyStore().equals(attention.keys())
                || !append.valueStore().equals(attention.values())) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " key/value source",
                    "attention over the store this layer appends to",
                    "appends to "
                            + append.keyStore().name()
                            + " but attends over "
                            + attention.keys().name());
        }
    }
}
