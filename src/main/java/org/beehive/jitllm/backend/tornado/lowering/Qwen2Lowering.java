package org.beehive.jllm.backend.tornado.lowering;

import java.util.List;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlanSingleToken;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.ProgramComponent;
import org.beehive.jllm.program.op.Attention;
import org.beehive.jllm.program.op.KvAppend;
import org.beehive.jllm.program.op.MatVec;
import org.beehive.jllm.program.op.OperationKind;
import org.beehive.jllm.program.op.RoPE;
import org.beehive.jllm.program.op.RopeLayout;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Lowers a Qwen2 single-token {@link InferenceProgram} onto TornadoVM task graphs.
 *
 * <p>The second family, and the one that shows the seam is real: it differs from Llama by three
 * {@code BiasAdd} components and a NEOX_HALF rotary layout, and by nothing else. Both differences
 * are checked here rather than assumed — a validator that accepted either layout would accept a
 * Llama program described as Qwen2, which is the failure this class exists to make impossible.
 */
public final class Qwen2Lowering implements FamilyLowering {

    static final String FAMILY = "Qwen2 single-token at F16 or Q8_0";

    private final CompileOptions compileOptions;
    private final DeviceCapabilities capabilities;

    public Qwen2Lowering(CompileOptions compileOptions, DeviceCapabilities capabilities) {
        this.compileOptions = compileOptions;
        this.capabilities = capabilities;
    }

    @Override
    public ArchitectureId architecture() {
        return ArchitectureId.of("qwen2");
    }

    @Override
    public void validate(InferenceProgram program) {
        ProgramShape.validateSkeleton(
                FAMILY, program, ArchitectureId.of("qwen2"), Qwen2Lowering::validateLayer);
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

    /**
     * The three {@code BiasAdd} sit between the projections and the rotary step, which is where the
     * bias is actually applied — putting them after the rotation would be a different model.
     */
    private static void validateLayer(ProgramComponent.Composite layer, DataType weights) {
        OperationKind[] expected = {
            OperationKind.RMS_NORM, OperationKind.MAT_VEC, OperationKind.MAT_VEC,
            OperationKind.MAT_VEC, OperationKind.BIAS_ADD, OperationKind.BIAS_ADD,
            OperationKind.BIAS_ADD, OperationKind.ROPE, OperationKind.KV_APPEND,
            OperationKind.ATTENTION, OperationKind.MAT_VEC, OperationKind.RESIDUAL_ADD,
            OperationKind.RMS_NORM, OperationKind.MAT_VEC, OperationKind.MAT_VEC,
            OperationKind.SWIGLU, OperationKind.MAT_VEC, OperationKind.RESIDUAL_ADD
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

        RoPE rope = (RoPE) ProgramShape.leafOperation(inner.get(7));
        if (rope.layout() != RopeLayout.NEOX_HALF) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " rotary layout",
                    RopeLayout.NEOX_HALF.name(),
                    rope.layout().name());
        }

        Attention attention = (Attention) ProgramShape.leafOperation(inner.get(9));
        if (attention.window().isPresent()) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " attention window",
                    "full causal attention",
                    "a sliding window of " + attention.window().getAsInt());
        }
        float expectedScale = (float) Math.sqrt(attention.headDimension());
        if (Float.compare(attention.scale(), expectedScale) != 0) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " attention scale",
                    "sqrt(headDimension) = " + expectedScale,
                    String.valueOf(attention.scale()));
        }

        KvAppend append = (KvAppend) ProgramShape.leafOperation(inner.get(8));
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
