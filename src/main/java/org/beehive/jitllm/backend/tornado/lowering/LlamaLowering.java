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
 * Lowers a Llama single-token {@link InferenceProgram} onto TornadoVM task graphs.
 *
 * <p>A lowering that checked the architecture string and then called the legacy builder regardless
 * of the components it was handed would be a wrapper, and the program would determine nothing. This
 * one <b>verifies the supported topology before constructing anything</b> — the ordered components,
 * their operands, dtypes and geometry, the phases, the attention scaling, the key/value append, and
 * the policy-dependent components. Anything missing, reordered or incompatible fails by name.
 *
 * <p>It recognizes <b>one exact supported pattern</b>. That is the whole design: this is not a
 * fusion-rule engine and must not become one.
 *
 * <h2>The mapping is many-to-many, and the existing builders own it</h2>
 *
 * <p>Twelve logical components become N+2 graphs and 9 to 11 tasks per layer. One {@code RmsNorm}
 * becomes two or three tasks; three {@code MatVec} become one {@code qkv_projection}; {@code RoPE}
 * and {@code KvAppend} fuse; {@code rms_ffn_gate_up} covers four operations. That fusion is what
 * the performance depends on, so this class <b>delegates to the existing builders unchanged</b>
 * rather than reproducing them. No kernel body is touched, and nothing under {@code
 * tornadovm.kernels} is referenced from here.
 *
 * <h2>Direction</h2>
 *
 * <p>{@code tornadovm.lowering} → {@code program}, {@code runtime}, TornadoVM. Never the reverse:
 * the generic program package has no dependency that could carry a task graph, a device or a fusion
 * choice.
 */
public final class LlamaLowering implements FamilyLowering {

    static final String FAMILY = "Llama-shaped single-token at F16 or Q8_0";

    private final CompileOptions compileOptions;
    private final DeviceCapabilities capabilities;
    private final ArchitectureId architecture;

    public LlamaLowering(CompileOptions compileOptions, DeviceCapabilities capabilities) {
        this(compileOptions, capabilities, ArchitectureId.of("llama"));
    }

    /**
     * The same validation under another family's name.
     *
     * <p>Mistral runs {@code InferenceCore.forwardJava} — Llama's own method — so its programs are
     * this shape, and a second class asserting the same fifteen kinds in the same order would be a
     * copy pretending to be a check. The <b>architecture is still matched exactly</b>: this
     * lowering refuses a program whose signature names a family it was not constructed for, so
     * "same shape" never becomes "same program".
     */
    public LlamaLowering(
            CompileOptions compileOptions,
            DeviceCapabilities capabilities,
            ArchitectureId architecture) {
        this.compileOptions = compileOptions;
        this.capabilities = capabilities;
        this.architecture = architecture;
    }

    @Override
    public ArchitectureId architecture() {
        return architecture;
    }

    /**
     * Validates, then builds the compiled program.
     *
     * <p>The binding domain is not a parameter of this call by accident: it is fixed at
     * construction of the resulting plan and never varies afterwards, because a captured graph
     * bakes device addresses (capability C1). What varies per invocation are values written into
     * control arrays.
     *
     * @throws UnsupportedProgramException if the program is not this slice's supported topology
     */
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

    // validation

    /**
     * The supported pattern, checked in order.
     *
     * <p>Written as explicit structural checks rather than as a rule table: one family, one
     * execution mode. The skeleton every single-token decoder shares lives in {@link ProgramShape};
     * <b>the layer sequence below is this family's own</b>, and stays written out even where
     * another family's looks like it.
     */
    @Override
    public void validate(InferenceProgram program) {
        ProgramShape.validateSkeleton(FAMILY, program, architecture, LlamaLowering::validateLayer);
    }

    /** One layer, in the exact order this family performs it, in one weight representation. */
    private static void validateLayer(ProgramComponent.Composite layer, DataType weights) {
        OperationKind[] expected = {
            OperationKind.RMS_NORM, OperationKind.MAT_VEC, OperationKind.MAT_VEC,
            OperationKind.MAT_VEC, OperationKind.ROPE, OperationKind.KV_APPEND,
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

        RoPE rope = (RoPE) ProgramShape.leafOperation(inner.get(4));
        if (rope.layout() != RopeLayout.INTERLEAVED) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " rotary layout",
                    RopeLayout.INTERLEAVED.name(),
                    rope.layout().name());
        }

        Attention attention = (Attention) ProgramShape.leafOperation(inner.get(6));
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
