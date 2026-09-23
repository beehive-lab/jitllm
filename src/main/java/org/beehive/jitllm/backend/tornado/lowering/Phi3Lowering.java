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
import org.beehive.jllm.program.op.SplitFusedQkv;
import org.beehive.jllm.program.op.SplitGateUp;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Lowers a Phi3 single-token {@link InferenceProgram} onto TornadoVM task graphs.
 *
 * <p>Lowering is a delegation, as for every other family: the fused Phi3 kernels already exist and
 * already fuse projection, split and activation — {@code splitGateUpAndSiLU} does all three. That
 * is fusion doing its job, not the program being wrong.
 */
public final class Phi3Lowering implements FamilyLowering {

    static final String FAMILY = "Phi3 single-token at F16 or Q8_0";

    private final CompileOptions compileOptions;
    private final DeviceCapabilities capabilities;

    public Phi3Lowering(CompileOptions compileOptions, DeviceCapabilities capabilities) {
        this.compileOptions = compileOptions;
        this.capabilities = capabilities;
    }

    @Override
    public ArchitectureId architecture() {
        return ArchitectureId.of("phi3");
    }

    @Override
    public void validate(InferenceProgram program) {
        ProgramShape.validateSkeleton(
                FAMILY, program, ArchitectureId.of("phi3"), Phi3Lowering::validateLayer);
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
            OperationKind.RMS_NORM, OperationKind.MAT_VEC, OperationKind.SPLIT_FUSED_QKV,
            OperationKind.ROPE, OperationKind.KV_APPEND, OperationKind.ATTENTION,
            OperationKind.MAT_VEC, OperationKind.RESIDUAL_ADD, OperationKind.RMS_NORM,
            OperationKind.MAT_VEC, OperationKind.SPLIT_GATE_UP, OperationKind.SWIGLU,
            OperationKind.MAT_VEC, OperationKind.RESIDUAL_ADD
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

        // Each split must read exactly what its projection produced. A mismatch reads past the end
        // of the buffer, silently, and every structural rule above still passes.
        MatVec qkvProjection = (MatVec) ProgramShape.leafOperation(inner.get(1));
        SplitFusedQkv qkvSplit = (SplitFusedQkv) ProgramShape.leafOperation(inner.get(2));
        if (qkvProjection.rows() != qkvSplit.fusedWidth()) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " fused QKV width",
                    String.valueOf(qkvProjection.rows()),
                    String.valueOf(qkvSplit.fusedWidth()));
        }
        if (!qkvProjection.output().equals(qkvSplit.fused())) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " fused QKV operand",
                    "the split reads the projection's output",
                    "projection writes "
                            + qkvProjection.output().name()
                            + ", split reads "
                            + qkvSplit.fused().name());
        }

        MatVec gateUpProjection = (MatVec) ProgramShape.leafOperation(inner.get(9));
        SplitGateUp gateUpSplit = (SplitGateUp) ProgramShape.leafOperation(inner.get(10));
        if (gateUpProjection.rows() != gateUpSplit.fusedWidth()) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " fused gate/up width",
                    String.valueOf(gateUpProjection.rows()),
                    String.valueOf(gateUpSplit.fusedWidth()));
        }
        if (!gateUpProjection.output().equals(gateUpSplit.fused())) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " fused gate/up operand",
                    "the split reads the projection's output",
                    "projection writes "
                            + gateUpProjection.output().name()
                            + ", split reads "
                            + gateUpSplit.fused().name());
        }

        RoPE rope = (RoPE) ProgramShape.leafOperation(inner.get(3));
        if (rope.layout() != RopeLayout.NEOX_HALF) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " rotary layout",
                    RopeLayout.NEOX_HALF.name(),
                    rope.layout().name());
        }

        Attention attention = (Attention) ProgramShape.leafOperation(inner.get(5));
        if (attention.window().isPresent()) {
            throw new UnsupportedProgramException(
                    FAMILY,
                    layer.name() + " attention window",
                    "full causal attention",
                    "a sliding window of " + attention.window().getAsInt());
        }

        KvAppend append = (KvAppend) ProgramShape.leafOperation(inner.get(4));
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
