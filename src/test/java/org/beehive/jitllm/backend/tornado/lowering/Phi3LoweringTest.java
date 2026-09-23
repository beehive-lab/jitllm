package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.architecture.LlamaProgramDescription;
import org.beehive.jitllm.model.architecture.Phi3ProgramDescription;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.MatVec;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.program.op.SplitFusedQkv;
import org.beehive.jitllm.program.op.SplitGateUp;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Each projection is one {@code MatVec} — that is what the model performs — followed by the split
 * that is also what the model performs. The assertions worth having are the two widths: a split
 * reading a different width than its projection produced addresses past the end of the buffer, and
 * every structural rule still passes.
 */
public class Phi3LoweringTest {

    private static final int DIM = 64;
    private static final int LAYERS = 2;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HIDDEN = 128;

    private final Phi3Lowering phi3 =
            new Phi3Lowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));
    private final LlamaLowering llama =
            new LlamaLowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));

    @Test
    public void theDescribedProgramIsPhi3sOperationSequence() {
        InferenceProgram program = program(DataType.F16);
        phi3.validate(program);
        assertTrue(phi3.supports(program));

        assertEquals(
                "one projection then its split, twice",
                List.of(
                        OperationKind.RMS_NORM,
                        OperationKind.MAT_VEC,
                        OperationKind.SPLIT_FUSED_QKV,
                        OperationKind.ROPE,
                        OperationKind.KV_APPEND,
                        OperationKind.ATTENTION,
                        OperationKind.MAT_VEC,
                        OperationKind.RESIDUAL_ADD,
                        OperationKind.RMS_NORM,
                        OperationKind.MAT_VEC,
                        OperationKind.SPLIT_GATE_UP,
                        OperationKind.SWIGLU,
                        OperationKind.MAT_VEC,
                        OperationKind.RESIDUAL_ADD),
                layerKinds(program));
    }

    /** Each fused projection stays one MatVec, of the width the split reads. */
    @Test
    public void theProjectionsAreOneMatVecEachAndTheWidthsAgree() {
        List<ProgramComponent> layer = layer(program(DataType.F16)).children();

        MatVec qkv = (MatVec) ((ProgramComponent.Leaf) layer.get(1)).operation();
        SplitFusedQkv qkvSplit = (SplitFusedQkv) ((ProgramComponent.Leaf) layer.get(2)).operation();
        assertEquals(
                "dim + 2 x kvHeads x headSize", DIM + 2 * (KV_HEADS * (DIM / HEADS)), qkv.rows());
        assertEquals(qkv.rows(), qkvSplit.fusedWidth());
        assertEquals("the split reads what the projection wrote", qkv.output(), qkvSplit.fused());

        MatVec gateUp = (MatVec) ((ProgramComponent.Leaf) layer.get(9)).operation();
        SplitGateUp gateUpSplit = (SplitGateUp) ((ProgramComponent.Leaf) layer.get(10)).operation();
        assertEquals("2 x hiddenDim", 2 * HIDDEN, gateUp.rows());
        assertEquals(gateUp.rows(), gateUpSplit.fusedWidth());
        assertEquals(HIDDEN, gateUpSplit.width());
        assertEquals(gateUp.output(), gateUpSplit.fused());
    }

    /**
     * A split that does not match its projection is refused.
     *
     * <p>This is the check the whole slice turns on: halves of the wrong width still look like
     * halves, and the read runs off the end of a buffer nothing else inspects.
     */
    @Test
    public void aSplitThatDoesNotMatchItsProjectionIsRefused() {
        InferenceProgram original = program(DataType.F16);
        ProgramComponent.Composite layer = layer(original);
        List<ProgramComponent> children = new ArrayList<>(layer.children());
        ProgramComponent.Leaf split = (ProgramComponent.Leaf) children.get(10);
        SplitGateUp op = (SplitGateUp) split.operation();
        children.set(
                10,
                new ProgramComponent.Leaf(
                        split.name(),
                        new SplitGateUp(
                                op.fused(), op.gate(), op.up(), op.width() + 1, op.dataType()),
                        split.phases()));

        assertRefused(
                "a gate/up split one element wider than its projection",
                () ->
                        phi3.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), children, layer.phases()))));
    }

    @Test
    public void phi3AndLlamaRefuseEachOthersPrograms() {
        InferenceProgram phi3Program = program(DataType.F16);
        InferenceProgram llamaProgram =
                LlamaProgramDescription.build(
                        new LlamaConfiguration(
                                "FP16", DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, 48, 32, 1e-5f,
                                500000f),
                        DataType.F16,
                        DataType.F32,
                        false,
                        false);

        assertFalse(llama.supports(phi3Program));
        assertFalse(phi3.supports(llamaProgram));
        assertNotEquals(phi3Program.signature(), llamaProgram.signature());
    }

    @Test
    public void bothWeightRepresentationsAreSupportedAndAreNotOneProgram() {
        InferenceProgram f16 = program(DataType.F16);
        InferenceProgram q8 = program(DataType.Q8_0);
        phi3.validate(q8);
        assertEquals(layerKinds(f16), layerKinds(q8));
        assertNotEquals(f16.signature(), q8.signature());
    }

    // helpers

    private static InferenceProgram program(DataType weights) {
        return Phi3ProgramDescription.build(config(), weights, DataType.F32, false, false);
    }

    private static Phi3Configuration config() {
        return new Phi3Configuration(
                "FP16", DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, 48, 32, 1e-5f, 10000f);
    }

    private static ProgramComponent.Composite layer(InferenceProgram program) {
        return (ProgramComponent.Composite) program.components().get(1);
    }

    private static List<OperationKind> layerKinds(InferenceProgram program) {
        List<OperationKind> kinds = new ArrayList<>();
        for (ProgramComponent child : layer(program).children()) {
            kinds.add(((ProgramComponent.Leaf) child).operation().kind());
        }
        return kinds;
    }

    private static InferenceProgram rebuild(
            InferenceProgram original, int index, ProgramComponent replacement) {
        List<ProgramComponent> components = new ArrayList<>(original.components());
        components.set(index, replacement);
        return InferenceProgram.of(
                new org.beehive.jitllm.program.ProgramSignature(
                        original.signature().architecture(),
                        original.signature().policy(),
                        original.signature().capacity(),
                        components,
                        original.signature().phases(),
                        original.signature().bindings()));
    }

    private static void assertRefused(String what, Runnable body) {
        try {
            body.run();
            fail("expected " + what + " to be refused");
        } catch (UnsupportedProgramException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("expected"));
        }
    }
}
