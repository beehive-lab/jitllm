package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.architecture.LlamaProgramDescription;
import org.beehive.jitllm.model.architecture.Qwen2ProgramDescription;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.program.op.RoPE;
import org.beehive.jitllm.program.op.RopeLayout;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The interesting assertions are the <b>refusals in both directions</b>. A validator that accepted
 * the other family's program would be recognizing an architecture string and nothing else, which is
 * the failure mode the whole lowering design exists to avoid — and it would show up not as a crash
 * but as a plausible-looking wrong plan.
 */
public class Qwen2LoweringTest {

    private static final int DIM = 64;
    private static final int LAYERS = 2;

    private final Qwen2Lowering qwen2 =
            new Qwen2Lowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));
    private final LlamaLowering llama =
            new LlamaLowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));

    @Test
    public void theDescribedProgramIsQwen2sOperationSequence() {
        InferenceProgram program = qwen2Program(DataType.F16);
        qwen2.validate(program); // must not throw
        assertTrue(qwen2.supports(program));

        assertEquals(
                "embedding, two layers, final norm, projection",
                1 + LAYERS + 2,
                program.components().size());

        List<OperationKind> layer = kindsOf(program, 1);
        assertEquals(
                "the three projection biases sit between the projections and the rotation",
                List.of(
                        OperationKind.RMS_NORM,
                        OperationKind.MAT_VEC,
                        OperationKind.MAT_VEC,
                        OperationKind.MAT_VEC,
                        OperationKind.BIAS_ADD,
                        OperationKind.BIAS_ADD,
                        OperationKind.BIAS_ADD,
                        OperationKind.ROPE,
                        OperationKind.KV_APPEND,
                        OperationKind.ATTENTION,
                        OperationKind.MAT_VEC,
                        OperationKind.RESIDUAL_ADD,
                        OperationKind.RMS_NORM,
                        OperationKind.MAT_VEC,
                        OperationKind.MAT_VEC,
                        OperationKind.SWIGLU,
                        OperationKind.MAT_VEC,
                        OperationKind.RESIDUAL_ADD),
                layer);
    }

    /** Acceptance 2: NEOX_HALF, and the interleaved arrangement is refused. */
    @Test
    public void theInterleavedRotaryArrangementIsRefused() {
        InferenceProgram original = qwen2Program(DataType.F16);
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) original.components().get(1);
        assertEquals(
                RopeLayout.NEOX_HALF,
                ((RoPE) ((ProgramComponent.Leaf) layer.children().get(7)).operation()).layout());

        List<ProgramComponent> children = new ArrayList<>(layer.children());
        ProgramComponent.Leaf rope = (ProgramComponent.Leaf) children.get(7);
        RoPE original_ = (RoPE) rope.operation();
        children.set(
                7,
                new ProgramComponent.Leaf(
                        rope.name(),
                        new RoPE(
                                original_.query(),
                                original_.key(),
                                original_.frequencies(),
                                original_.headDimension(),
                                original_.ropeTheta(),
                                RopeLayout.INTERLEAVED,
                                original_.dataType()),
                        rope.phases()));

        assertRefused(
                "Llama's rotary arrangement in a Qwen2 program",
                () ->
                        qwen2.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), children, layer.phases()))));
    }

    /**
     * Acceptance 3: each family refuses the other's program.
     *
     * <p>Both directions, because one direction is satisfiable by an architecture-string check.
     * Qwen2's program has three components Llama's layer does not, and Llama's has a rotary layout
     * Qwen2 never uses; each lowering must notice its own reason.
     */
    @Test
    public void eachFamilyRefusesTheOthersProgram() {
        InferenceProgram qwen2Program = qwen2Program(DataType.F16);
        InferenceProgram llamaProgram =
                LlamaProgramDescription.build(
                        llamaConfig(), DataType.F16, DataType.F32, false, false);

        assertFalse(
                "Llama's lowering must not accept a Qwen2 program", llama.supports(qwen2Program));
        assertFalse(
                "Qwen2's lowering must not accept a Llama program", qwen2.supports(llamaProgram));
        assertTrue(llama.supports(llamaProgram));
        assertTrue(qwen2.supports(qwen2Program));

        assertNotEquals(
                "and the two are not one program",
                qwen2Program.signature(),
                llamaProgram.signature());
    }

    /** Acceptance 4: Q8_0 as for Llama — same sequence, different signature, still supported. */
    @Test
    public void bothWeightRepresentationsAreSupportedAndAreNotOneProgram() {
        InferenceProgram f16 = qwen2Program(DataType.F16);
        InferenceProgram q8 = qwen2Program(DataType.Q8_0);

        qwen2.validate(q8);
        assertEquals("Q8_0 adds no operation", kindsOf(f16, 1), kindsOf(q8, 1));
        assertNotEquals(f16.signature(), q8.signature());
    }

    @Test
    public void anUnsupportedWeightRepresentationIsRefused() {
        assertFalse(qwen2.supports(qwen2Program(DataType.BF16)));
        assertFalse(qwen2.supports(qwen2Program(DataType.F32)));
    }

    // helpers

    private static InferenceProgram qwen2Program(DataType weights) {
        return Qwen2ProgramDescription.build(qwen2Config(), weights, DataType.F32, false, false);
    }

    private static Qwen2Configuration qwen2Config() {
        return new Qwen2Configuration(
                "FP16", DIM, 128, LAYERS, 4, 2, 16, 16, 48, 32, 32, false, 1e-5f, 1000000f);
    }

    private static LlamaConfiguration llamaConfig() {
        return new LlamaConfiguration("FP16", DIM, 128, LAYERS, 4, 2, 48, 32, 1e-5f, 500000f);
    }

    private static List<OperationKind> kindsOf(InferenceProgram program, int componentIndex) {
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) program.components().get(componentIndex);
        List<OperationKind> kinds = new ArrayList<>();
        for (ProgramComponent child : layer.children()) {
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
            assertTrue(
                    "the refusal must name what was expected and what was found",
                    expected.getMessage().contains("expected")
                            && expected.getMessage().contains("found"));
        }
    }
}
