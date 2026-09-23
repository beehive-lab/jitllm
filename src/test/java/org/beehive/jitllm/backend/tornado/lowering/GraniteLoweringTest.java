package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.architecture.GraniteProgramDescription;
import org.beehive.jitllm.model.architecture.LlamaProgramDescription;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The assertion that matters is the fifth: the attention multiplier <b>replaces</b> the
 * conventional division. A Granite program that divided by the same number would satisfy every
 * structural rule in this file and compute a different model, so the mode is checked explicitly.
 */
public class GraniteLoweringTest {

    private static final int DIM = 64;
    private static final int LAYERS = 2;

    private final GraniteLowering granite =
            new GraniteLowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));
    private final LlamaLowering llama =
            new LlamaLowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));

    @Test
    public void theDescribedProgramCarriesTheFourScaleComponents() {
        InferenceProgram program = program(DataType.F16);
        granite.validate(program);
        assertTrue(granite.supports(program));

        List<OperationKind> top = new ArrayList<>();
        for (ProgramComponent component : program.components()) {
            if (component instanceof ProgramComponent.Leaf leaf) {
                top.add(leaf.operation().kind());
            }
        }
        assertEquals(
                "embedding, its scale, then the final norm, the projection and its scale",
                List.of(
                        OperationKind.EMBEDDING_LOOKUP,
                        OperationKind.SCALE,
                        OperationKind.RMS_NORM,
                        OperationKind.VOCAB_PROJECTION,
                        OperationKind.SCALE),
                top);

        List<OperationKind> layer = layerKinds(program);
        assertEquals(
                "a scale sits on each residual branch, before it is added back",
                OperationKind.SCALE,
                layer.get(8));
        assertEquals(OperationKind.RESIDUAL_ADD, layer.get(9));
        assertEquals(OperationKind.SCALE, layer.get(15));
        assertEquals(OperationKind.RESIDUAL_ADD, layer.get(16));
    }

    @Test
    public void theAttentionScaleMultipliesAndDividingIsRefused() {
        InferenceProgram original = program(DataType.F16);
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) original.components().get(2);
        ProgramComponent.Leaf attention = (ProgramComponent.Leaf) layer.children().get(6);
        Attention op = (Attention) attention.operation();
        assertEquals(Attention.ScoreScaling.MULTIPLY, op.scoreScaling());

        List<ProgramComponent> divided = new ArrayList<>(layer.children());
        divided.set(
                6,
                new ProgramComponent.Leaf(
                        attention.name(),
                        new Attention(
                                op.query(),
                                op.keys(),
                                op.values(),
                                op.output(),
                                op.heads(),
                                op.keyValueHeads(),
                                op.headDimension(),
                                op.scale(),
                                Attention.ScoreScaling.DIVIDE,
                                op.window(),
                                op.dataType()),
                        attention.phases()));

        assertRefused(
                "an attention that divides where Granite multiplies",
                () ->
                        granite.validate(
                                rebuild(
                                        original,
                                        2,
                                        new ProgramComponent.Composite(
                                                layer.name(), divided, layer.phases()))));
    }

    @Test
    public void aGraniteProgramWithoutItsLogitScaleIsRefused() {
        InferenceProgram original = program(DataType.F16);
        List<ProgramComponent> components = new ArrayList<>(original.components());
        components.removeLast(); // the logit scale
        assertRefused(
                "a Granite program with no logit scale",
                () ->
                        granite.validate(
                                InferenceProgram.of(
                                        new org.beehive.jitllm.program.ProgramSignature(
                                                original.signature().architecture(),
                                                        original.signature().policy(),
                                                original.signature().capacity(), components,
                                                phasesFor(components),
                                                        original.signature().bindings()))));
    }

    @Test
    public void eachFamilyRefusesTheOthersProgram() {
        InferenceProgram graniteProgram = program(DataType.F16);
        InferenceProgram llamaProgram =
                LlamaProgramDescription.build(
                        new LlamaConfiguration(
                                "FP16", DIM, 128, LAYERS, 4, 2, 48, 32, 1e-5f, 500000f),
                        DataType.F16,
                        DataType.F32,
                        false,
                        false);

        assertFalse(llama.supports(graniteProgram));
        assertFalse(granite.supports(llamaProgram));
        assertNotEquals(graniteProgram.signature(), llamaProgram.signature());
    }

    @Test
    public void bothWeightRepresentationsAreSupportedAndAreNotOneProgram() {
        InferenceProgram f16 = program(DataType.F16);
        InferenceProgram q8 = program(DataType.Q8_0);
        granite.validate(q8);
        assertEquals(layerKinds(f16), layerKinds(q8));
        assertNotEquals(f16.signature(), q8.signature());
    }

    // helpers

    private static InferenceProgram program(DataType weights) {
        return GraniteProgramDescription.build(config(), weights, DataType.F32, false, false);
    }

    private static GraniteConfiguration config() {
        return new GraniteConfiguration(
                "FP16",
                DIM,
                128,
                LAYERS,
                4,
                2,
                48,
                32,
                1e-5f,
                10000f,
                12.0f,
                0.22f,
                0.0078125f,
                16.0f,
                true);
    }

    private static List<OperationKind> layerKinds(InferenceProgram program) {
        ProgramComponent.Composite layer = (ProgramComponent.Composite) program.components().get(2);
        List<OperationKind> kinds = new ArrayList<>();
        for (ProgramComponent child : layer.children()) {
            kinds.add(((ProgramComponent.Leaf) child).operation().kind());
        }
        return kinds;
    }

    private static List<org.beehive.jitllm.program.PhaseSelection> phasesFor(
            List<ProgramComponent> components) {
        List<Integer> prefill = new ArrayList<>();
        List<Integer> decode = new ArrayList<>();
        for (int i = 0; i < components.size(); i++) {
            if (components.get(i).phases().contains(org.beehive.jitllm.program.PhaseId.PREFILL)) {
                prefill.add(i);
            }
            decode.add(i);
        }
        return List.of(
                new org.beehive.jitllm.program.PhaseSelection(
                        org.beehive.jitllm.program.PhaseId.PREFILL, prefill),
                new org.beehive.jitllm.program.PhaseSelection(
                        org.beehive.jitllm.program.PhaseId.DECODE, decode));
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
