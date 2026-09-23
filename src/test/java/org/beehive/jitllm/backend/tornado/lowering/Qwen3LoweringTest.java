package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.model.architecture.Qwen2ProgramDescription;
import org.beehive.jllm.model.architecture.Qwen3ProgramDescription;
import org.beehive.jllm.model.qwen2.Qwen2Configuration;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.ProgramComponent;
import org.beehive.jllm.program.op.OperationKind;
import org.beehive.jllm.program.op.RmsNorm;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Both are invisible to structural rules. A norm over the whole query instead of per head, and a
 * projection width taken from {@code dim} instead of from {@code attention.key_length}, each
 * produce a program that validates and a model that computes something else. So the grouping is
 * asserted, and the widths are asserted against a configuration whose key length deliberately does
 * <b>not</b> equal {@code dim / heads}.
 */
public class Qwen3LoweringTest {

    private static final int DIM = 64;
    private static final int LAYERS = 2;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;

    /** Deliberately not {@code DIM / HEADS} (16): that equality is what hides an addressing bug. */
    private static final int HEAD_KEY = 32;

    private static final int HEAD_VALUE = 32;

    private final Qwen3Lowering qwen3 =
            new Qwen3Lowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));
    private final Qwen2Lowering qwen2 =
            new Qwen2Lowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));

    @Test
    public void theDescribedProgramIsQwen3sOperationSequence() {
        InferenceProgram program = program(DataType.F16);
        qwen3.validate(program);
        assertTrue(qwen3.supports(program));

        assertEquals(
                "two norms where Qwen2 has three biases",
                List.of(
                        OperationKind.RMS_NORM,
                        OperationKind.MAT_VEC,
                        OperationKind.MAT_VEC,
                        OperationKind.MAT_VEC,
                        OperationKind.RMS_NORM,
                        OperationKind.RMS_NORM,
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
                layerKinds(program));
    }

    @Test
    public void theQueryAndKeyNormsAreGroupedByHead() {
        List<ProgramComponent> layer = layer(program(DataType.F16)).children();
        RmsNorm queryNorm = (RmsNorm) ((ProgramComponent.Leaf) layer.get(4)).operation();
        RmsNorm keyNorm = (RmsNorm) ((ProgramComponent.Leaf) layer.get(5)).operation();

        assertEquals("one group per query head", HEADS, queryNorm.groups());
        assertEquals(HEAD_VALUE, queryNorm.groupLength());
        assertEquals("one group per key/value head", KV_HEADS, keyNorm.groups());
        assertEquals(HEAD_VALUE, keyNorm.groupLength());
    }

    /** An ungrouped norm normalizes across head boundaries. It must be refused, not accepted. */
    @Test
    public void anUngroupedQueryNormIsRefused() {
        InferenceProgram original = program(DataType.F16);
        ProgramComponent.Composite layer = layer(original);
        List<ProgramComponent> children = new ArrayList<>(layer.children());
        ProgramComponent.Leaf norm = (ProgramComponent.Leaf) children.get(4);
        RmsNorm op = (RmsNorm) norm.operation();
        children.set(
                4,
                new ProgramComponent.Leaf(
                        norm.name(),
                        new RmsNorm(
                                op.input(), op.weight(), op.output(), op.epsilon(), op.dataType()),
                        norm.phases()));

        assertRefused(
                "a query norm over the whole vector",
                () ->
                        qwen3.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), children, layer.phases()))));
    }

    /** The projections are sized from the key length, not from {@code dim / heads}. */
    @Test
    public void theProjectionWidthsComeFromTheHeadLengths() {
        List<ProgramComponent> layer = layer(program(DataType.F16)).children();
        var query =
                (org.beehive.jllm.program.op.MatVec)
                        ((ProgramComponent.Leaf) layer.get(1)).operation();
        var key =
                (org.beehive.jllm.program.op.MatVec)
                        ((ProgramComponent.Leaf) layer.get(2)).operation();
        var output =
                (org.beehive.jllm.program.op.MatVec)
                        ((ProgramComponent.Leaf) layer.get(9)).operation();

        assertEquals("heads × key_length, not dim", HEADS * HEAD_KEY, query.rows());
        assertEquals("kv heads × value_length", KV_HEADS * HEAD_VALUE, key.rows());
        assertEquals(
                "the output projection reads the query width", HEADS * HEAD_KEY, output.columns());
        assertEquals(DIM, output.rows());
    }

    @Test
    public void qwen2AndQwen3RefuseEachOthersPrograms() {
        InferenceProgram qwen3Program = program(DataType.F16);
        InferenceProgram qwen2Program =
                Qwen2ProgramDescription.build(
                        new Qwen2Configuration(
                                "FP16", DIM, 128, LAYERS, HEADS, KV_HEADS, 16, 16, 48, 32, 32,
                                false, 1e-5f, 1000000f),
                        DataType.F16,
                        DataType.F32,
                        false,
                        false);

        assertFalse(qwen2.supports(qwen3Program));
        assertFalse(qwen3.supports(qwen2Program));
        assertNotEquals(qwen3Program.signature(), qwen2Program.signature());
    }

    @Test
    public void bothWeightRepresentationsAreSupportedAndAreNotOneProgram() {
        InferenceProgram f16 = program(DataType.F16);
        InferenceProgram q8 = program(DataType.Q8_0);
        qwen3.validate(q8);
        assertEquals(layerKinds(f16), layerKinds(q8));
        assertNotEquals(f16.signature(), q8.signature());
    }

    // helpers

    private static InferenceProgram program(DataType weights) {
        return Qwen3ProgramDescription.build(config(), weights, DataType.F32, false, false);
    }

    private static Qwen3Configuration config() {
        return new Qwen3Configuration(
                "FP16",
                DIM,
                128,
                LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_KEY,
                HEAD_VALUE,
                48,
                32,
                32,
                false,
                1e-5f,
                1000000f);
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
                new org.beehive.jllm.program.ProgramSignature(
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
