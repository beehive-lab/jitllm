package org.beehive.jllm.program;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.beehive.jllm.program.op.MatVec;
import org.beehive.jllm.program.op.OperandRef;
import org.beehive.jllm.program.op.RmsNorm;
import org.beehive.jllm.program.op.VocabProjection;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.Shape;
import org.beehive.jllm.runtime.tensor.TensorRole;
import org.junit.Test;

/**
 * The value tests are not stylistic. A {@link ProgramSignature} is the compiled-program cache key,
 * so a field that compares by identity is a silent recompile of a program that already exists — or
 * a claimed match that is not one. The array check below is the one that would catch it.
 */
public class InferenceProgramTest {

    private static final OperandRef X = new OperandRef.Activation("x");
    private static final OperandRef XB = new OperandRef.Activation("xb");
    private static final OperandRef LOGITS = new OperandRef.Activation("logits");
    private static final OperandRef.Weight NORM =
            new OperandRef.Weight("rms", TensorRole.OUTPUT_NORM);
    private static final OperandRef.Weight WEIGHT =
            new OperandRef.Weight("wq", TensorRole.ATTENTION_QUERY);
    private static final OperandRef.Weight OUTPUT =
            new OperandRef.Weight("wcls", TensorRole.OUTPUT);

    private static final Set<PhaseId> BOTH = EnumSet.allOf(PhaseId.class);
    private static final Set<PhaseId> DECODE_ONLY = EnumSet.of(PhaseId.DECODE);

    /**
     * The worked example the phase model exists for: prefill runs the layer work and stops before
     * the vocabulary projection, so its selection is a <b>strict prefix</b> here — and in general
     * an ordered subsequence that need not be contiguous.
     */
    @Test
    public void prefillSkipsTheVocabularyProjection() {
        InferenceProgram program = program();

        assertEquals(
                List.of("norm", "qProjection"),
                program.componentsFor(PhaseId.PREFILL).stream()
                        .map(ProgramComponent::name)
                        .toList());
        assertEquals(
                List.of("norm", "qProjection", "logits"),
                program.componentsFor(PhaseId.DECODE).stream()
                        .map(ProgramComponent::name)
                        .toList());
    }

    /** A phase the program does not declare is an error, not an empty list. */
    @Test
    public void anUndeclaredPhaseIsRefused() {
        ProgramSignature decodeOnly =
                new ProgramSignature(
                        ArchitectureId.of("llama"),
                        "single-token",
                        capacity(),
                        List.of(leaf("logits", projection(), DECODE_ONLY)),
                        List.of(new PhaseSelection(PhaseId.DECODE, List.of(0))),
                        bindings());
        try {
            InferenceProgram.of(decodeOnly).componentsFor(PhaseId.PREFILL);
            fail("a program without a prefill phase must say so rather than return nothing");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("PREFILL"));
        }
    }

    /** A selection may skip a component in the middle, not only at the end. */
    @Test
    public void aPhaseMaySelectANonContiguousSubsequence() {
        PhaseSelection selection = new PhaseSelection(PhaseId.PREFILL, List.of(0, 2, 5));
        assertEquals(List.of(0, 2, 5), selection.componentIndices());
    }

    /** What a selection may never do: reorder, or repeat. */
    @Test
    public void aPhaseMayNotReorderOrRepeatComponents() {
        assertRejected("reordering", () -> new PhaseSelection(PhaseId.DECODE, List.of(2, 1)));
        assertRejected("repeating", () -> new PhaseSelection(PhaseId.DECODE, List.of(1, 1)));
    }

    /** A selection cannot name a component the program did not declare. */
    @Test
    public void aPhaseMayNotSelectAComponentTheProgramDoesNotHave() {
        assertRejected(
                "selecting past the end",
                () ->
                        new ProgramSignature(
                                ArchitectureId.of("llama"),
                                "p",
                                capacity(),
                                List.of(leaf("only", projection(), DECODE_ONLY)),
                                List.of(new PhaseSelection(PhaseId.DECODE, List.of(0, 1))),
                                bindings()));
    }

    /** Nor one that does not participate in that phase. */
    @Test
    public void aPhaseMayNotSelectAComponentThatDoesNotDeclareIt() {
        assertRejected(
                "selecting a decode-only component for prefill",
                () ->
                        new ProgramSignature(
                                ArchitectureId.of("llama"),
                                "p",
                                capacity(),
                                List.of(leaf("logits", projection(), DECODE_ONLY)),
                                List.of(new PhaseSelection(PhaseId.PREFILL, List.of(0))),
                                bindings()));
    }

    /**
     * No field of any program type is an array, at any depth. Arrays compare by identity, and this
     * is a cache key.
     */
    @Test
    public void noProgramTypeHoldsAnArray() {
        List<Class<?>> types =
                new ArrayList<>(
                        List.of(
                                ProgramSignature.class,
                                InferenceProgram.class,
                                PhaseSelection.class,
                                CapacityShape.class,
                                ProgramComponent.Leaf.class,
                                ProgramComponent.Composite.class,
                                BindingEntry.ProgramFixed.class,
                                BindingEntry.InvocationValue.class,
                                BindingEntry.HostVisibleResult.class));
        List<String> offenders = new ArrayList<>();
        for (Class<?> type : types) {
            for (Field f : type.getDeclaredFields()) {
                if (f.getType().isArray()) {
                    offenders.add(type.getSimpleName() + "." + f.getName());
                }
            }
        }
        if (!offenders.isEmpty()) {
            fail(
                    "array fields compare by identity and must not appear in a cache key: "
                            + offenders);
        }
    }

    /** Structurally equal signatures are equal, and agree on hash code. */
    @Test
    public void signaturesCompareByContent() {
        assertEquals(signature(), signature());
        assertEquals(signature().hashCode(), signature().hashCode());
    }

    /** Capacity is part of identity: a program sized differently is a different program. */
    @Test
    public void capacityParticipatesInIdentity() {
        ProgramSignature wider =
                new ProgramSignature(
                        ArchitectureId.of("llama"),
                        "single-token",
                        new CapacityShape(8192, 16, 16, 512, 256, 8),
                        signature().components(),
                        signature().phases(),
                        signature().bindings());
        assertNotEquals(signature(), wider);
    }

    /** So is the architecture, and so is the policy. */
    @Test
    public void architectureAndPolicyParticipateInIdentity() {
        assertNotEquals(
                signature(),
                new ProgramSignature(
                        ArchitectureId.of("qwen3"),
                        "single-token",
                        capacity(),
                        signature().components(),
                        signature().phases(),
                        signature().bindings()));
        assertNotEquals(
                signature(),
                new ProgramSignature(
                        ArchitectureId.of("llama"),
                        "prefill-decode",
                        capacity(),
                        signature().components(),
                        signature().phases(),
                        signature().bindings()));
    }

    /**
     * Collections are copied defensively: mutating the caller's list must not reach the signature.
     */
    @Test
    public void collectionsAreCopiedDefensively() {
        List<ProgramComponent> mutable = new ArrayList<>(signature().components());
        ProgramSignature built =
                new ProgramSignature(
                        ArchitectureId.of("llama"),
                        "single-token",
                        capacity(),
                        mutable,
                        signature().phases(),
                        bindings());
        mutable.clear();
        assertEquals(
                "the signature must not change when the caller's list does",
                3,
                built.components().size());
    }

    /** Bindings must be in index order and dense from zero — the canonical form. */
    @Test
    public void bindingsMustBeCanonicallyOrdered() {
        List<BindingEntry> outOfOrder =
                List.of(
                        new BindingEntry.ProgramFixed(
                                1,
                                BindingRole.WEIGHT,
                                "w",
                                DataType.F16,
                                Shape.of(4096, 4096),
                                Direction.IN),
                        new BindingEntry.ProgramFixed(
                                0,
                                BindingRole.CONTROL,
                                "pos",
                                DataType.F32,
                                Shape.of(2),
                                Direction.IN_OUT));
        assertRejected(
                "bindings out of index order",
                () ->
                        new ProgramSignature(
                                ArchitectureId.of("llama"),
                                "p",
                                capacity(),
                                signature().components(),
                                signature().phases(),
                                outOfOrder));
    }

    /** Phases must be declared once each, in PhaseId order. */
    @Test
    public void phasesMustBeCanonicallyOrdered() {
        List<PhaseSelection> reversed =
                List.of(
                        new PhaseSelection(PhaseId.DECODE, List.of(0)),
                        new PhaseSelection(PhaseId.PREFILL, List.of(0)));
        assertRejected(
                "phases out of order",
                () ->
                        new ProgramSignature(
                                ArchitectureId.of("llama"),
                                "p",
                                capacity(),
                                List.of(leaf("norm", norm(), BOTH)),
                                reversed,
                                bindings()));
    }

    /** The three binding classes are distinguishable, and a scalar result says it is one. */
    @Test
    public void theBindingSurfaceSeparatesArraysValuesAndResults() {
        ProgramSignature signature = signature();
        assertEquals(2, signature.programFixed().size());
        assertEquals(1, signature.invocationValues().size());
        assertEquals(1, signature.results().size());
        assertTrue(
                "a sampled token is a scalar, not a float tensor",
                signature.results().getFirst().isScalar());
        assertEquals(ValueType.I32, signature.results().getFirst().scalarType());
    }

    /** A composite may not hide a child that runs in a phase the composite does not. */
    @Test
    public void aCompositeMustCoverItsChildrensPhases() {
        assertRejected(
                "a child running in a phase its parent does not",
                () ->
                        new ProgramComponent.Composite(
                                "layer", List.of(leaf("logits", projection(), BOTH)), DECODE_ONLY));
    }

    // fixtures

    private static InferenceProgram program() {
        return InferenceProgram.of(signature());
    }

    private static ProgramSignature signature() {
        return new ProgramSignature(
                ArchitectureId.of("llama"),
                "single-token",
                capacity(),
                List.of(
                        leaf("norm", norm(), BOTH),
                        leaf("qProjection", matVec(), BOTH),
                        leaf("logits", projection(), DECODE_ONLY)),
                List.of(
                        new PhaseSelection(PhaseId.PREFILL, List.of(0, 1)),
                        new PhaseSelection(PhaseId.DECODE, List.of(0, 1, 2))),
                bindings());
    }

    private static List<BindingEntry> bindings() {
        return List.of(
                new BindingEntry.ProgramFixed(
                        0,
                        BindingRole.WEIGHT,
                        "wq",
                        DataType.F16,
                        Shape.of(4096, 4096),
                        Direction.IN),
                new BindingEntry.ProgramFixed(
                        1,
                        BindingRole.CONTROL,
                        "control",
                        DataType.F32,
                        Shape.of(2),
                        Direction.IN_OUT),
                new BindingEntry.InvocationValue(2, ValueId.POSITION, ValueType.I32, 1, 0, 1),
                new BindingEntry.HostVisibleResult(
                        3, ResultId.SAMPLED_TOKEN, ValueType.I32, 1, 1, 1));
    }

    private static CapacityShape capacity() {
        return new CapacityShape(4096, 16, 16, 512, 256, 8);
    }

    private static ProgramComponent leaf(
            String name, org.beehive.jllm.program.op.Operation operation, Set<PhaseId> phases) {
        return new ProgramComponent.Leaf(name, operation, phases);
    }

    private static RmsNorm norm() {
        return new RmsNorm(X, Optional.of(NORM), XB, 1e-5f, DataType.F32);
    }

    private static MatVec matVec() {
        return new MatVec(WEIGHT, XB, X, 4096, 4096, DataType.F16);
    }

    private static VocabProjection projection() {
        return new VocabProjection(OUTPUT, X, LOGITS, 128256, DataType.F16);
    }

    private static void assertRejected(String why, Runnable construction) {
        try {
            construction.run();
            fail("must be refused: " + why);
        } catch (IllegalArgumentException expected) {
            // the contract
        }
    }
}
