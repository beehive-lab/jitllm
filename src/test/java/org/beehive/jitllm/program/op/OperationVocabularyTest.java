package org.beehive.jitllm.program.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import org.junit.Test;

/**
 * Value behaviour is not a style preference here. An operation ends up inside a {@code
 * ProgramSignature}, which is the compiled-program cache key, so a field that compares by identity
 * is a silent cache miss that recompiles the same program — or a claimed match that is not one The
 * array check below is the one that would have caught it.
 */
public class OperationVocabularyTest {

    private static final OperandRef.Weight W =
            new OperandRef.Weight("w", TensorRole.ATTENTION_QUERY);
    private static final OperandRef A = new OperandRef.Activation("a");
    private static final OperandRef B = new OperandRef.Activation("b");
    private static final OperandRef C = new OperandRef.Activation("c");
    private static final OperandRef IDS = new OperandRef.Activation("ids");
    private static final OperandRef HU = new OperandRef.Activation("hiddenUp");
    private static final OperandRef EO = new OperandRef.Activation("expertOut");

    private static List<Operation> everyOperation() {
        return List.of(
                new RmsNorm(A, Optional.of(W), B, 1e-5f, DataType.F32),
                new RoPE(
                        A, B, Optional.empty(), 128, 500000f, RopeLayout.INTERLEAVED, DataType.F32),
                new MatVec(W, A, B, 4096, 4096, DataType.Q8_0),
                new MatMul(W, A, B, 4096, 4096, 8, DataType.F16),
                new KvAppend(A, B, C, IDS, 1024, DataType.F16),
                new Attention(
                        A,
                        B,
                        C,
                        new OperandRef.Activation("d"),
                        32,
                        8,
                        128,
                        0.088f,
                        OptionalInt.empty(),
                        DataType.F16),
                new Softmax(A, B, 32, 512, DataType.F32),
                new SwiGLU(A, B, C, DataType.F32),
                new L2Norm(A, B, 16, 128, 1e-6f, DataType.F32),
                new CausalConv1d(A, W, B, C, 10240, 4, DataType.F32),
                new DeltaRuleUpdate(
                        A,
                        B,
                        C,
                        IDS,
                        HU,
                        EO,
                        new OperandRef.Activation("out"),
                        48,
                        16,
                        128,
                        DataType.F32),
                new GatedNorm(A, B, W, C, 48, 128, 1e-6f, DataType.F32),
                new GeGLU(A, B, C, DataType.F32),
                new ResidualAdd(A, B, C, DataType.F32),
                new BiasAdd(A, W, B, DataType.F32),
                new Scale(A, B, 8.0f, DataType.F32),
                new SplitFusedQkv(
                        A, B, C, new OperandRef.Activation("v"), 3072, 1024, DataType.F32),
                new SplitGateUp(A, B, C, 8192, DataType.F32),
                new MoeRouter(
                        A,
                        W,
                        B,
                        C,
                        new OperandRef.Activation("ids"),
                        8,
                        2,
                        MoeRouter.RouterNormalization.SOFTMAX_OVER_ALL_EXPERTS,
                        DataType.F32),
                new ExpertFeedForward(
                        A,
                        0,
                        B,
                        W,
                        W,
                        W,
                        C,
                        new OperandRef.Activation("hu"),
                        new OperandRef.Activation("eo"),
                        32,
                        64,
                        DataType.Q8_0),
                new WeightedAccumulate(
                        A, B, C, 0, WeightedAccumulate.GateActivation.NONE, 64, DataType.F32),
                new EmbeddingLookup(W, A, B, 4096, DataType.F16),
                new VocabProjection(W, A, B, 128256, DataType.Q8_0),
                new LogitSoftCap(A, B, 30.0f, DataType.F32),
                new ArgMax(A, B, DataType.F32),
                new Sample(A, B, DataType.F32));
    }

    /**
     * Qwen3 normalizes query and key per head. Emitting one leaf per head would say "normalize the
     * whole vector" N times — a different computation that would still look like a program.
     */
    @Test
    public void aGroupedNormStatesItsGroupsAndIsRefusedWithoutALength() {
        RmsNorm ungrouped = new RmsNorm(A, Optional.of(W), B, 1e-5f, DataType.F32);
        assertEquals(1, ungrouped.groups());
        assertFalse(ungrouped.isGrouped());

        RmsNorm perHead = new RmsNorm(A, Optional.of(W), B, 1e-5f, DataType.F32, 16, 128);
        assertTrue(perHead.isGrouped());
        assertEquals(16, perHead.groups());
        assertEquals(128, perHead.groupLength());
        assertNotEquals("grouping is part of the operation's identity", ungrouped, perHead);

        try {
            new RmsNorm(A, Optional.of(W), B, 1e-5f, DataType.F32, 16, 0);
            fail("a grouped norm with no group length must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("group length"));
        }
    }

    @Test
    public void theAttentionScoreScalingDirectionIsPartOfTheOperation() {
        Attention divide =
                new Attention(
                        A,
                        B,
                        C,
                        new OperandRef.Activation("d"),
                        32,
                        8,
                        128,
                        11.3137f,
                        OptionalInt.empty(),
                        DataType.F16);
        Attention multiply =
                new Attention(
                        A,
                        B,
                        C,
                        new OperandRef.Activation("d"),
                        32,
                        8,
                        128,
                        11.3137f,
                        Attention.ScoreScaling.MULTIPLY,
                        OptionalInt.empty(),
                        DataType.F16);

        assertEquals(
                "the conventional constructor divides",
                Attention.ScoreScaling.DIVIDE,
                divide.scoreScaling());
        assertNotEquals(
                "the same number applied the other way is a different operation", divide, multiply);
    }

    @Test
    public void theGateUpSplitStatesItsWidth() {
        SplitGateUp split = new SplitGateUp(A, B, C, 8192, DataType.F32);
        assertEquals("the projection this reads must produce 2 x width", 16384, split.fusedWidth());
        assertEquals(List.of(A), split.inputs());
        assertEquals(2, split.outputs().size());
    }

    @Test
    public void theFusedQkvSplitStatesItsWidths() {
        SplitFusedQkv split =
                new SplitFusedQkv(
                        A, B, C, new OperandRef.Activation("v"), 3072, 1024, DataType.F32);
        assertEquals("the projection this reads must produce q + 2·kv", 5120, split.fusedWidth());
        assertEquals(List.of(A), split.inputs());
        assertEquals(3, split.outputs().size());
    }

    /**
     * Every kind is realized by exactly one operation type, and no operation exists outside the
     * closed set. A kind with no type is a name nothing can use; two types on one kind is a second
     * dispatch key, which is what Rule 15 exists to prevent.
     */
    @Test
    public void everyKindHasExactlyOneOperationType() {
        Set<OperationKind> seen = EnumSet.noneOf(OperationKind.class);
        for (Operation op : everyOperation()) {
            assertTrue(
                    "two operation types claim "
                            + op.kind()
                            + "; the second is "
                            + op.getClass().getSimpleName(),
                    seen.add(op.kind()));
        }
        Set<OperationKind> missing = EnumSet.allOf(OperationKind.class);
        missing.removeAll(seen);
        assertTrue("OperationKind values with no operation type: " + missing, missing.isEmpty());
    }

    /** The vocabulary is closed: {@code Operation} permits exactly the types exercised here. */
    @Test
    public void theOperationHierarchyIsSealedAndFullyCovered() {
        assertTrue("Operation must be sealed", Operation.class.isSealed());
        Set<String> permitted = new TreeSet<>();
        for (Class<?> c : Operation.class.getPermittedSubclasses()) {
            permitted.add(c.getName());
        }
        Set<String> exercised = new TreeSet<>();
        for (Operation op : everyOperation()) {
            exercised.add(op.getClass().getName());
        }
        assertEquals("every permitted operation type must be exercised", permitted, exercised);
    }

    /**
     * No field of any operation is an array. Java arrays compare by identity, so one here would
     * break signature equality without breaking any test that only compared an operation to itself.
     */
    @Test
    public void noOperationHoldsAnArray() {
        List<String> offenders = new ArrayList<>();
        List<Class<?>> types =
                new ArrayList<>(List.of(OperandRef.Weight.class, OperandRef.Activation.class));
        for (Class<?> c : Operation.class.getPermittedSubclasses()) {
            types.add(c);
        }
        for (Class<?> type : types) {
            for (Field f : type.getDeclaredFields()) {
                if (f.getType().isArray()) {
                    offenders.add(type.getSimpleName() + "." + f.getName());
                }
            }
        }
        if (!offenders.isEmpty()) {
            fail(
                    "array fields compare by identity and must not appear in the vocabulary: "
                            + offenders);
        }
    }

    /** Structurally equal operations are equal, and agree on hash code. */
    @Test
    public void operationsCompareByContent() {
        for (Operation op : everyOperation()) {
            Operation twin = rebuild(op);
            assertEquals(op.getClass().getSimpleName() + " must compare by content", op, twin);
            assertEquals(
                    op.getClass().getSimpleName() + " hash codes must agree",
                    op.hashCode(),
                    twin.hashCode());
        }
    }

    /** Operations differing in configuration are different operations, not the same one. */
    @Test
    public void configurationParticipatesInEquality() {
        assertNotEquals(
                new RmsNorm(A, Optional.of(W), B, 1e-5f, DataType.F32),
                new RmsNorm(A, Optional.of(W), B, 1e-6f, DataType.F32));
        assertNotEquals(
                new MatVec(W, A, B, 4096, 4096, DataType.Q8_0),
                new MatVec(W, A, B, 4096, 11008, DataType.Q8_0));
        assertNotEquals(
                new RoPE(
                        A, B, Optional.empty(), 128, 500000f, RopeLayout.INTERLEAVED, DataType.F32),
                new RoPE(
                        A, B, Optional.empty(), 128, 10000f, RopeLayout.INTERLEAVED, DataType.F32));
        assertNotEquals(
                "the gate activation distinguishes a routing weight from a gate score",
                new WeightedAccumulate(
                        A, B, C, 0, WeightedAccumulate.GateActivation.NONE, 64, DataType.F32),
                new WeightedAccumulate(
                        A, B, C, 0, WeightedAccumulate.GateActivation.LOGISTIC, 64, DataType.F32));
        assertNotEquals(
                "the scale factor is model configuration and part of the operation",
                new Scale(A, B, 8.0f, DataType.F32),
                new Scale(A, B, 4.0f, DataType.F32));
        assertNotEquals(
                "the router's expert count is part of the operation",
                new MoeRouter(
                        A,
                        W,
                        B,
                        C,
                        IDS,
                        8,
                        2,
                        MoeRouter.RouterNormalization.SOFTMAX_OVER_ALL_EXPERTS,
                        DataType.F32),
                new MoeRouter(
                        A,
                        W,
                        B,
                        C,
                        IDS,
                        16,
                        2,
                        MoeRouter.RouterNormalization.SOFTMAX_OVER_ALL_EXPERTS,
                        DataType.F32));
        assertNotEquals(
                "the rotary layout is not decoration: both read valid floats",
                new RoPE(
                        A, B, Optional.empty(), 128, 500000f, RopeLayout.INTERLEAVED, DataType.F32),
                new RoPE(A, B, Optional.empty(), 128, 500000f, RopeLayout.NEOX_HALF, DataType.F32));
        assertNotEquals(
                "greedy and stochastic selection are different operations",
                (Operation) new ArgMax(A, B, DataType.F32),
                (Operation) new Sample(A, B, DataType.F32));
    }

    /**
     * The representation is part of what an operation is. Two matrix-vector products over the same
     * operands at different precisions are different operations, and must reach different
     * implementations.
     */
    @Test
    public void theRepresentationParticipatesInEquality() {
        assertNotEquals(
                new MatVec(W, A, B, 4096, 4096, DataType.F16),
                new MatVec(W, A, B, 4096, 4096, DataType.Q8_0));
        for (Operation op : everyOperation()) {
            assertTrue(op.kind() + " must state a representation", op.dataType() != null);
        }
    }

    /**
     * Operand lists are immutable, so a caller cannot rewrite an operation's shape after the fact.
     */
    @Test
    public void operandListsAreImmutable() {
        for (Operation op : everyOperation()) {
            assertImmutable(op.getClass().getSimpleName() + ".inputs()", op.inputs());
            assertImmutable(op.getClass().getSimpleName() + ".outputs()", op.outputs());
        }
    }

    /** Every operation names at least one input and one output; a no-op is not vocabulary. */
    @Test
    public void everyOperationHasOperands() {
        for (Operation op : everyOperation()) {
            assertTrue(op.kind() + " must read something", !op.inputs().isEmpty());
            assertTrue(op.kind() + " must write something", !op.outputs().isEmpty());
        }
    }

    /**
     * In-place operations say so by naming the same operand on both sides. RoPE is the case: it
     * rotates the query and key projections where they lie.
     */
    @Test
    public void inPlaceOperationsRepeatTheirOperands() {
        RoPE rope =
                new RoPE(
                        A, B, Optional.empty(), 128, 500000f, RopeLayout.INTERLEAVED, DataType.F32);
        assertEquals(List.of(A, B), rope.outputs());
        assertTrue(rope.inputs().containsAll(rope.outputs()));
    }

    /** Configuration that cannot describe real work is refused at construction. */
    @Test
    public void impossibleConfigurationIsRejected() {
        assertRejected(() -> new MatVec(W, A, B, 0, 4096, DataType.Q8_0));
        assertRejected(() -> new MatMul(W, A, B, 4096, 4096, 0, DataType.F16));
        assertRejected(
                () -> new Attention(A, B, C, A, 32, 0, 128, 1f, OptionalInt.empty(), DataType.F16));
        assertRejected(
                "query heads must divide evenly among key/value heads",
                () -> new Attention(A, B, C, A, 32, 7, 128, 1f, OptionalInt.empty(), DataType.F16));
        assertRejected(() -> new VocabProjection(W, A, B, 0, DataType.Q8_0));
        assertRejected(() -> new EmbeddingLookup(W, A, B, -1, DataType.F16));
        assertRejected(
                () ->
                        new RoPE(
                                A,
                                B,
                                Optional.empty(),
                                0,
                                500000f,
                                RopeLayout.INTERLEAVED,
                                DataType.F32));
        assertRejected(
                "a model without soft-capping omits the operation rather than passing 0",
                () -> new LogitSoftCap(A, B, 0f, DataType.F32));
        assertRejected(
                "full attention is the empty window, not a zero-length one",
                () -> new Attention(A, B, C, A, 32, 8, 128, 1f, OptionalInt.of(0), DataType.F16));
    }

    private static Operation rebuild(Operation op) {
        return switch (op) {
            case RmsNorm o ->
                    new RmsNorm(o.input(), o.weight(), o.output(), o.epsilon(), o.dataType());
            case L2Norm o ->
                    new L2Norm(
                            o.input(),
                            o.output(),
                            o.groups(),
                            o.groupLength(),
                            o.epsilon(),
                            o.dataType());
            case CausalConv1d o ->
                    new CausalConv1d(
                            o.input(),
                            o.weight(),
                            o.window(),
                            o.output(),
                            o.channels(),
                            o.kernel(),
                            o.dataType());
            case DeltaRuleUpdate o ->
                    new DeltaRuleUpdate(
                            o.query(),
                            o.key(),
                            o.value(),
                            o.decay(),
                            o.beta(),
                            o.state(),
                            o.output(),
                            o.valueHeads(),
                            o.keyHeads(),
                            o.stateDim(),
                            o.dataType());
            case GatedNorm o ->
                    new GatedNorm(
                            o.input(),
                            o.gate(),
                            o.weight(),
                            o.output(),
                            o.groups(),
                            o.groupLength(),
                            o.epsilon(),
                            o.dataType());
            case RoPE o ->
                    new RoPE(
                            o.query(),
                            o.key(),
                            o.frequencies(),
                            o.headDimension(),
                            o.ropeTheta(),
                            o.layout(),
                            o.dataType());
            case MatVec o ->
                    new MatVec(
                            o.weight(), o.input(), o.output(), o.rows(), o.columns(), o.dataType());
            case MatMul o ->
                    new MatMul(
                            o.weight(),
                            o.input(),
                            o.output(),
                            o.rows(),
                            o.columns(),
                            o.batchSize(),
                            o.dataType());
            case KvAppend o ->
                    new KvAppend(
                            o.key(),
                            o.value(),
                            o.keyStore(),
                            o.valueStore(),
                            o.width(),
                            o.dataType());
            case Attention o ->
                    new Attention(
                            o.query(),
                            o.keys(),
                            o.values(),
                            o.output(),
                            o.heads(),
                            o.keyValueHeads(),
                            o.headDimension(),
                            o.scale(),
                            o.window(),
                            o.dataType());
            case Softmax o ->
                    new Softmax(o.input(), o.output(), o.rows(), o.columns(), o.dataType());
            case SwiGLU o -> new SwiGLU(o.gate(), o.up(), o.output(), o.dataType());
            case GeGLU o -> new GeGLU(o.gate(), o.up(), o.output(), o.dataType());
            case LogitSoftCap o -> new LogitSoftCap(o.input(), o.output(), o.cap(), o.dataType());
            case ResidualAdd o -> new ResidualAdd(o.left(), o.right(), o.output(), o.dataType());
            case BiasAdd o -> new BiasAdd(o.input(), o.bias(), o.output(), o.dataType());
            case Scale o -> new Scale(o.input(), o.output(), o.factor(), o.dataType());
            case SplitFusedQkv o ->
                    new SplitFusedQkv(
                            o.fused(),
                            o.query(),
                            o.key(),
                            o.value(),
                            o.queryWidth(),
                            o.keyValueWidth(),
                            o.dataType());
            case SplitGateUp o ->
                    new SplitGateUp(o.fused(), o.gate(), o.up(), o.width(), o.dataType());
            case MoeRouter o ->
                    new MoeRouter(
                            o.input(),
                            o.routerWeight(),
                            o.scores(),
                            o.selectedIds(),
                            o.selectedWeights(),
                            o.numberOfExperts(),
                            o.topK(),
                            o.normalization(),
                            o.dataType());
            case ExpertFeedForward o ->
                    new ExpertFeedForward(
                            o.input(),
                            o.expertIndex(),
                            o.selectedIds(),
                            o.gateWeights(),
                            o.upWeights(),
                            o.downWeights(),
                            o.hidden(),
                            o.hiddenUp(),
                            o.output(),
                            o.expertHiddenDim(),
                            o.modelDim(),
                            o.dataType());
            case WeightedAccumulate o ->
                    new WeightedAccumulate(
                            o.stream(),
                            o.branch(),
                            o.weightSource(),
                            o.weightIndex(),
                            o.gate(),
                            o.length(),
                            o.dataType());
            case EmbeddingLookup o ->
                    new EmbeddingLookup(
                            o.table(),
                            o.tokenIds(),
                            o.output(),
                            o.embeddingDimension(),
                            o.dataType());
            case VocabProjection o ->
                    new VocabProjection(
                            o.weight(), o.input(), o.output(), o.vocabularySize(), o.dataType());
            case ArgMax o -> new ArgMax(o.logits(), o.output(), o.dataType());
            case Sample o -> new Sample(o.logits(), o.output(), o.dataType());
        };
    }

    private static void assertImmutable(String what, List<OperandRef> list) {
        try {
            list.add(new OperandRef.Activation("intruder"));
            fail(what + " must be immutable");
        } catch (UnsupportedOperationException expected) {
            // the contract
        }
    }

    private static void assertRejected(Runnable construction) {
        assertRejected("construction must be refused", construction);
    }

    private static void assertRejected(String why, Runnable construction) {
        try {
            construction.run();
            fail(why);
        } catch (IllegalArgumentException expected) {
            // the contract
        }
    }
}
