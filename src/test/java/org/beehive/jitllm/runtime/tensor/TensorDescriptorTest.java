package org.beehive.jitllm.runtime.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TensorDescriptorTest {

    private static final TensorLayout Q8_0_LAYOUT =
            new TensorLayout.BlockQuantized(32, 34, TensorLayout.ScaleFormat.FP16);

    private static TensorDescriptor descriptor(String name, TensorRole role, long... dimensions) {
        return new TensorDescriptor(name, DataType.Q8_0, Shape.of(dimensions), role, Q8_0_LAYOUT);
    }

    // ── Shape ─────────────────────────────────────────────────────────────────

    /**
     * The reason dimensions are kept at all: these two hold the same number of elements and are not
     * interchangeable, and an element count cannot tell them apart.
     */
    @Test
    public void transposedShapesAreNotEqual() {
        assertNotEquals(Shape.of(4096, 11008), Shape.of(11008, 4096));
        assertEquals(Shape.of(4096, 11008).elementCount(), Shape.of(11008, 4096).elementCount());
    }

    @Test
    public void theElementCountIsTheProductOfTheDimensions() {
        assertEquals(45088768L, Shape.of(4096, 11008).elementCount());
        assertEquals(4096L, Shape.of(4096).elementCount());
        assertEquals(24L, Shape.of(2, 3, 4).elementCount());
    }

    @Test
    public void dimensionOrderIsPreservedRatherThanNormalized() {
        Shape shape = Shape.of(4096, 11008);
        assertEquals(
                "dimension 0 is the fastest-varying one, as GGUF states it",
                4096,
                shape.dimension(0));
        assertEquals(11008, shape.dimension(1));
        assertEquals(2, shape.rank());
    }

    @Test
    public void aShapeDoesNotShareItsArrayWithAnyone() {
        long[] dimensions = {4096, 11008};
        Shape shape = Shape.of(dimensions);
        dimensions[0] = 1;
        assertEquals(4096, shape.dimension(0));
        shape.dimensions()[1] = 1;
        assertEquals(11008, shape.dimension(1));
    }

    @Test
    public void nonsensicalShapesAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(new long[0]));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(4096, 0));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(-1));
    }

    /** A product that overflows a long is an error, not a negative element count. */
    @Test
    public void anOverflowingProductIsRefusedRatherThanWrapped() {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> Shape.of(Long.MAX_VALUE, 2));
        assertTrue(failure.getMessage(), failure.getMessage().contains("overflow"));
    }

    /**
     * The count is a long because that is the honest number; the int-indexed check belongs where
     * storage is actually allocated, and it names the tensor so the failure is actionable.
     */
    @Test
    public void aTensorTooLargeForAnIntIndexedArrayFailsByName() {
        Shape huge = Shape.of(4_000_000_000L);
        assertEquals(4_000_000_000L, huge.elementCount());
        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> huge.elementCountAsInt("blk.0.ffn_up.weight"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("blk.0.ffn_up.weight"));
    }

    // ── Layout ────────────────────────────────────────────────────────────────

    @Test
    public void aBlockLayoutSizesByBlocksAndRoundsUp() {
        assertEquals(34L, Q8_0_LAYOUT.byteSize(32));
        assertEquals(68L, Q8_0_LAYOUT.byteSize(33)); // a partial block still costs a block
        assertEquals(34L * 128, Q8_0_LAYOUT.byteSize(32 * 128));
    }

    @Test
    public void aDenseLayoutSizesByValue() {
        assertEquals(4096L * 4, new TensorLayout.Dense(4).byteSize(4096));
        assertEquals(4096L * 2, new TensorLayout.Dense(2).byteSize(4096));
    }

    @Test
    public void anEmptyBlockIsRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TensorLayout.BlockQuantized(0, 34, TensorLayout.ScaleFormat.FP16));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout.Dense(0));
    }

    // ── Descriptor ────────────────────────────────────────────────────────────

    @Test
    public void aDescriptorSaysWhatATensorIsAndHowBigItIs() {
        TensorDescriptor wq =
                descriptor("blk.0.attn_q.weight", TensorRole.ATTENTION_QUERY, 2048, 2048);
        assertEquals(DataType.Q8_0, wq.dataType());
        assertEquals(TensorRole.ATTENTION_QUERY, wq.role());
        assertEquals(2048L * 2048, wq.elementCount());
        assertEquals(Q8_0_LAYOUT.byteSize(2048L * 2048), wq.byteSize());
    }

    /**
     * The validation flat tensors cannot do. Two weights of the same size and different shape are
     * exactly the mix-up that produces plausible garbage rather than an error.
     */
    @Test
    public void aMisboundWeightIsAnErrorRatherThanAWrongAnswer() {
        TensorDescriptor expected = descriptor("expected", TensorRole.FFN_UP, 4096, 11008);
        TensorDescriptor transposed =
                descriptor("blk.0.ffn_up.weight", TensorRole.FFN_UP, 11008, 4096);

        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> transposed.requireCompatibleWith(expected));
        assertTrue(failure.getMessage(), failure.getMessage().contains("blk.0.ffn_up.weight"));

        descriptor("blk.0.ffn_up.weight", TensorRole.FFN_UP, 4096, 11008)
                .requireCompatibleWith(expected);
    }

    @Test
    public void aDataTypeMismatchIsCaughtToo() {
        TensorDescriptor expected = descriptor("expected", TensorRole.OUTPUT, 4096);
        TensorDescriptor wrongType =
                new TensorDescriptor(
                        "output.weight",
                        DataType.F16,
                        Shape.of(4096),
                        TensorRole.OUTPUT,
                        new TensorLayout.Dense(2));
        assertThrows(
                IllegalArgumentException.class, () -> wrongType.requireCompatibleWith(expected));
    }
}
