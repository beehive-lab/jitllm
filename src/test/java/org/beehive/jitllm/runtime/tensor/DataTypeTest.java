package org.beehive.jllm.runtime.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.EnumSet;
import java.util.Set;
import org.junit.Test;

public class DataTypeTest {

    /**
     * The value set is the decision. A new constant appearing here without a runtime that stores or
     * computes with it is the drift this test exists to catch — that is how an execution vocabulary
     * turns back into a copy of the file format's.
     */
    @Test
    public void theValueSetIsWhatTheRuntimeExecutes() {
        assertEquals(
                EnumSet.of(
                        DataType.F32,
                        DataType.F16,
                        DataType.BF16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                EnumSet.allOf(DataType.class));
    }

    /**
     * The only narrowing left is a real one.
     *
     * <p>This replaces three tests that asserted the opposite: that every block quantization
     * "materializes as Q8_0", that none of them could be a materialization target, and that the
     * device could execute none of them. All three encoded one premise — that a quantized weight
     * cannot reach an accelerator in its own layout — and it was wrong. It is what made a 4-bit
     * model occupy twice its size on a device.
     *
     * <p>{@link DataType#BF16} still narrows, because that is a loss of mantissa bits for want of
     * BF16 arithmetic: a property of the representation, not a capability gap dressed up as one.
     */
    @Test
    public void onlyBf16Narrows() {
        assertSame(DataType.F16, DataType.BF16.narrowedFallback());
        for (DataType type : DataType.values()) {
            if (type != DataType.BF16) {
                assertSame(
                        type + " must be kept as it is, not converted",
                        type,
                        type.narrowedFallback());
            }
        }
    }

    @Test
    public void quantizationIsAPropertyOfTheRepresentation() {
        assertFalse(DataType.F32.isQuantized());
        assertFalse(DataType.F16.isQuantized());
        assertTrue(DataType.Q8_0.isQuantized());
        assertTrue(DataType.Q4_K.isQuantized());
    }

    /**
     * Block size and scale layout belong to {@code TensorLayout}. If they ever appear here, {@code
     * Q8_0} with one scale arrangement becomes a different data type from {@code Q8_0} with
     * another, and operations can no longer be parameterized by dtype alone.
     */
    @Test
    public void theTypeCarriesNoStorageDetail() {
        Set<String> methods = new java.util.TreeSet<>();
        for (java.lang.reflect.Method method : DataType.class.getDeclaredMethods()) {
            if (method.getDeclaringClass() == DataType.class && !method.isSynthetic()) {
                methods.add(method.getName());
            }
        }
        assertEquals(Set.of("values", "valueOf", "isQuantized", "narrowedFallback"), methods);
    }

    /**
     * Nothing narrows to something that is itself narrowed.
     *
     * <p>What this used to say was that no fallback is "format-decoded", which was a statement
     * about storage. The property worth keeping is the fixed point: applying the narrowing twice
     * changes nothing, so a caller never has to chase a chain.
     */
    @Test
    public void narrowingIsAFixedPoint() {
        for (DataType type : DataType.values()) {
            DataType once = type.narrowedFallback();
            assertSame(
                    type + " narrows to something that narrows again",
                    once,
                    once.narrowedFallback());
        }
    }
}
