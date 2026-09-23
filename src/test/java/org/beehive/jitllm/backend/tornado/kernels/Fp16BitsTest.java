package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;

/**
 * {@link TransformerBatchPrefillKernels#fp16BitsOf} against the JDK's own conversion.
 *
 * <p>The kernel helper exists because the PTX backend cannot lower {@code new
 * HalfFloat(v).getHalfFloatValue()} inside the tensor-core kernels. It replaces a conversion the
 * platform used to perform, so it has to round identically: round-to-nearest-even, subnormals
 * included. A helper that merely truncated would bias every Q8_0 weight in the same direction and
 * cost the model its CPU parity without failing to compile.
 */
public class Fp16BitsTest {

    @Test
    public void roundsIdenticallyToFloatToFloat16() {
        Random random = new Random(7);
        for (int i = 0; i < 2_000_000; i++) {
            float v = sample(random, i & 3);
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                continue;
            }
            int expected = Float.floatToFloat16(v) & 0xFFFF;
            if (expected == 0x7C00 || expected == 0xFC00) {
                continue; // out of range for a dequantized weight; the helper saturates instead
            }
            assertEquals(
                    "fp16 bits for " + v, expected, TransformerBatchPrefillKernels.fp16BitsOf(v));
        }
    }

    @Test
    public void saturatesRatherThanOverflowing() {
        assertEquals(0x7BFF, TransformerBatchPrefillKernels.fp16BitsOf(1e30f));
        assertEquals(0xFBFF, TransformerBatchPrefillKernels.fp16BitsOf(-1e30f));
    }

    @Test
    public void handlesZeroAndTheSubnormalBoundary() {
        assertEquals(0x0000, TransformerBatchPrefillKernels.fp16BitsOf(0.0f));
        // Negative zero comes back as positive zero, as the helper documents: no Q8_0 product
        // reaches it, and the two are interchangeable in the GEMM that consumes these bits.
        assertEquals(0x0000, TransformerBatchPrefillKernels.fp16BitsOf(-0.0f));
        // Exactly half the smallest subnormal: a tie, and zero is the even neighbour.
        assertEquals(0x0000, TransformerBatchPrefillKernels.fp16BitsOf(2.9802322E-8f));
        assertEquals(0x0001, TransformerBatchPrefillKernels.fp16BitsOf(5.9604645E-8f));
    }

    /** Weight-shaped values, wide values, tiny values, and real dequantized Q8_0 products. */
    private static float sample(Random random, int kind) {
        switch (kind) {
            case 0:
                return (random.nextFloat() - 0.5f) * 4f;
            case 1:
                return (random.nextFloat() - 0.5f) * 130000f;
            case 2:
                return (random.nextFloat() - 0.5f) * 1e-6f;
            default:
                float scale =
                        Float.float16ToFloat(Float.floatToFloat16(random.nextFloat() * 0.05f));
                return (random.nextInt(255) - 127) * scale;
        }
    }
}
