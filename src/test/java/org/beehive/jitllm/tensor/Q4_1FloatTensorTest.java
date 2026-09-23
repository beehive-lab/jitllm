package org.beehive.jitllm.tensor;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tensor.standard.Q4_1FloatTensor;
import org.junit.Test;

/**
 * Q4_1 against the format it claims to decode.
 *
 * <p>The reconstruction is {@code d * q + m} with an <b>unsigned</b> nibble, where Q4_0 is {@code d
 * * (q - 8)}. Getting the recentering wrong is a plausible mistake that shifts every weight in a
 * block by a constant — which a dot product against a mean-zero activation barely shows, and which
 * this test pins directly instead.
 */
public class Q4_1FloatTensorTest {

    private static final int BLOCK = GGMLType.Q4_1.getBlockSize(); // 32
    private static final int TYPE = GGMLType.Q4_1.getTypeSize(); // 20

    /** Packs {@code nibbles} into Q4_1 blocks with the given per-block scale and minimum. */
    private static MemorySegment encode(int[] nibbles, float[] scales, float[] mins, Arena arena) {
        int blocks = nibbles.length / BLOCK;
        MemorySegment segment = arena.allocate((long) blocks * TYPE);
        for (int b = 0; b < blocks; b++) {
            int base = b * TYPE;
            segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, base, Float.floatToFloat16(scales[b]));
            segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, base + 2, Float.floatToFloat16(mins[b]));
            // Element i < 16 goes in the low nibble of byte i; element i >= 16 in the high nibble
            // of byte i - 16. Same packing as Q4_0.
            for (int i = 0; i < BLOCK / 2; i++) {
                int lo = nibbles[b * BLOCK + i] & 0xF;
                int hi = nibbles[b * BLOCK + i + BLOCK / 2] & 0xF;
                segment.set(ValueLayout.JAVA_BYTE, base + 4 + i, (byte) (lo | (hi << 4)));
            }
        }
        return segment;
    }

    @Test
    public void reconstructsScaleTimesNibblePlusMinimum() {
        try (Arena arena = Arena.ofConfined()) {
            int[] nibbles = new int[BLOCK];
            for (int i = 0; i < BLOCK; i++) {
                nibbles[i] = i % 16;
            }
            float[] scales = {0.25f};
            float[] mins = {-1.5f};
            Q4_1FloatTensor tensor =
                    new Q4_1FloatTensor(BLOCK, encode(nibbles, scales, mins, arena));

            for (int i = 0; i < BLOCK; i++) {
                float expected =
                        Float.float16ToFloat(Float.floatToFloat16(0.25f)) * nibbles[i]
                                + Float.float16ToFloat(Float.floatToFloat16(-1.5f));
                assertEquals("element " + i, expected, tensor.getFloat(i), 1e-6f);
            }
        }
    }

    /**
     * The vectorized dot product must agree with reading the tensor element by element.
     *
     * <p>Several blocks with different scales and minimums, and a length that is not a whole number
     * of vector lanes, so the alignment prologue and the scalar tail both run.
     */
    @Test
    public void vectorizedDotAgreesWithElementwise() {
        try (Arena arena = Arena.ofConfined()) {
            Random random = new Random(4242L);
            int blocks = 5;
            int size = blocks * BLOCK;
            int[] nibbles = new int[size];
            for (int i = 0; i < size; i++) {
                nibbles[i] = random.nextInt(16);
            }
            float[] scales = new float[blocks];
            float[] mins = new float[blocks];
            for (int b = 0; b < blocks; b++) {
                scales[b] = 0.05f + random.nextFloat() * 0.2f;
                mins[b] = (random.nextFloat() - 0.5f);
            }
            Q4_1FloatTensor weights =
                    new Q4_1FloatTensor(size, encode(nibbles, scales, mins, arena));

            float[] raw = new float[size];
            for (int i = 0; i < size; i++) {
                raw[i] = (float) random.nextGaussian();
            }
            FloatTensor activations = new ArrayFloatTensor(raw);

            for (int[] span : new int[][] {{0, size}, {32, 96}, {0, 100}, {64, 33}}) {
                int offset = span[0];
                int length = span[1];
                double expected = 0;
                for (int i = 0; i < length; i++) {
                    expected += (double) weights.getFloat(offset + i) * raw[offset + i];
                }
                float actual = weights.dot(offset, activations, offset, length);
                assertEquals(
                        "dot at " + offset + " length " + length,
                        (float) expected,
                        actual,
                        Math.max(1e-4f, Math.abs((float) expected) * 1e-5f));
            }
        }
    }
}
