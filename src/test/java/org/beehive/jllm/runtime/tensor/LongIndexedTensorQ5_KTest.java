package org.beehive.jllm.runtime.tensor;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jllm.tensor.standard.Q5_KFloatTensor;
import org.junit.Test;

/**
 * The long-indexed Q5_K reader against {@link Q5_KFloatTensor}, which is the accepted host decoder
 * for this format.
 *
 * <p>Two implementations agreeing is necessary but not sufficient — they could agree and both be a
 * different format — so the bytes are adversarial rather than plausible: every scale/min nibble
 * combination the 12-byte packing can produce is exercised by filling that field with a counter,
 * and the quant fields take all-zero, all-ones, alternating and random patterns. What is asserted
 * is <b>bit identity</b>, because this change is meant to move no arithmetic at all.
 */
public class LongIndexedTensorQ5_KTest {

    private static final int QK_K = 256;
    private static final int BLOCK_BYTES = 176;

    @Test
    public void readsTheSameValuesAsTheHostDecoder() {
        int blocks = 64;
        int elements = blocks * QK_K;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate((long) blocks * BLOCK_BYTES);
            Random rnd = new Random(20260917L);
            for (int b = 0; b < blocks; b++) {
                long off = (long) b * BLOCK_BYTES;
                // d and dmin: alternate sign, subnormal and normal magnitudes.
                seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, off, (short) (b * 517 + 1));
                seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, off + 2, (short) (0x8000 | (b * 311)));
                // 12 scale/min bytes: a counter, so every 6-bit packing boundary is crossed.
                for (int i = 0; i < 12; i++) {
                    seg.set(ValueLayout.JAVA_BYTE, off + 4 + i, (byte) (b * 12 + i));
                }
                // qh (32 bytes) and qs (128 bytes): the four adversarial patterns, by block.
                for (int i = 0; i < 32; i++) {
                    seg.set(ValueLayout.JAVA_BYTE, off + 16 + i, pattern(b, i, rnd));
                }
                for (int i = 0; i < 128; i++) {
                    seg.set(ValueLayout.JAVA_BYTE, off + 48 + i, pattern(b, i, rnd));
                }
            }

            Q5_KFloatTensor reference = new Q5_KFloatTensor(elements, seg);
            LongIndexedTensor candidate = new LongIndexedTensor(seg, DataType.Q5_K);

            for (int i = 0; i < elements; i++) {
                float want = reference.getFloat(i);
                float got = candidate.valueAt(i);
                assertEquals(
                        "element " + i + " (block " + (i / QK_K) + ", offset " + (i % QK_K) + ")",
                        Float.floatToRawIntBits(want),
                        Float.floatToRawIntBits(got));
            }
        }
    }

    private static byte pattern(int block, int i, Random rnd) {
        return switch (block % 4) {
            case 0 -> (byte) 0x00;
            case 1 -> (byte) 0xFF;
            case 2 -> (byte) ((i % 2 == 0) ? 0xAA : 0x55);
            default -> (byte) rnd.nextInt(256);
        };
    }
}
