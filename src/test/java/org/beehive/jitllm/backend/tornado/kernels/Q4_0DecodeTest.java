package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.beehive.jitllm.tensor.standard.Q4_0FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * The device Q4_0 decode against the host's, on the same bytes.
 *
 * <p><b>Not circular</b>: the bytes are random, and the two decoders are independent
 * implementations reading them — one through {@link MemorySegment}, one through {@link ByteArray}.
 * Any 18-byte block is a structurally valid Q4_0 block, so random content exercises both nibble
 * halves and the full range of scales, including the negatives and the subnormals a hand-picked
 * example would not reach.
 *
 * <p>The failure this guards against is the one that has no other symptom. A decode that recentres
 * by the wrong constant, or reads the high nibble where the low one belongs, produces weights of
 * plausible magnitude and a model that generates fluent, wrong text at normal throughput — the
 * failure mode that a GPU-versus-GPU comparison cannot see, because both sides would be equally
 * wrong.
 *
 * <p>A unit test rather than an accelerator one: this is the arithmetic, and it is the half that
 * can be wrong silently. That the kernels around it also <i>run</i> is a separate question, settled
 * by a real model on a real device.
 */
public class Q4_0DecodeTest {

    private static final int QK = 32;
    private static final int BLOCK_BYTES = 18;

    @Test
    public void theDeviceDecodeMatchesTheHostDecodeOnEveryElement() {
        int blocks = 16;
        int elements = blocks * QK;
        byte[] raw = new byte[blocks * BLOCK_BYTES];
        // Fixed seed: a disagreement must be reproducible, not something that shows up one run in
        // ten.
        new Random(20260908L).nextBytes(raw);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);

            Q4_0FloatTensor host = new Q4_0FloatTensor(elements, segment);
            // Built element-wise rather than wrapped: a TornadoNativeArray segment carries a
            // 16-byte header, so wrapping a bare segment would shift every offset by it and this
            // would be testing the harness rather than the decode.
            ByteArray device = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                device.set(i, raw[i]);
            }

            for (int i = 0; i < elements; i++) {
                int blockIndex = i / QK;
                int withinBlock = i - blockIndex * QK;
                float expected = host.getFloat(i);
                float actual =
                        TransformerComputeKernelsQ4_0.decode(
                                device, blockIndex * BLOCK_BYTES, withinBlock);
                if (Float.isNaN(expected)) {
                    // A random scale can be NaN; the two must still agree that it is.
                    assertEquals(
                            "element " + i + ": host NaN, device " + actual, Float.NaN, actual, 0f);
                    continue;
                }
                assertEquals(
                        "element "
                                + i
                                + " (block "
                                + blockIndex
                                + ", offset "
                                + withinBlock
                                + ") decodes differently on the device than on the host",
                        expected,
                        actual,
                        0f);
            }
        }
    }

    /**
     * The nibble is unsigned and recentred by eight, and each half of a block reads its own nibble
     * of the same byte.
     *
     * <p>Asserted against hand-built bytes as well as against the host, because "the two agree" is
     * only half the claim — they could agree and both be a different format. These values come from
     * the specification.
     */
    @Test
    public void decodesTheSpecifiedLayout() {
        byte[] raw = new byte[BLOCK_BYTES];
        // d = 2.0 in fp16 is 0x4000.
        raw[0] = 0x00;
        raw[1] = 0x40;
        // Byte i holds element i in its low nibble and element i + 16 in its high nibble.
        raw[2] = (byte) 0x0F; // element 0 -> q 15, element 16 -> q 0
        raw[3] = (byte) 0x80; // element 1 -> q 0,  element 17 -> q 8

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);
            ByteArray device = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                device.set(i, raw[i]);
            }

            // d * (q - 8), with d = 2.
            assertEquals(14.0f, TransformerComputeKernelsQ4_0.decode(device, 0, 0), 0f);
            assertEquals(-16.0f, TransformerComputeKernelsQ4_0.decode(device, 0, 16), 0f);
            assertEquals(-16.0f, TransformerComputeKernelsQ4_0.decode(device, 0, 1), 0f);
            assertEquals(0.0f, TransformerComputeKernelsQ4_0.decode(device, 0, 17), 0f);

            Q4_0FloatTensor host = new Q4_0FloatTensor(QK, segment);
            assertEquals(14.0f, host.getFloat(0), 0f);
            assertEquals(-16.0f, host.getFloat(16), 0f);
            assertEquals(-16.0f, host.getFloat(1), 0f);
            assertEquals(0.0f, host.getFloat(17), 0f);
        }
    }
}
