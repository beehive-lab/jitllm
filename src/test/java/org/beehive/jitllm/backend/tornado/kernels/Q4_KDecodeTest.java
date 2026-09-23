package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.beehive.jitllm.tensor.standard.Q4_KFloatTensor;
import org.beehive.jitllm.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * The device Q4_K decode against the host's, on the same bytes.
 *
 * <p><b>Not circular</b>: the bytes are random, and the two decoders are independent
 * implementations reading them — one through {@link MemorySegment}, one through {@link ByteArray}.
 * Any 144-byte block is a structurally valid Q4_K super-block, so random content exercises every
 * scale/min packing branch, including the {@code subBlock >= 4} case where the 6-bit values
 * straddle two bytes, which a hand-picked example would likely miss.
 *
 * <p>A unit test rather than an accelerator one: this is the arithmetic, and it is the half that
 * can be wrong silently. That it also runs correctly *on a device* is established by the Devstral
 * run the amendment was made for.
 */
public class Q4_KDecodeTest {

    private static final int QK_K = 256;
    private static final int Q4_K_BLOCK_BYTES = 144;
    private static final int Q6_K_BLOCK_BYTES = 210;

    @Test
    public void theDeviceDecodeMatchesTheHostDecodeOnEveryElement() {
        int blocks = 8;
        int elements = blocks * QK_K;
        byte[] raw = new byte[blocks * Q4_K_BLOCK_BYTES];
        // Fixed seed: a disagreement must be reproducible, not something that shows up one run in
        // ten.
        new Random(20260904L).nextBytes(raw);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);

            Q4_KFloatTensor host = new Q4_KFloatTensor(elements, segment);
            // Built element-wise rather than wrapped: a TornadoNativeArray segment carries a
            // 16-byte header, so wrapping a bare segment would shift every offset by it and this
            // would be testing the harness rather than the decode.
            ByteArray device = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                device.set(i, raw[i]);
            }

            for (int i = 0; i < elements; i++) {
                int blockIndex = i / QK_K;
                int withinBlock = i - blockIndex * QK_K;
                float expected = host.getFloat(i);
                float actual =
                        TransformerComputeKernelsQ4_K.decode(
                                device, blockIndex * Q4_K_BLOCK_BYTES, withinBlock);
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
                        0.0f);
            }
        }
    }

    /**
     * The same equivalence for Q6_K, which the mixed K-quant path needs alongside Q4_K.
     *
     * <p>Q6_K's decode is the more intricate of the two — a weight's low nibble comes from one of
     * two {@code ql} planes and its high two bits from a group-dependent shift of {@code qh}, with
     * a signed per-16 scale — so restating it deserves the same check rather than more confidence.
     */
    @Test
    public void theDeviceQ6_KDecodeMatchesTheHostDecodeOnEveryElement() {
        int blocks = 8;
        int elements = blocks * QK_K;
        byte[] raw = new byte[blocks * Q6_K_BLOCK_BYTES];
        new Random(20260904L).nextBytes(raw);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(
                    raw, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, raw.length);

            Q6_KFloatTensor host = new Q6_KFloatTensor(elements, segment);
            ByteArray device = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                device.set(i, raw[i]);
            }

            for (int i = 0; i < elements; i++) {
                int blockIndex = i / QK_K;
                int withinBlock = i - blockIndex * QK_K;
                assertEquals(
                        "element "
                                + i
                                + " (block "
                                + blockIndex
                                + ", offset "
                                + withinBlock
                                + ") decodes differently on the device than on the host",
                        host.getFloat(i),
                        TransformerComputeKernelsQ6_K.decode(
                                device, blockIndex * Q6_K_BLOCK_BYTES, withinBlock),
                        0.0f);
            }
        }
    }
}
