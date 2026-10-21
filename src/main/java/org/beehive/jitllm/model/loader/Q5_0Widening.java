package org.beehive.jitllm.model.loader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Rewrites Q5_0 blocks as Q8_0 blocks, exactly.
 *
 * <p>Both formats hold 32 values per block with one FP16 scale, and a Q5_0 value is a 5-bit integer
 * in {@code [-16, 15]}, which a signed byte holds unchanged. So the Q8_0 block keeps the scale and
 * stores each value as its byte: the decoded values are bit-identical, and every Q8_0 kernel reads
 * the tensor without a Q5_0 path of its own. The cost is 34 bytes a block where Q5_0 used 22.
 */
public final class Q5_0Widening {

    private static final int BLOCK = 32;
    private static final int Q5_0_BLOCK_BYTES = 22;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private Q5_0Widening() {}

    /**
     * @param source the Q5_0 data, starting at {@code sourceOffset}
     * @param elements values in the tensor, a multiple of 32
     * @param headerBytes zeroed bytes to leave in front of the result, for a device array header
     */
    public static MemorySegment toQ8_0(
            MemorySegment source, long sourceOffset, long elements, long headerBytes) {
        if (elements % BLOCK != 0) {
            throw new IllegalArgumentException("Q5_0 tensor of " + elements + " values");
        }
        long blocks = elements / BLOCK;
        MemorySegment out = Arena.ofAuto().allocate(headerBytes + blocks * Q8_0_BLOCK_BYTES, 4);
        out.asSlice(0, headerBytes).fill((byte) 0);
        for (long b = 0; b < blocks; b++) {
            long in = sourceOffset + b * Q5_0_BLOCK_BYTES;
            long o = headerBytes + b * Q8_0_BLOCK_BYTES;
            out.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    o,
                    source.get(ValueLayout.JAVA_SHORT_UNALIGNED, in));
            int qh =
                    source.get(
                            ValueLayout.JAVA_INT_UNALIGNED.withOrder(
                                    java.nio.ByteOrder.LITTLE_ENDIAN),
                            in + 2);
            for (int j = 0; j < BLOCK / 2; j++) {
                int packed = source.get(ValueLayout.JAVA_BYTE, in + 6 + j) & 0xFF;
                int lo = (packed & 0x0F) | (((qh >>> j) & 1) << 4);
                int hi = (packed >>> 4) | (((qh >>> (j + 16)) & 1) << 4);
                out.set(ValueLayout.JAVA_BYTE, o + 2 + j, (byte) (lo - 16));
                out.set(ValueLayout.JAVA_BYTE, o + 2 + j + BLOCK / 2, (byte) (hi - 16));
            }
        }
        return out;
    }
}
