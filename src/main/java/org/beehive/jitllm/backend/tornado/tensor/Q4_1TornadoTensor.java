package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#Q4_1} format, retained on the device.
 *
 * <p>Q4_0's affine sibling. 32 weights in 20 bytes:
 *
 * <pre>
 *   offset 0   d   (fp16)   the block's scale
 *   offset 2   m   (fp16)   the block's minimum
 *   offset 4   qs  (16 B)   32 4-bit weights, low nibbles first
 * </pre>
 *
 * <p>A weight is {@code d * q + m} with an <b>unsigned</b> nibble — no recentring by eight, which
 * is the whole difference from Q4_0 and the one that stays plausible when it is wrong.
 *
 * <p>Present because Qwen3.8-27B holds its first eight {@code ffn_down} tensors this way and every
 * other one as Q4_0. Materializing just those eight would be a silent conversion of 0.4 GB, and the
 * point of retention is that it does not happen.
 */
public class Q4_1TornadoTensor extends TornadoTensor {

    /** Weights per block. */
    public static final int QK = 32;

    /** Bytes per block: 2 (d) + 2 (m) + 16 (packed nibbles). */
    public static final int BLOCK_BYTES = 20;

    /** Byte offset of the packed nibbles within a block. */
    public static final int QS_OFFSET = 4;

    private final ByteArray tornadoNativeArray;

    public Q4_1TornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q4_1TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q4_1TornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_1;
    }
}
