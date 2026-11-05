package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#IQ4_XS} format, retained on the device. 256 weights in
 * 136 bytes, in eight groups of 32:
 *
 * <pre>
 *   offset 0   d         (fp16)   the block's scale
 *   offset 2   scales_h  (16 b)   the high two bits of each group's 6-bit scale, group g at bit 2g
 *   offset 4   scales_l  (4 B)    the low four bits of the group scales, group g in byte g/2
 *   offset 8   qs        (128 B)  16 bytes per group: weights 0..15 in the low nibbles, 16..31 high
 * </pre>
 *
 * <p>A weight is {@code d * (ls - 32) * kvalues_iq4nl[q]}.
 *
 * <p>Present because quantizers mix it into K-quant files: {@code Qwen3-0.6B-UD-Q4_K_XL.gguf} holds
 * five layers' q, k, gate and up projections as IQ4_XS.
 */
public class IQ4_XSTornadoTensor extends TornadoTensor {

    /** Weights per block. */
    public static final int QK = 256;

    /** Bytes per block: 2 (d) + 2 (high scale bits) + 4 (low scale bits) + 128 (nibbles). */
    public static final int BLOCK_BYTES = 136;

    private final ByteArray tornadoNativeArray;

    public IQ4_XSTornadoTensor(ByteArray byteArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
    }

    public static IQ4_XSTornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new IQ4_XSTornadoTensor(ByteArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.IQ4_XS;
    }
}
