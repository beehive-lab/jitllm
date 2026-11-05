package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#Q5_0} format, retained on the device. 32 weights in 22
 * bytes:
 *
 * <pre>
 *   offset 0   d   (fp16)   the block's scale
 *   offset 2   qh  (32 b)   the fifth bit of each weight, weight j at bit j
 *   offset 6   qs  (16 B)   the low four bits, weights 0..15 in the low nibbles, 16..31 in the high
 * </pre>
 *
 * <p>A weight is {@code d * (q - 16)} with {@code q} its five bits, unsigned.
 *
 * <p>Present because quantizers mix it into K-quant files: {@code
 * Qwen2.5-0.5B-Instruct-Q4_K_M.gguf} holds its q, k, output, gate and up projections as Q5_0.
 */
public class Q5_0TornadoTensor extends TornadoTensor {

    /** Weights per block. */
    public static final int QK = 32;

    /** Bytes per block: 2 (d) + 4 (fifth bits) + 16 (packed nibbles). */
    public static final int BLOCK_BYTES = 22;

    private final ByteArray tornadoNativeArray;

    public Q5_0TornadoTensor(ByteArray byteArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
    }

    public static Q5_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new Q5_0TornadoTensor(ByteArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q5_0;
    }
}
