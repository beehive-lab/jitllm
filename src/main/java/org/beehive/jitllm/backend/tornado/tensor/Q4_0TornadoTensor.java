package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#Q4_0} format, <b>retained</b> on the device.
 *
 * <p>Backed by the file's own bytes, exactly as {@link Q8_0TornadoTensor} and {@link
 * Q4_KTornadoTensor} are: the segment is wrapped, not converted, so nothing is materialized at
 * load. Each block covers 32 weights in 18 bytes — 4.5 bits per weight against Q8_0's 8.5:
 *
 * <pre>
 *   offset 0   d   (fp16)   the block's single scale
 *   offset 2   qs  (16 B)   32 4-bit weights, low nibbles first
 * </pre>
 *
 * <p>A weight is {@code d * (q - 8)}, decoded inside the dot product by {@link
 * org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0}. The layers pass {@link
 * #asByteArray()} just as they do for Q8_0, and only the kernel that reads it differs.
 *
 * <p><b>Why this exists</b>: materializing Q4_0 as Q8_0 at load nearly doubles a model's device
 * footprint. That is what kept Devstral's Q4_K off a 24 GiB machine until {@link Q4_KTornadoTensor}
 * retained it, and it is the same arithmetic here — a Q4_0 file costs its own size on the device
 * rather than twice it.
 */
public class Q4_0TornadoTensor extends TornadoTensor {

    /** Weights per block. */
    public static final int QK = 32;

    /** Bytes per block: 2 (d) + 16 (packed nibbles). */
    public static final int BLOCK_BYTES = 18;

    /** Byte offset of the packed nibbles within a block. */
    public static final int QS_OFFSET = 2;

    private final ByteArray tornadoNativeArray;

    public Q4_0TornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q4_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q4_0TornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }
}
