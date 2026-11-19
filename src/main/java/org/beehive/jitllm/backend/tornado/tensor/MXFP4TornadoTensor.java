package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A tensor in the {@link GGMLType#MXFP4} format, retained on the device as the file stores it: 17
 * bytes per 32 weights, one E8M0 exponent byte and 16 bytes of E2M1 nibbles.
 */
public class MXFP4TornadoTensor extends TornadoTensor {

    public static final int QK = 32;

    public static final int BLOCK_BYTES = 17;

    private final ByteArray tornadoNativeArray;

    public MXFP4TornadoTensor(ByteArray byteArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
    }

    public static MXFP4TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new MXFP4TornadoTensor(ByteArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.MXFP4;
    }
}
