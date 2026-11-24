package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * A tensor in the {@link GGMLType#MXFP4} format, retained on the device as the file stores it: 17
 * bytes per 32 weights, one E8M0 exponent byte and 16 bytes of E2M1 nibbles.
 *
 * <p>The device reads it as ints ({@link #asIntArray}), a view over the same mapped bytes: a block
 * is then five aligned int loads and a funnel shift, where single bytes would be seventeen loads.
 * The view is what is uploaded, so the device holds the tensor once. It needs the tensor to be a
 * whole number of ints, which every file of whole 32-value rows of an even block count is.
 */
public class MXFP4TornadoTensor extends TornadoTensor {

    public static final int QK = 32;

    public static final int BLOCK_BYTES = 17;

    private final ByteArray tornadoNativeArray;

    private final IntArray ints;

    public MXFP4TornadoTensor(ByteArray byteArray, IntArray ints, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
        this.ints = ints;
    }

    public static MXFP4TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        IntArray ints =
                segment.byteSize() % Integer.BYTES == 0
                        ? IntArray.fromSegmentShallow(segment)
                        : null;
        return new MXFP4TornadoTensor(ByteArray.fromSegmentShallow(segment), ints, shape);
    }

    /** The same bytes as ints, for aligned loads; what the device holds. */
    public IntArray asIntArray() {
        if (ints == null) {
            throw new UnsupportedOperationException(
                    "an MXFP4 tensor of " + shape() + " is not a whole number of ints");
        }
        return ints;
    }

    @Override
    public Object deviceArray() {
        return asIntArray();
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
