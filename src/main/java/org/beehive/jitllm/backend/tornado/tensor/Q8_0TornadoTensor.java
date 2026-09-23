package org.beehive.jllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * This class represents a quantized tensor in the {@link GGMLType#Q8_0} format. It is backed by a
 * {@link ByteArray} containing both the quantized values and the scale factors. The underlying
 * {@link ByteArray} contains N Q8_0 blocks, where N is the tensor size divided by 32.: Each Q8_0
 * Block has the following layout: [Scale Factor (fp16) - 2 bytes] [Quantized Value 0 (int8) - 1
 * byte]. [Quantized Value 31 (int8) - 1 byte]
 */
public class Q8_0TornadoTensor extends TornadoTensor {

    private final ByteArray
            tornadoNativeArray; // Unified Q8_0 tensor in the memorySegment of the ByteArray

    public Q8_0TornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q8_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q8_0TornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }
}
