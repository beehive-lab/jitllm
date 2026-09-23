package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class FP32TornadoTensor extends TornadoTensor {
    private final FloatArray tornadoNativeArray;

    public FP32TornadoTensor(FloatArray floatArray) {
        this.tornadoNativeArray = floatArray;
    }

    public static FP32TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new FP32TornadoTensor(FloatArray.fromSegmentShallow(segment));
    }

    @Override
    public FloatArray asFloatArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }
}
