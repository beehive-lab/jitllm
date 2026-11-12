package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class FP32TornadoTensor extends TornadoTensor {
    private final FloatArray tornadoNativeArray;

    public FP32TornadoTensor(FloatArray floatArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = floatArray;
    }

    public static FP32TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new FP32TornadoTensor(FloatArray.fromSegmentShallow(segment), shape);
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
