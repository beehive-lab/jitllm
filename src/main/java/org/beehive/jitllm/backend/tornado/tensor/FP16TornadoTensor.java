package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.Shape;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class FP16TornadoTensor extends TornadoTensor {
    private final HalfFloatArray tornadoNativeArray;

    public FP16TornadoTensor(HalfFloatArray halfFloatArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = halfFloatArray;
    }

    public static FP16TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new FP16TornadoTensor(HalfFloatArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public HalfFloatArray asHalfFloatArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }
}
