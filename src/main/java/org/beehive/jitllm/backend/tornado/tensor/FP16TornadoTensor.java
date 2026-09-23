package org.beehive.jllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class FP16TornadoTensor extends TornadoTensor {
    private final HalfFloatArray tornadoNativeArray;

    public FP16TornadoTensor(HalfFloatArray halfFloatArray) {
        this.tornadoNativeArray = halfFloatArray;
    }

    public static FP16TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new FP16TornadoTensor(HalfFloatArray.fromSegmentShallow(segment));
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
