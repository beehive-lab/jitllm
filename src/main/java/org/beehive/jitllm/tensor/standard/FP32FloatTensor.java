package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

public final class FP32FloatTensor extends FloatTensor {
    final int size;
    final MemorySegment segment;

    public FP32FloatTensor(int size, MemorySegment segment) {
        this.size = size;
        this.segment = segment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return null;
    }

    @Override
    public float getFloat(int index) {
        return segment.get(ValueLayout.OfFloat.JAVA_FLOAT, index * Float.BYTES);
    }

    @Override
    public void setFloat(int index, float value) {
        segment.set(ValueLayout.OfFloat.JAVA_FLOAT, index * Float.BYTES, value);
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector is not yet implemented.");
    }
}
