package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

/**
 * A writable half-precision tensor: storage in IEEE 754 binary16, arithmetic in FP32.
 *
 * <p>The CPU's key/value cache when FP16 storage is selected. Every value is rounded to half
 * precision when it is written and widened back when it is read, so accumulation stays FP32 — the
 * same contract as the device's FP16 cache.
 */
public final class ArrayHalfFloatTensor extends FloatTensor {

    final short[] values;

    public ArrayHalfFloatTensor(short[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        return new ArrayHalfFloatTensor(new short[FloatTensor.numberOfElements(dims)]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return Float.float16ToFloat(values[index]);
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = Float.floatToFloat16(value);
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return MemorySegment.ofArray(values);
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        float[] widened = new float[species.length()];
        for (int i = 0; i < widened.length; i++) {
            widened[i] = Float.float16ToFloat(values[index + i]);
        }
        return FloatVector.fromArray(species, widened, 0);
    }
}
