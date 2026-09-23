package org.beehive.jllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jllm.format.GGMLType;

/**
 * {@link FloatTensor} backed by raw BF16 (bfloat16) data.
 *
 * <p>BF16 stores the upper 16 bits of an IEEE-754 binary32 value (same sign/exponent layout,
 * truncated mantissa), so widening to float32 is a plain left-shift by 16 bits -- no exponent
 * rebiasing is needed, unlike IEEE binary16 (F16).
 */
public final class BF16FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public BF16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.BF16;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return null;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, index * (long) GGMLType.BFLOAT16_BYTES);
        return Float.intBitsToFloat(((int) bits) << 16);
    }
}
