/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.tensor.standard;

import java.lang.foreign.MemorySegment;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

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
