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
