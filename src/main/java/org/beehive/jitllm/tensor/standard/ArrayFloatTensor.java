/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Portions of this file are derived from Llama3.java
 * (https://github.com/mukel/llama3.java),
 * Copyright (c) 2024 Alfonso² Peterssen and Copyright (c) 2024 Andrej Karpathy.
 * Used under the MIT License; see the LICENSE file.
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
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

public final class ArrayFloatTensor extends FloatTensor {

    final float[] values;

    public ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return values[index];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return MemorySegment.ofArray(values);
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
}
