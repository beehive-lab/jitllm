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
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.format.GGMLType;

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
