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
