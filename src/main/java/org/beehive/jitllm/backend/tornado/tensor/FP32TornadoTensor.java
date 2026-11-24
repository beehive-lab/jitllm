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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class FP32TornadoTensor extends TornadoTensor {
    private final FloatArray tornadoNativeArray;

    public FP32TornadoTensor(FloatArray floatArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = floatArray;
    }

    public static FP32TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new FP32TornadoTensor(FloatArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public FloatArray asFloatArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }
}
