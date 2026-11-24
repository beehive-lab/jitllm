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
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * This class represents a quantized tensor in the {@link GGMLType#Q8_0} format. It is backed by a
 * {@link ByteArray} containing both the quantized values and the scale factors. The underlying
 * {@link ByteArray} contains N Q8_0 blocks, where N is the tensor size divided by 32.: Each Q8_0
 * Block has the following layout: [Scale Factor (fp16) - 2 bytes] [Quantized Value 0 (int8) - 1
 * byte]. [Quantized Value 31 (int8) - 1 byte]
 */
public class Q8_0TornadoTensor extends TornadoTensor {

    private final ByteArray
            tornadoNativeArray; // Unified Q8_0 tensor in the memorySegment of the ByteArray

    public Q8_0TornadoTensor(ByteArray byteArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
    }

    public static Q8_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new Q8_0TornadoTensor(ByteArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }
}
