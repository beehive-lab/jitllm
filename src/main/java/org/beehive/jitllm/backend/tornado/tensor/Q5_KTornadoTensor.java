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
 * A quantized tensor in the {@link GGMLType#Q5_K} format, retained on the device.
 *
 * <p>Q4_K with a fifth bit. 256 weights in 176 bytes — 5.5 bits per weight against Q8_0's 8.5:
 *
 * <pre>
 *   offset   0   d       (fp16)    super-block scale for the quantized scales
 *   offset   2   dmin    (fp16)    super-block scale for the quantized minima
 *   offset   4   scales  (12 B)    eight 6-bit scale/min pairs, packed as in Q4_K
 *   offset  16   qh      (32 B)    the fifth bit of each of the 256 weights
 *   offset  48   qs     (128 B)    the low four bits
 * </pre>
 *
 * <p>A weight is {@code d * scale(sub) * q - dmin * min(sub)} where {@code q} is five bits. The
 * fifth bit lives in a separate plane indexed by position rather than beside its nibble, which is
 * the part that is easy to address wrongly and impossible to notice: a wrong high bit shifts a
 * weight by 16 quantization steps and leaves the output fluent.
 *
 * <p>Present because every one of Qwen3.8-27B's 48 recurrent {@code ssm_out} projections is Q5_K.
 */
public class Q5_KTornadoTensor extends TornadoTensor {

    /** Weights per super-block. */
    public static final int QK_K = 256;

    /** Bytes per super-block. */
    public static final int BLOCK_BYTES = 176;

    /** Byte offset of the packed 6-bit scale/min pairs. */
    public static final int SCALES_OFFSET = 4;

    /** Byte offset of the fifth-bit plane. */
    public static final int QH_OFFSET = 16;

    /** Byte offset of the low four bits. */
    public static final int QS_OFFSET = 48;

    private final ByteArray tornadoNativeArray;

    public Q5_KTornadoTensor(ByteArray byteArray, Shape shape) {
        super(shape);
        this.tornadoNativeArray = byteArray;
    }

    public static Q5_KTornadoTensor fromTornadoMemorySegment(MemorySegment segment, Shape shape) {
        return new Q5_KTornadoTensor(ByteArray.fromSegmentShallow(segment), shape);
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q5_K;
    }
}
