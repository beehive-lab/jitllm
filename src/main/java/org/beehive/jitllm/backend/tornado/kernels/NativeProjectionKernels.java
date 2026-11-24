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
package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Kernels that prepare quantized weights for a projection run by a native library. */
public final class NativeProjectionKernels {

    private static final int Q8_0_BLOCK = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private NativeProjectionKernels() {}

    /**
     * Decodes the first {@code n} weights of a Q8_0 tensor to FP32, row-major as the file stores
     * them: each 34-byte block is an FP16 scale and 32 signed bytes, and a weight is {@code scale *
     * q}.
     */
    public static void dequantizeQ8_0ToFP32(
            KernelContext context, ByteArray w, FloatArray out, int n) {
        int i = context.globalIdx;
        if (i < n) {
            int block = i / Q8_0_BLOCK;
            int offset = block * Q8_0_BLOCK_BYTES;
            float scale = w.getHalfFloat(offset).getFloat32();
            out.set(i, scale * w.get(offset + 2 + i % Q8_0_BLOCK));
        }
    }
}
