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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Batched-prefill kernels specific to the Qwen2 family. */
public final class Qwen2BatchPrefillKernels {

    private Qwen2BatchPrefillKernels() {}

    /**
     * Adds the Q, K and V projection biases to every row of a batch: {@code q[t, r] += qBias[r]}
     * and likewise for K and V. One thread per (token, output row) over {@code dim + 2 * kvDim}
     * rows a token; threads past {@code batch} rows do nothing.
     */
    public static void batchedQKVBias(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray qBias,
            FloatArray kBias,
            FloatArray vBias,
            int dim,
            int kvDim,
            int batch) {
        int index = context.globalIdx;
        int rowsPerToken = dim + 2 * kvDim;
        if (index < batch * rowsPerToken) {
            int token = index / rowsPerToken;
            int row = index % rowsPerToken;
            if (row < dim) {
                int o = token * dim + row;
                qBatch.set(o, qBatch.get(o) + qBias.get(row));
            } else if (row < dim + kvDim) {
                int r = row - dim;
                int o = token * kvDim + r;
                kBatch.set(o, kBatch.get(o) + kBias.get(r));
            } else {
                int r = row - dim - kvDim;
                int o = token * kvDim + r;
                vBatch.set(o, vBatch.get(o) + vBias.get(r));
            }
        }
    }

    /**
     * {@link #batchedQKVBias} over the packed {@code [q|k|v]} rows the tensor-core QKV GEMM writes:
     * row {@code r} of token {@code t} sits at {@code t * (dim + 2 * kvDim) + r}. One thread per
     * (token, output row); threads past {@code batch} tokens, and the GEMM's padding rows, are left
     * alone.
     */
    public static void batchedPackedQKVBias(
            KernelContext context,
            FloatArray qkvBatch,
            FloatArray qBias,
            FloatArray kBias,
            FloatArray vBias,
            int dim,
            int kvDim,
            int batch) {
        int index = context.globalIdx;
        int rowsPerToken = dim + 2 * kvDim;
        if (index < batch * rowsPerToken) {
            int row = index % rowsPerToken;
            float bias;
            if (row < dim) {
                bias = qBias.get(row);
            } else if (row < dim + kvDim) {
                bias = kBias.get(row - dim);
            } else {
                bias = vBias.get(row - dim - kvDim);
            }
            qkvBatch.set(index, qkvBatch.get(index) + bias);
        }
    }
}
