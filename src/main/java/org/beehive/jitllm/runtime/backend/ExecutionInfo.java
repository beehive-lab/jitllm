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
package org.beehive.jitllm.runtime.backend;

import org.beehive.jitllm.api.Experimental;

/**
 * Selected execution settings, describing the prepared plan rather than requested flags. The kernel
 * descriptions apply to prefill; native libraries choose their own algorithms. A prefill batch size
 * of one also describes the single-token execution path.
 */
@Experimental
public record ExecutionInfo(
        String backend,
        String device,
        String runtime,
        String mode,
        int prefillBatchSize,
        String kvCache,
        String prefillKernels,
        String prefillAttention,
        boolean cudaGraphs,
        boolean stagedTransfers,
        String nativeLibraries) {
    public ExecutionInfo(
            String backend,
            String device,
            String runtime,
            String mode,
            int prefillBatchSize,
            String kvCache,
            String prefillKernels,
            String prefillAttention,
            boolean cudaGraphs,
            boolean stagedTransfers) {
        this(
                backend,
                device,
                runtime,
                mode,
                prefillBatchSize,
                kvCache,
                prefillKernels,
                prefillAttention,
                cudaGraphs,
                stagedTransfers,
                "none");
    }
}
