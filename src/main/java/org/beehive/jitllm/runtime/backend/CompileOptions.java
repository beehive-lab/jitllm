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

/**
 * What changes the compiled artefact without changing the program's description.
 *
 * <p>A cache-key component, and separate from the program on purpose: the same {@link
 * org.beehive.jitllm.program.InferenceProgram} compiled with CUDA graph capture on and off is the
 * same computation and two different compiled programs.
 *
 * @param cudaGraphCapture whether {@code withCUDAGraph()} is applied — the property behind {@code
 *     jitllm.cudaGraphs}, and the reason <a
 *     href="././././././././docs/architecture/memory-and-concurrency.md">capability C1</a>'s fixed
 *     device addresses matter
 */
public record CompileOptions(boolean cudaGraphCapture) {

    /** A stable string for the cache key and for diagnostics. */
    public String fingerprint() {
        return cudaGraphCapture ? "cudaGraph=on" : "cudaGraph=off";
    }
}
