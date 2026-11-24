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
package org.beehive.jitllm.integration.cli;

import java.util.concurrent.atomic.AtomicBoolean;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/** The command-line warnings for experimental options, each printed once per process. */
public final class ExperimentalWarnings {

    private static final AtomicBoolean NATIVE_LIBRARIES = new AtomicBoolean();

    static final String NATIVE_LIBRARIES_WARNING =
            "WARNING: native libraries (--with-native-libraries) are experimental. They are"
                    + " implemented for batched prefill on CUDA tensor-core devices only, for Qwen3"
                    + " F16 (cuBLAS projections, cuDNN attention) and Gemma 4 Q8_0/Q4_0 (cuBLAS"
                    + " projections), and keep FP16 projection copies beside the weights, so they"
                    + " need more device memory than the JIT kernels;"
                    + " any other configuration is refused.";

    private ExperimentalWarnings() {}

    /** Prints the native-libraries warning if {@code policy} asks for them, once. */
    public static void nativeLibraries(ExecutionPolicy policy) {
        if (policy.nativeLibraries() && NATIVE_LIBRARIES.compareAndSet(false, true)) {
            System.err.println(NATIVE_LIBRARIES_WARNING);
        }
    }
}
