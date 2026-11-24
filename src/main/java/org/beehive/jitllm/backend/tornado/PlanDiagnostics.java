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
package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/** Diagnostics for the settings applied by the three master-plan constructors. */
final class PlanDiagnostics {
    private PlanDiagnostics() {}

    static ExecutionInfo describe(
            State state, String mode, int width, String kernels, String attention) {
        return describe(state, mode, width, kernels, attention, "none");
    }

    static ExecutionInfo describe(
            State state,
            String mode,
            int width,
            String kernels,
            String attention,
            String nativeLibraries) {
        var backend = TornadoRuntimeProvider.getTornadoRuntime().getBackend(0);
        String backendName = backend.getBackendType().name();
        boolean cuda = backendName.equals("CUDA");
        var descriptor = TornadoExecutionPlan.class.getModule().getDescriptor();
        String version = descriptor == null ? "" : descriptor.rawVersion().orElse("");
        return new ExecutionInfo(
                backendName,
                backend.getDefaultDevice().getPhysicalDevice().getDeviceName(),
                "TornadoVM" + (version.isEmpty() ? " (version unavailable)" : " " + version),
                mode,
                width,
                state.usesFp16KeyValueCache() ? "FP16" : "FP32",
                kernels,
                attention,
                cuda && TornadoVMMasterPlan.CUDA_GRAPHS,
                cuda,
                nativeLibraries);
    }
}
