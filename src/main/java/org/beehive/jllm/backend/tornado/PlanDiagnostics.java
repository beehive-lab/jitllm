package org.beehive.jllm.backend.tornado;

import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.runtime.backend.ExecutionInfo;
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
