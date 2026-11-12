package org.beehive.jitllm.backend.tornado.ops;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/**
 * What an implementation is chosen for: the session's options (for example {@code
 * --no-native-libraries}) and what the device can do (for example tensor cores).
 */
public record Target(ExecutionPolicy policy, DeviceCapabilities capabilities) {

    /** The target for a session with {@code policy} on the device in use. */
    public static Target of(ExecutionPolicy policy) {
        return new Target(policy, TornadoDevices.target().capabilities());
    }

    public boolean supports(DeviceCapability capability) {
        return capabilities.supports(capability);
    }

    /**
     * Whether cuBLAS can run here: native libraries allowed (no {@code --no-native-libraries}),
     * tensor cores, and cuBLAS loads. Every cuBLAS and cuDNN implementation reads this, and so do
     * the checks made before a plan exists, such as the memory check.
     */
    public boolean cublas() {
        return policy.nativeLibraries()
                && supports(DeviceCapability.TENSOR_CORE_MMA)
                && NativePrefillSupport.cublasAvailable();
    }
}
