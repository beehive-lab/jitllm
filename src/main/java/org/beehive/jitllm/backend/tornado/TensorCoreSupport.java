package org.beehive.jllm.backend.tornado;

import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.runtime.backend.DeviceCapability;

/**
 * Whether the active device can execute the tensor-core (MMA) batch-prefill kernels. TornadoVM
 * lowers the MMA intrinsics ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) only on the NVIDIA
 * CUDA backend.
 */
public final class TensorCoreSupport {

    private TensorCoreSupport() {}

    public static boolean isTensorCoreCapableBackend() {
        return TornadoDevices.current().capabilities().supports(DeviceCapability.TENSOR_CORE_MMA);
    }

    /** Whether the int8 tensor-core kernel family runs on this device (see the capability). */
    public static boolean isInt8MmaCapable() {
        return TornadoDevices.current()
                .capabilities()
                .supports(DeviceCapability.INT8_TENSOR_CORE_MMA);
    }
}
