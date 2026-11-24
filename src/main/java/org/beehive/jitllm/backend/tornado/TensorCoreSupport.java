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

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

/**
 * Whether the active device can execute the tensor-core (MMA) batch-prefill kernels. TornadoVM
 * lowers the MMA intrinsics ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) only on the NVIDIA
 * CUDA backend.
 */
public final class TensorCoreSupport {

    private TensorCoreSupport() {}

    public static boolean isTensorCoreCapableBackend() {
        return TornadoDevices.target().capabilities().supports(DeviceCapability.TENSOR_CORE_MMA);
    }

    /** Whether the int8 tensor-core kernel family runs on this device (see the capability). */
    public static boolean isInt8MmaCapable() {
        return TornadoDevices.target()
                .capabilities()
                .supports(DeviceCapability.INT8_TENSOR_CORE_MMA);
    }
}
