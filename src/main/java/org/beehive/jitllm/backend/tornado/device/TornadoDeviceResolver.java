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
package org.beehive.jitllm.backend.tornado.device;

import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceResolver;

/**
 * The Tornado backend's {@link DeviceResolver} — registered in {@code META-INF/services},
 * delegating to the one place this process asks TornadoVM what it is running on.
 *
 * <p>Adds no resolution logic of its own: {@link TornadoDevices#current()} already resolves once,
 * lazily, and stably, which is exactly the contract {@link DeviceResolver} asks for.
 */
public final class TornadoDeviceResolver implements DeviceResolver {

    @Override
    public Device resolve() {
        return TornadoDevices.current();
    }
}
