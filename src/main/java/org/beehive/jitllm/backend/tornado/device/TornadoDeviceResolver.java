package org.beehive.jllm.backend.tornado.device;

import org.beehive.jllm.runtime.backend.Device;
import org.beehive.jllm.runtime.backend.DeviceResolver;

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
