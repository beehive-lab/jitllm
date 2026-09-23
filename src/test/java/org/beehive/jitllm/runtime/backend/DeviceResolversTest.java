package org.beehive.jitllm.runtime.backend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Optional;
import org.junit.Test;

public class DeviceResolversTest {

    private static Device device(BackendId backend, String handle) {
        DeviceId id = DeviceId.of(backend, handle);
        return new Device() {
            @Override
            public DeviceId id() {
                return id;
            }

            @Override
            public DeviceCapabilities capabilities() {
                return DeviceCapabilities.NONE;
            }

            @Override
            public String displayName() {
                return handle;
            }
        };
    }

    private static DeviceResolver resolverOf(Device device) {
        return () -> device;
    }

    @Test
    public void zeroResolversIsAValidEmptyAnswer() {
        // Unlike KvStorageFactories: a backend.cpu-only build registers none, and that is normal.
        assertEquals(Optional.empty(), DeviceResolvers.fromDiscovered(List.of()));
    }

    @Test
    public void oneResolverIsReturned() {
        DeviceResolver resolver = resolverOf(device(BackendId.METAL, "Apple Metal"));
        assertSame(resolver, DeviceResolvers.fromDiscovered(List.of(resolver)).orElseThrow());
    }

    @Test
    public void duplicateResolversFailDeterministically() {
        DeviceResolver first = resolverOf(device(BackendId.CUDA, "a"));
        DeviceResolver second = resolverOf(device(BackendId.OPENCL, "b"));
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () -> DeviceResolvers.fromDiscovered(List.of(first, second)));
        assertTrue(
                thrown.getMessage(), thrown.getMessage().contains("more than one device resolver"));
    }

    @Test
    public void discoveredReflectsWhateverServiceLoaderActuallyFinds() {
        // The real seam: on this project's own build, backend.tornado always registers
        // TornadoDeviceResolver, so discovered() must not throw and must not be ambiguous here.
        // What it resolves to (a real device, or the no-accelerator placeholder) is not asserted —
        // that is TornadoDevicesTest's and DeviceDiscoveryAccelTest's job.
        Optional<DeviceResolver> resolver = DeviceResolvers.discovered();
        assertTrue(
                "this project's own backend.tornado module must register exactly one resolver",
                resolver.isPresent());
        assertFalse(DeviceResolvers.loaded().isEmpty());
    }
}
