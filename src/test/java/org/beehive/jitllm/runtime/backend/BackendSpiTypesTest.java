package org.beehive.jitllm.runtime.backend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Set;
import org.junit.Test;

public class BackendSpiTypesTest {

    private static Device device(
            BackendId backend, String handle, String name, DeviceCapability... capabilities) {
        DeviceId id = DeviceId.of(backend, handle);
        DeviceCapabilities caps = DeviceCapabilities.of(capabilities);
        return new Device() {
            @Override
            public DeviceId id() {
                return id;
            }

            @Override
            public DeviceCapabilities capabilities() {
                return caps;
            }

            @Override
            public String displayName() {
                return name;
            }
        };
    }

    // --- BackendId -------------------------------------------------------------------------

    @Test
    public void backendIdsAreComparedByValueAndCanonicalizedByCase() {
        assertEquals(BackendId.CUDA, BackendId.of("CUDA"));
        assertEquals(BackendId.CUDA.hashCode(), BackendId.of("  cuda  ").hashCode());
        assertNotEquals(BackendId.CUDA, BackendId.METAL);
    }

    @Test
    public void anUnknownBackendIsRepresentable() {
        // The point of not using an enum: a backend can arrive without editing this package.
        BackendId invented = BackendId.of("vulkan");
        assertEquals("vulkan", invented.id());
        assertNotEquals(BackendId.CUDA, invented);
    }

    @Test
    public void aBlankBackendIdIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> BackendId.of("   "));
    }

    // --- DeviceId --------------------------------------------------------------------------

    @Test
    public void deviceIdentityIsTheBackendAndTheHandleTogether() {
        assertEquals(DeviceId.of(BackendId.CUDA, "0"), DeviceId.of(BackendId.CUDA, "0"));
        // The same handle on two backends is two devices, which is why the backend is in the id.
        assertNotEquals(DeviceId.of(BackendId.CUDA, "0"), DeviceId.of(BackendId.OPENCL, "0"));
    }

    @Test
    public void capabilitiesAreNotPartOfDeviceIdentity() {
        // The scheduler mode is overridable, so two lowerings on one device can differ. If
        // capabilities were folded into the identity, the cache key could not tell them apart --
        // and if the identity varied with them, one device would key as several.
        Device plain = device(BackendId.CUDA, "0", "NVIDIA");
        Device capable = device(BackendId.CUDA, "0", "NVIDIA", DeviceCapability.WARP_SHUFFLE);
        assertEquals(plain.id(), capable.id());
        assertNotEquals(plain.capabilities(), capable.capabilities());
    }

    // --- DeviceCapabilities ----------------------------------------------------------------

    @Test
    public void theCapabilityFingerprintDoesNotDependOnIterationOrder() {
        DeviceCapabilities one =
                DeviceCapabilities.of(
                        DeviceCapability.TENSOR_CORE_MMA, DeviceCapability.WARP_SHUFFLE);
        DeviceCapabilities other =
                DeviceCapabilities.of(
                        DeviceCapability.WARP_SHUFFLE, DeviceCapability.TENSOR_CORE_MMA);
        assertEquals(one.fingerprint(), other.fingerprint());
        assertEquals(one, other);
    }

    @Test
    public void anEmptyCapabilitySetHasAReadableFingerprint() {
        assertEquals("none", DeviceCapabilities.NONE.fingerprint());
    }

    @Test
    public void supportsAllIsWhatSelectionFiltersOn() {
        DeviceCapabilities caps = DeviceCapabilities.of(DeviceCapability.WARP_SHUFFLE);
        assertTrue(caps.supports(DeviceCapability.WARP_SHUFFLE));
        assertFalse(caps.supports(DeviceCapability.TENSOR_CORE_MMA));
        assertFalse(
                caps.supportsAll(
                        Set.of(DeviceCapability.WARP_SHUFFLE, DeviceCapability.TENSOR_CORE_MMA)));
    }

    // --- DeviceSelector --------------------------------------------------------------------

    @Test
    public void anEmptySelectorConstrainsNothing() {
        assertTrue(DeviceSelector.any().matches(device(BackendId.CPU, "host", "host CPU")));
        assertTrue(DeviceSelector.any().backendId().isEmpty());
        assertTrue(DeviceSelector.any().index().isEmpty());
    }

    @Test
    public void aSelectorFiltersOnBackendNameAndCapability() {
        Device cuda =
                device(
                        BackendId.CUDA,
                        "0",
                        "NVIDIA GeForce RTX 5090 Laptop GPU",
                        DeviceCapability.TENSOR_CORE_MMA);
        assertTrue(DeviceSelector.backend(BackendId.CUDA).matches(cuda));
        assertFalse(DeviceSelector.backend(BackendId.OPENCL).matches(cuda));
        assertTrue(DeviceSelector.any().withNameContaining("rtx 5090").matches(cuda));
        assertFalse(DeviceSelector.any().withNameContaining("radeon").matches(cuda));
        assertTrue(DeviceSelector.any().requiring(DeviceCapability.TENSOR_CORE_MMA).matches(cuda));
        assertFalse(
                DeviceSelector.any().requiring(DeviceCapability.SPLIT_KV_ATTENTION).matches(cuda));
    }

    @Test
    public void twoDifferentSelectorsCanNameOneDevice() {
        // This is the reason a selector must never be a cache key: keying on the request
        // would compile this device's programs twice.
        Device cuda = device(BackendId.CUDA, "0", "NVIDIA GeForce RTX 5090 Laptop GPU");
        DeviceSelector byBackend = DeviceSelector.backend(BackendId.CUDA);
        DeviceSelector byName = DeviceSelector.any().withNameContaining("nvidia");
        assertTrue(byBackend.matches(cuda));
        assertTrue(byName.matches(cuda));
        assertNotEquals(byBackend, byName);
        assertEquals(cuda.id(), cuda.id());
    }

    @Test
    public void selectorsAreImmutableUnderChaining() {
        DeviceSelector base = DeviceSelector.any();
        DeviceSelector derived = base.withBackend(BackendId.METAL).withIndex(1);
        assertTrue(base.backendId().isEmpty());
        assertEquals(BackendId.METAL, derived.backendId().orElseThrow());
        assertEquals(1, derived.index().orElseThrow());
    }

    @Test
    public void aNegativeDeviceIndexIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> DeviceSelector.any().withIndex(-1));
    }
}
