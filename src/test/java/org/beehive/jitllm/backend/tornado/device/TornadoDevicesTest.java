package org.beehive.jitllm.backend.tornado.device;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.junit.Test;

/**
 * These run wherever the unit gate runs, including on a machine with no accelerator: resolution
 * never throws, and the no-device answer is a normal one. What they cannot assert is <i>which</i>
 * capabilities a given machine has — that is a property of the machine — so they assert the
 * invariants instead: the answer is stable, the predicates agree with the resolved capabilities,
 * and the cache-key label is the one it always was.
 */
public class TornadoDevicesTest {

    @Test
    public void resolutionNeverThrowsAndIsStable() {
        Device first = TornadoDevices.current();
        assertNotNull(first);
        // Stability is the point, not the speed: a cache-key component that changes underneath the
        // cache produces one compiled program per lookup.
        assertSame(first, TornadoDevices.current());
        assertEquals(first.id(), TornadoDevices.current().id());
    }

    @Test
    public void theIdentityIsTheBackendAndTheDisplayNameTogether() {
        Device device = TornadoDevices.current();
        assertEquals(device.backend(), device.id().backend());
        assertEquals(device.displayName(), device.id().handle());
        assertTrue(
                "a display name must be usable in an error message",
                !device.displayName().isBlank());
    }

    @Test
    public void tensorCoreSupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current().capabilities().supports(DeviceCapability.TENSOR_CORE_MMA),
                TensorCoreSupport.isTensorCoreCapableBackend());
    }

    @Test
    public void int8MmaSupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.INT8_TENSOR_CORE_MMA),
                TensorCoreSupport.isInt8MmaCapable());
    }

    /** The compute capability is read from the device's own description, or is unknown. */
    @Test
    public void theComputeCapabilityComesFromTheDeviceDescription() {
        assertEquals(
                120,
                TornadoDevices.cudaComputeCapability("id=0x1\nDevice version   : CUDA 12.0\nx"));
        assertEquals(89, TornadoDevices.cudaComputeCapability("Device version : CUDA 8.9"));
        assertEquals(61, TornadoDevices.cudaComputeCapability("Device version: CUDA 6.1\n"));
        assertEquals(-1, TornadoDevices.cudaComputeCapability("Device version : OpenCL 3.0"));
        assertEquals(-1, TornadoDevices.cudaComputeCapability(""));
        assertEquals(-1, TornadoDevices.cudaComputeCapability(null));
    }

    /**
     * Each kernel family is granted by what the device executes, not by the backend's name: the
     * tensor-core families (FP16 and int8) need compute capability 8.0, the packed-integer path
     * 6.1, and an unreadable capability grants none of them. OpenCL and Metal never get them.
     */
    @Test
    public void tensorCoreGrantsFollowTheComputeCapability() {
        var cuda = uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA;
        var opencl = uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.OPENCL;
        var ampere =
                TornadoDevices.capabilitiesOf(cuda, "NVIDIA CUDA", "Device version : CUDA 8.0");
        assertTrue(ampere.supports(DeviceCapability.TENSOR_CORE_MMA));
        assertTrue(ampere.supports(DeviceCapability.INT8_TENSOR_CORE_MMA));
        assertTrue(ampere.supports(DeviceCapability.PACKED_INTEGER_DOT));
        var blackwell =
                TornadoDevices.capabilitiesOf(cuda, "NVIDIA CUDA", "Device version : CUDA 12.0");
        assertTrue(blackwell.supports(DeviceCapability.INT8_TENSOR_CORE_MMA));
        var turing =
                TornadoDevices.capabilitiesOf(cuda, "NVIDIA CUDA", "Device version : CUDA 7.5");
        assertFalse("no cp.async below 8.0", turing.supports(DeviceCapability.TENSOR_CORE_MMA));
        assertFalse(turing.supports(DeviceCapability.INT8_TENSOR_CORE_MMA));
        assertTrue("dp4a from 6.1", turing.supports(DeviceCapability.PACKED_INTEGER_DOT));
        var maxwell =
                TornadoDevices.capabilitiesOf(cuda, "NVIDIA CUDA", "Device version : CUDA 5.2");
        assertFalse(maxwell.supports(DeviceCapability.PACKED_INTEGER_DOT));
        var unknown = TornadoDevices.capabilitiesOf(cuda, "NVIDIA CUDA", "");
        assertFalse(
                "an unreadable device gets no architecture-gated grant",
                unknown.supports(DeviceCapability.TENSOR_CORE_MMA));
        assertFalse(unknown.supports(DeviceCapability.PACKED_INTEGER_DOT));
        var cl = TornadoDevices.capabilitiesOf(opencl, "NVIDIA CUDA", "Device version : CUDA 12.0");
        assertFalse(
                "OpenCL never lowers the MMA intrinsics",
                cl.supports(DeviceCapability.TENSOR_CORE_MMA));
        assertFalse(cl.supports(DeviceCapability.PACKED_INTEGER_DOT));
    }

    @Test
    public void warpShuffleSupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current().capabilities().supports(DeviceCapability.WARP_SHUFFLE),
                SchedulerDetectionService.isWarpShuffleSupported());
    }

    /**
     * Metal parity, task 5→6 follow-up. Deliberately independent of {@link
     * #warpShuffleSupportAgreesWithTheResolvedCapability} — {@code SUBGROUP_SHUFFLE_32} and {@code
     * WARP_SHUFFLE} are verified for different kernels on different backends and must not be
     * conflated (see {@code DeviceCapability.SUBGROUP_SHUFFLE_32}'s javadoc).
     */
    @Test
    public void subgroupShuffle32SupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32),
                SchedulerDetectionService.isSubgroupShuffle32Supported());
    }

    /**
     * The specific, narrow claim this capability is allowed to make: granted only where this
     * project measured it (Metal), never inferred from "not NVIDIA" or any other broad rule. A
     * device that is neither NVIDIA-class nor Metal (an OpenCL platform, say) must not pick up this
     * capability by accident.
     */
    @Test
    public void subgroupShuffle32IsNeverGrantedToADeviceThatIsNotMetal() {
        Device device = TornadoDevices.current();
        boolean isMetal =
                "Apple Metal".equals(device.displayName())
                        || device.id().backend().toString().equals("metal");
        if (!isMetal) {
            assertFalse(
                    "SUBGROUP_SHUFFLE_32 is verified for Metal only",
                    device.capabilities().supports(DeviceCapability.SUBGROUP_SHUFFLE_32));
        }
    }

    @Test
    public void theMetalPredicateIsTheAbsenceOfSplitKvAttention() {
        // It reads as a backend question and is a capability question: Metal is where the split-KV
        // kernel fails to JIT, and that is what every call site actually branches on.
        assertEquals(
                !TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SPLIT_KV_ATTENTION),
                SchedulerDetectionService.isMetalBackend());
    }

    @Test
    public void aMachineWithoutAnAcceleratorStillHasAStableAnswer() {
        Device device = TornadoDevices.current();
        if ("unavailable".equals(device.displayName())) {
            assertTrue(
                    "no device means no optional capabilities",
                    device.capabilities().asSet().isEmpty());
            assertEquals("none", device.capabilities().fingerprint());
        }
    }
}
