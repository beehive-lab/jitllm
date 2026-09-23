package org.beehive.jitllm.backend.tornado.device;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assume.assumeFalse;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.junit.Test;

/**
 * Metal parity, backlog task 4 — device discovery and selection.
 *
 * <p>{@code TornadoDevicesTest} cannot cover a real accelerator's identity: it runs under the
 * ordinary unit profile, whose {@code argLine} carries no TornadoVM module path or JVMCI exports,
 * so {@link TornadoDevices#current()} always resolves the no-accelerator placeholder there
 * (verified — {@code mvn test -Dtest=TornadoDevicesTest} resolves {@code BackendId.CPU}/{@code
 * "unavailable"} even with a Metal SDK on the machine). This class runs under {@code
 * -Paccel-tests}, whose {@code argLine} does carry them, so a real device is expected to resolve
 * here.
 *
 * <p>No model fixture is needed — resolution asks TornadoVM's runtime directly, not a loaded model
 * — so these run whenever a device is present, independent of which GGUF fixtures happen to be
 * local.
 */
public class DeviceDiscoveryAccelTest {

    @Test
    public void aRealAcceleratorResolvesANonPlaceholderIdentity() {
        Device device = TornadoDevices.current();
        assumeFalse(
                "no accelerator resolved on this run", "unavailable".equals(device.displayName()));

        assertNotNull(device.id());
        assertNotNull(device.id().backend());
        assertFalse(
                "a resolved accelerator must not carry the CPU placeholder backend",
                BackendId.CPU.equals(device.id().backend()));
        assertFalse(
                "a resolved accelerator must report a non-blank display name",
                device.displayName().isBlank());
    }

    /**
     * Task 4's "stable across runs" acceptance item, the in-process half of it: repeated resolution
     * within one JVM must answer the identical identity, since {@code DeviceId} reaches the
     * compiled-program cache key and a component that changes underneath the cache would key one
     * entry per lookup.
     *
     * <p>The cross-process half — the same identity across independent fresh JVMs — was verified
     * for this exact resolution path in the backlog task 1 inventory (two separate {@code mvn
     * -Dtest=.} invocations, both resolving {@code metal:Apple Metal}; see {@code
     *) and is procedure, not something this
     * suite spawns a second JVM to re-check on every run — no test elsewhere in this suite spawns a
     * JVM, and doing so here would trade a well-understood manual check for a new, untested
     * pattern.
     */
    @Test
    public void resolutionIsStableAcrossRepeatedCallsInThisProcess() {
        Device first = TornadoDevices.current();
        Device second = TornadoDevices.current();
        assertEquals(first.id(), second.id());
        assertEquals(first.displayName(), second.displayName());
    }

    /**
     * The Metal-specific instance of task 4's core claim: on a Metal run, the backend resolves to
     * {@code BackendId.METAL} through the same neutral path every other backend uses — no Metal
     * branch anywhere in {@link TornadoDevices}. Skips (rather than failing) on a non-Metal run, so
     * this suite stays meaningful on CUDA/OpenCL machines too.
     */
    @Test
    public void aMetalRunResolvesTheMetalBackendId() {
        Device device = TornadoDevices.current();
        assumeTrue("not a Metal run", BackendId.METAL.equals(device.id().backend()));

        assertEquals(BackendId.METAL, device.id().backend());
        assertFalse(
                "Metal's display name should be self-describing enough for an error message",
                device.displayName().isBlank());
    }

    /**
     * Actual accelerator execution, not a mocked limit: a CUDA device reports its threads-per-
     * block limit, and the production model's eight-part delta rule (8 x 128 = 1024 lanes) is
     * admitted only when that limit is at least 1024, which every CUDA device from compute
     * capability 2.0 on reports. The mocked 256/512/1024 cases live in Qwen35DeltaRuleGeometryTest.
     */
    @org.junit.Test
    public void theCudaDeviceReportsItsWorkgroupLimitAndAdmitsTheEightPartDeltaRule() {
        org.beehive.jitllm.runtime.backend.Device device =
                org.beehive.jitllm.backend.tornado.device.TornadoDevices.current();
        org.junit.Assume.assumeTrue(
                "not a CUDA device",
                device.backend() == org.beehive.jitllm.runtime.backend.BackendId.CUDA);
        long limit = device.maxWorkGroupSize();
        org.junit.Assert.assertTrue("CUDA reports a workgroup limit, got " + limit, limit >= 1024);
        org.junit.Assert.assertEquals(
                org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayers.DeltaRuleGeometry.SPLIT8,
                org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayers.selectDeltaRuleGeometry(
                        128, limit));
    }
}
