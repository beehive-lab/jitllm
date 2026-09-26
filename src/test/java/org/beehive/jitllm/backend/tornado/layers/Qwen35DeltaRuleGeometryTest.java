package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.backend.tornado.layers.Qwen35FFNLayers.DeltaRuleGeometry;
import org.junit.Test;

/**
 * The decode delta-rule kernel is chosen from the device's workgroup limit, and the grid follows
 * the same choice. Pure unit cover with the limit given, no accelerator: the value head is 128 wide
 * as on the 27B, so the eight-part kernel needs 1024 lanes, the two-part one 256.
 */
public class Qwen35DeltaRuleGeometryTest {

    private static final int HEAD = 128;

    @Test
    public void aLimitOf1024TakesTheEightPartKernel() {
        assertEquals(DeltaRuleGeometry.SPLIT8, Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, 1024));
        assertEquals(1024, DeltaRuleGeometry.SPLIT8.localSize(HEAD));
    }

    @Test
    public void aLimitOf512FallsBackToTheTwoPartKernel() {
        assertEquals(DeltaRuleGeometry.SPLIT2, Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, 512));
        assertEquals(256, DeltaRuleGeometry.SPLIT2.localSize(HEAD));
    }

    @Test
    public void aLimitOf256StillAdmitsTheTwoPartKernelExactly() {
        assertEquals(DeltaRuleGeometry.SPLIT2, Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, 256));
    }

    @Test
    public void aSmallerOrUnknownLimitTakesTheLanePerColumnKernel() {
        assertEquals(
                DeltaRuleGeometry.LANE_PER_COLUMN,
                Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, 128));
        assertEquals(
                DeltaRuleGeometry.LANE_PER_COLUMN,
                Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, 0));
        assertEquals(
                DeltaRuleGeometry.LANE_PER_COLUMN,
                Qwen35FFNLayers.selectDeltaRuleGeometry(HEAD, -1));
    }

    @Test
    public void aWidthTheSplitDoesNotDivideTakesTheLanePerColumnKernel() {
        // 8 does not divide 100; 2 does, so the two-part form is still available.
        assertEquals(DeltaRuleGeometry.SPLIT2, Qwen35FFNLayers.selectDeltaRuleGeometry(100, 1024));
        assertEquals(
                DeltaRuleGeometry.LANE_PER_COLUMN,
                Qwen35FFNLayers.selectDeltaRuleGeometry(99, 1024));
    }

    /** A device reporting {@code limit} lanes on {@code backend}; nothing else is read. */
    private static org.beehive.jitllm.runtime.backend.Device device(
            org.beehive.jitllm.runtime.backend.BackendId backend, long limit) {
        return new org.beehive.jitllm.runtime.backend.Device() {
            @Override
            public org.beehive.jitllm.runtime.backend.DeviceId id() {
                return org.beehive.jitllm.runtime.backend.DeviceId.of(backend, "test");
            }

            @Override
            public org.beehive.jitllm.runtime.backend.DeviceCapabilities capabilities() {
                return org.beehive.jitllm.runtime.backend.DeviceCapabilities.NONE;
            }

            @Override
            public String displayName() {
                return "test";
            }

            @Override
            public long maxWorkGroupSize() {
                return limit;
            }
        };
    }

    /**
     * CUDA keeps its full limit and the eight-part kernel; OpenCL, whose per-kernel limit is fixed
     * only when the driver compiles the kernel, is capped at the two-part kernel's 256 lanes — the
     * 1024-lane launch was refused there with CL_OUT_OF_RESOURCES.
     */
    @Test
    public void onlyCudaIsGivenTheFullDeviceLimit() {
        var cuda = device(org.beehive.jitllm.runtime.backend.BackendId.CUDA, 1024);
        var opencl = device(org.beehive.jitllm.runtime.backend.BackendId.OPENCL, 1024);
        assertEquals(1024, Qwen35FFNLayers.deltaRuleWorkGroupLimit(cuda));
        assertEquals(256, Qwen35FFNLayers.deltaRuleWorkGroupLimit(opencl));
        assertEquals(
                DeltaRuleGeometry.SPLIT8,
                Qwen35FFNLayers.selectDeltaRuleGeometry(
                        HEAD, Qwen35FFNLayers.deltaRuleWorkGroupLimit(cuda)));
        assertEquals(
                DeltaRuleGeometry.SPLIT2,
                Qwen35FFNLayers.selectDeltaRuleGeometry(
                        HEAD, Qwen35FFNLayers.deltaRuleWorkGroupLimit(opencl)));
        // A smaller or unknown limit is never raised.
        assertEquals(
                128,
                Qwen35FFNLayers.deltaRuleWorkGroupLimit(
                        device(org.beehive.jitllm.runtime.backend.BackendId.OPENCL, 128)));
        assertEquals(
                0,
                Qwen35FFNLayers.deltaRuleWorkGroupLimit(
                        device(org.beehive.jitllm.runtime.backend.BackendId.OPENCL, 0)));
    }

    @Test
    public void theGridIsTheGeometrysWorkgroupPerValueHead() {
        var config = syntheticConfig(32, HEAD);
        var split8 = Qwen35FFNLayers.deltaRuleWorker(DeltaRuleGeometry.SPLIT8, config);
        assertEquals(32L * 1024, split8.getGlobalWork()[0]);
        assertEquals(1024L, split8.getLocalWork()[0]);
        var split2 = Qwen35FFNLayers.deltaRuleWorker(DeltaRuleGeometry.SPLIT2, config);
        assertEquals(32L * 256, split2.getGlobalWork()[0]);
        assertEquals(256L, split2.getLocalWork()[0]);
        var lane = Qwen35FFNLayers.deltaRuleWorker(DeltaRuleGeometry.LANE_PER_COLUMN, config);
        assertEquals(32L * 128, lane.getGlobalWork()[0]);
        assertEquals(128L, lane.getLocalWork()[0]);
    }

    private static org.beehive.jitllm.model.qwen35.Qwen35Configuration syntheticConfig(
            int valueHeads, int stateSize) {
        return new org.beehive.jitllm.model.qwen35.Qwen35Configuration(
                "Q8_0",
                256,
                512,
                4,
                1,
                4,
                2,
                32,
                32,
                4,
                4,
                stateSize,
                1,
                valueHeads,
                valueHeads * stateSize,
                16,
                512,
                32,
                32,
                1e-6f,
                1e7f);
    }
}
