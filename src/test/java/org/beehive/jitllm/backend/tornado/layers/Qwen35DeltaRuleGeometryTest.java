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
