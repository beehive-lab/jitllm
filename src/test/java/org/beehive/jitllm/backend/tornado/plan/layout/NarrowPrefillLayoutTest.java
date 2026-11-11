package org.beehive.jitllm.backend.tornado.plan.layout;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.junit.After;
import org.junit.Test;

/**
 * The narrow family's width rule and where its graphs sit in the batch-prefill layout.
 *
 * <p>Class A: no device, no model file.
 */
public class NarrowPrefillLayoutTest {

    @After
    public void restore() {
        System.clearProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
    }

    @Test
    public void aQuarterOfThePlanInWholeTilesAndOnlyWhereNarrower() {
        assertEquals(0, BatchPrefillTransformerLayerTaskGraphs.narrowRows(128));
        assertEquals(128, BatchPrefillTransformerLayerTaskGraphs.narrowRows(256));
        assertEquals(128, BatchPrefillTransformerLayerTaskGraphs.narrowRows(384));
        assertEquals(128, BatchPrefillTransformerLayerTaskGraphs.narrowRows(512));
        assertEquals(256, BatchPrefillTransformerLayerTaskGraphs.narrowRows(1024));
        assertEquals(512, BatchPrefillTransformerLayerTaskGraphs.narrowRows(2048));
    }

    @Test
    public void thePropertyTurnsItOff() {
        System.setProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY, "false");
        assertEquals(0, BatchPrefillTransformerLayerTaskGraphs.narrowRows(512));
    }

    /** Qwen3-0.6B at 512: seven graphs per prefill family, three decode graphs. */
    @Test
    public void theNarrowGraphsFollowTheFallbackFamily() {
        var layout = new BatchPrefillDecodeForwardTaskGraphLayout(28, 7, 7, 3, 0, 7);
        assertEquals(1, layout.batchLayerIdx(0));
        assertEquals(8, layout.fallbackLayerIdx(0));
        assertEquals(15, layout.narrowLayerIdx(0));
        assertEquals(21, layout.narrowLayerIdx(6));
        assertEquals(22, layout.decodeActivationIdx());
        assertEquals(23, layout.decodeLayerGraphIdx(0));
        assertEquals(26, layout.logitsIdx());
        assertEquals(27, layout.totalGraphs());
        assertEquals(4, layout.layerGraphFamilies());
        assertArrayEquals(new int[] {7, 7, 7, 3}, layout.layerFamilyGraphCounts());
    }

    /** The JIT plan has no fallback family: the narrow graphs follow the primary's. */
    @Test
    public void withoutAFallbackTheyFollowThePrimary() {
        var layout = new BatchPrefillDecodeForwardTaskGraphLayout(28, 7, 0, 3, 2, 7);
        assertEquals(2, layout.batchActivationIdx());
        assertEquals(10, layout.narrowLayerIdx(0));
        assertEquals(17, layout.decodeActivationIdx());
        assertEquals(3, layout.layerGraphFamilies());
        assertArrayEquals(new int[] {7, 7, 3}, layout.layerFamilyGraphCounts());
    }

    /** No narrow family: the layout as it was. */
    @Test
    public void theFiveArgumentLayoutHasNone() {
        var layout = new BatchPrefillDecodeForwardTaskGraphLayout(28, 7, 7, 3, 0);
        assertEquals(0, layout.narrowLayerGraphs());
        assertEquals(15, layout.decodeActivationIdx());
        assertEquals(20, layout.totalGraphs());
    }
}
