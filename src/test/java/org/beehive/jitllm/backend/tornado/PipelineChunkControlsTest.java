package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * {@link TornadoVMMasterPlanPipeline#stageControls}: a stage reads a chunk's start position and
 * active rows as the session staged them, and its own key/value slot. The session's slot addresses
 * a shared pool's block table; a stage owns a one-sequence table, addressed at its own slot.
 */
public class PipelineChunkControlsTest {

    @Test
    public void aStageKeepsThePositionAndRowsAndTakesItsOwnSlot() {
        IntArray holder = new IntArray(3);
        // Second session of a shared pool: slot 1. The stage's private table has slot 0 only.
        TornadoVMMasterPlanPipeline.stageControls(new int[] {256, 128, 1}, holder, 0);
        assertArrayEquals(new int[] {256, 128, 0}, holder.toHeapArray());
    }

    @Test
    public void everyChunkOfAnOverlappedPrefillIsTranslated() {
        IntArray first = new IntArray(3);
        IntArray second = new IntArray(3);
        int[][] published = {{0, 128, 3}, {128, 128, 3}, {256, 40, 3}};
        for (int[] chunk : published) {
            TornadoVMMasterPlanPipeline.stageControls(chunk, first, 0);
            TornadoVMMasterPlanPipeline.stageControls(chunk, second, 0);
            int[] expected = {chunk[0], chunk[1], 0};
            assertArrayEquals(expected, first.toHeapArray());
            assertArrayEquals(expected, second.toHeapArray());
        }
        // The published controls themselves are not rewritten: another stage still reads them.
        assertArrayEquals(new int[] {256, 40, 3}, published[2]);
    }

    @Test
    public void aStageWithANonZeroSlotUsesIt() {
        IntArray holder = new IntArray(3);
        TornadoVMMasterPlanPipeline.stageControls(new int[] {16, 1, 0}, holder, 2);
        assertArrayEquals(new int[] {16, 1, 2}, holder.toHeapArray());
    }
}
