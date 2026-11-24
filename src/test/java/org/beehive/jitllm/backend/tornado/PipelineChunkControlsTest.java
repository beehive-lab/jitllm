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
