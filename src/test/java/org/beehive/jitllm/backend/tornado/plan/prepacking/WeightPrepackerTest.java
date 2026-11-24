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
package org.beehive.jitllm.backend.tornado.plan.prepacking;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;

/**
 * The path of a plan that does not use weight prepacking, which needs no device: {@link
 * WeightPrepacker#buildTaskGraphs} with no weight registered, and {@link
 * WeightPrepackingTaskGraphs#NONE}.
 */
public class WeightPrepackerTest {

    @Test
    public void withNoWeightRegisteredNothingIsBuilt() {
        WeightPrepackingTaskGraphs prepacking = new WeightPrepacker().buildTaskGraphs();

        assertSame(
                "the shared empty bundle: no graph, scratch buffer or flag was created",
                WeightPrepackingTaskGraphs.NONE,
                prepacking);
        assertEquals(0, prepacking.graphCount());
        assertTrue(prepacking.getImmutableTaskGraphs().isEmpty());
    }

    @Test
    public void noPrepackingGraphsAddNoGridsAndTouchNoPlan() {
        GridScheduler scheduler = new GridScheduler();
        WeightPrepackingTaskGraphs.NONE.updateGridScheduler(scheduler);
        // A null plan would fail on any selection or execution.
        WeightPrepackingTaskGraphs.NONE.run(null, scheduler);
    }
}
