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
