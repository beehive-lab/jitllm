package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/** The schedule helpers; the rendering of real plans is {@code TaskGraphChain*AccelTest}'s. */
public class TaskGraphChainPrinterTest {

    @Test
    public void rangesCollapseRunsOfGraphIndices() {
        assertEquals("[0..7]", TaskGraphChainPrinter.ranges(TaskGraphChainPrinter.span(0, 7)));
        assertEquals(
                "[0] + [8..14]",
                TaskGraphChainPrinter.ranges(
                        TaskGraphChainPrinter.concat(
                                List.of(0), TaskGraphChainPrinter.span(8, 14))));
        assertEquals("[3]", TaskGraphChainPrinter.ranges(List.of(3)));
        assertEquals("", TaskGraphChainPrinter.ranges(List.of()));
    }

    @Test
    public void familiesAreLabelledPerGraph() {
        Map<Integer, String> roles = new HashMap<>();
        TaskGraphChainPrinter.label(roles, 1, 3, "decode layers");
        TaskGraphChainPrinter.label(roles, 4, 1, "logits");
        assertEquals("decode layers 1/3", roles.get(1));
        assertEquals("decode layers 3/3", roles.get(3));
        assertEquals("logits", roles.get(4));
    }
}
