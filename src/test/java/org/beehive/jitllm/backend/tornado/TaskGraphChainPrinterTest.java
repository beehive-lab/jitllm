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
