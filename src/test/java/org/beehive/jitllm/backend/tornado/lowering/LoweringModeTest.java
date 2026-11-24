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
package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

public class LoweringModeTest {

    @Test
    public void unsetMeansAuto() {
        assertEquals(LoweringMode.AUTO, LoweringMode.parse(null, LoweringMode.AUTO));
        assertEquals(LoweringMode.AUTO, LoweringMode.parse("  ", LoweringMode.AUTO));
    }

    @Test
    public void theThreeModesParse() {
        assertEquals(LoweringMode.AUTO, LoweringMode.parse("auto", LoweringMode.OFF));
        assertEquals(LoweringMode.ON, LoweringMode.parse("on", LoweringMode.OFF));
        assertEquals(LoweringMode.OFF, LoweringMode.parse("off", LoweringMode.AUTO));
        assertEquals(
                LoweringMode.ON,
                LoweringMode.parse(
                        "  AUTO ".trim().equals("AUTO") ? "on" : "on", LoweringMode.OFF));
    }

    /**
     * The booleans keep working, because every existing script, test and invocation uses them. A
     * migration that broke them would be a breaking change dressed as a rename.
     */
    @Test
    public void booleansStillWork() {
        assertEquals(LoweringMode.ON, LoweringMode.parse("true", LoweringMode.OFF));
        assertEquals(LoweringMode.OFF, LoweringMode.parse("false", LoweringMode.AUTO));
        assertEquals(LoweringMode.ON, LoweringMode.parse("TRUE", LoweringMode.OFF));
    }

    /** A typo must not silently degrade to a path the user did not choose. */
    @Test
    public void anUnrecognisedValueThrowsRatherThanGuessing() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> LoweringMode.parse("yes", LoweringMode.AUTO));
        assertTrue("the message must name the offending value", e.getMessage().contains("yes"));
    }

    /**
     * The qualified set is exactly Llama F16 STANDARD.
     *
     * <p>Pinned as an equality rather than a containment check: the risk this milestone actually
     * carries is a combination being added without its evidence, and a containment check would not
     * notice.
     */
    @Test
    public void exactlyOneCombinationIsQualified() {
        assertEquals(1, LoweringQualification.qualified().size());
        assertTrue(
                LoweringQualification.isQualified(
                        ArchitectureId.of("llama"), DataType.F16, ExecutionMode.STANDARD));
    }

    /** Qualification is per dtype and per mode, never per family. */
    @Test
    public void qualificationDoesNotSpreadAcrossDtypeOrMode() {
        assertFalse(
                "Llama Q8_0 compiles different graphs and has not been measured",
                LoweringQualification.isQualified(
                        ArchitectureId.of("llama"), DataType.Q8_0, ExecutionMode.STANDARD));
        assertFalse(
                "no lowered prefill graph has ever executed",
                LoweringQualification.isQualified(
                        ArchitectureId.of("llama"), DataType.F16, ExecutionMode.PREFILL_DECODE));
        assertFalse(
                "Qwen3 is implemented but unmeasured",
                LoweringQualification.isQualified(
                        ArchitectureId.of("qwen3"), DataType.F16, ExecutionMode.STANDARD));
    }
}
