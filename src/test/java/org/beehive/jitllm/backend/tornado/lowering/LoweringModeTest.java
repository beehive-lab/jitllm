package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;
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
