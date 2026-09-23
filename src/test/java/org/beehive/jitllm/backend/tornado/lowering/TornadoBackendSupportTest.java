package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The supported set was taken from [the inventory] (./././././././), which recorded
 * `ForwardPlanFactory`: Llama and Qwen3 run all three execution modes, everything else is
 * `STANDARD`-only, and both weight representations are lowered where a lowering exists at all. This
 * test is what stops that set drifting silently.
 */
public class TornadoBackendSupportTest {

    private static final CompileOptions OPTIONS = new CompileOptions(false);
    private static final DeviceCapabilities CAPABILITIES =
            DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

    /** Llama and Qwen3 lower both representations — in {@code STANDARD} only. */
    @Test
    public void llamaAndQwen3LowerBothRepresentationsInStandardOnly() {
        for (String architecture : new String[] {"llama", "qwen3"}) {
            for (DataType dtype : new DataType[] {DataType.F16, DataType.Q8_0}) {
                ArchitectureId id = ArchitectureId.of(architecture);
                assertTrue(
                        architecture + " + " + dtype + " + STANDARD",
                        TornadoBackendSupport.supports(id, dtype, ExecutionMode.STANDARD));
                assertFalse(
                        architecture + " must not claim a lowered PREFILL_DECODE",
                        TornadoBackendSupport.supports(id, dtype, ExecutionMode.PREFILL_DECODE));
                assertFalse(
                        architecture + " must not claim a lowered BATCH_PREFILL_DECODE",
                        TornadoBackendSupport.supports(
                                id, dtype, ExecutionMode.BATCH_PREFILL_DECODE));
            }
        }
    }

    /** And STANDARD only for the rest, which is what the factory has always enforced. */
    @Test
    public void theStandardOnlyFamiliesAreStandardOnly() {
        for (String architecture :
                new String[] {"mistral", "qwen2", "deepseek-r1-distill-qwen", "granite", "phi3"}) {
            ArchitectureId id = ArchitectureId.of(architecture);
            assertTrue(
                    architecture + " runs STANDARD",
                    TornadoBackendSupport.supports(id, DataType.F16, ExecutionMode.STANDARD));
            assertFalse(
                    architecture + " must not claim PREFILL_DECODE",
                    TornadoBackendSupport.supports(id, DataType.F16, ExecutionMode.PREFILL_DECODE));
            assertFalse(
                    architecture + " must not claim BATCH_PREFILL_DECODE",
                    TornadoBackendSupport.supports(
                            id, DataType.F16, ExecutionMode.BATCH_PREFILL_DECODE));
        }
    }

    /**
     * A family with no lowering is unsupported, not an error — and four of ten are in that state.
     *
     * <p>Devstral, Gemma4 and Qwen2-MoE load and run on the legacy plan path. Reporting that as a
     * failure here would turn a working configuration into a broken one.
     */
    @Test
    public void familiesWithoutALoweringAreUnsupportedRatherThanBroken() {
        for (String architecture : new String[] {"devstral", "gemma4", "qwen2-moe"}) {
            assertFalse(
                    architecture + " has no Tornado lowering yet",
                    TornadoBackendSupport.supports(
                            ArchitectureId.of(architecture), DataType.F16, ExecutionMode.STANDARD));
        }
    }

    /** An unregistered architecture names itself, the triple, the backend and what is lowered. */
    @Test
    public void anUnregisteredArchitectureFailsByName() {
        try {
            TornadoBackendSupport.lowering(
                    ArchitectureId.of("gemma4"),
                    DataType.F16,
                    ExecutionMode.STANDARD,
                    OPTIONS,
                    CAPABILITIES);
            fail("gemma4 has no lowering; asking for one must be refused");
        } catch (UnsupportedOperationException expected) {
            String message = expected.getMessage();
            assertTrue(message, message.contains("gemma4"));
            assertTrue("must name the backend: " + message, message.contains("tornado"));
            assertTrue("must name the dtype: " + message, message.contains("F16"));
            assertTrue("must name the mode: " + message, message.contains("STANDARD"));
            assertTrue("must say what it does lower: " + message, message.contains("llama"));
            assertFalse(
                    "must not claim what the caller will do — that is selection policy, and it"
                            + " stops being true when the legacy path goes: "
                            + message,
                    message.contains("legacy"));
        }
    }

    /** The synthetic provider is discovered without any production file being edited. */
    @Test
    public void anArchitectureIsAddedByFilesAndAServiceLineAlone() {
        assertTrue(
                "the synthetic provider must be discovered: " + TornadoBackendSupport.registered(),
                TornadoBackendSupport.registered().contains(SyntheticLoweringProvider.ID));
        assertTrue(
                "and its supported triple must resolve",
                TornadoBackendSupport.supports(
                        SyntheticLoweringProvider.ID, DataType.F16, ExecutionMode.STANDARD));
        assertEquals(
                "to a lowering under its own identity",
                SyntheticLoweringProvider.ID,
                TornadoBackendSupport.lowering(
                                SyntheticLoweringProvider.ID,
                                DataType.F16,
                                ExecutionMode.STANDARD,
                                OPTIONS,
                                CAPABILITIES)
                        .architecture());
    }

    /** Its unsupported dtype and mode fail by all the relevant names. */
    @Test
    public void theSyntheticArchitecturesUnsupportedCombinationsFailByName() {
        try {
            TornadoBackendSupport.lowering(
                    SyntheticLoweringProvider.ID,
                    DataType.Q8_0,
                    ExecutionMode.STANDARD,
                    OPTIONS,
                    CAPABILITIES);
            fail("the synthetic provider lowers F16 only");
        } catch (UnsupportedOperationException expected) {
            String message = expected.getMessage();
            assertTrue(
                    message,
                    message.contains("Q8_0")
                            && message.contains(SyntheticLoweringProvider.ID.name())
                            && message.contains("STANDARD")
                            && message.contains("tornado"));
            assertTrue("must say what it lowers: " + message, message.contains("F16"));
        }
        try {
            TornadoBackendSupport.lowering(
                    SyntheticLoweringProvider.ID,
                    DataType.F16,
                    ExecutionMode.PREFILL_DECODE,
                    OPTIONS,
                    CAPABILITIES);
            fail("the synthetic provider runs STANDARD only");
        } catch (UnsupportedOperationException expected) {
            String message = expected.getMessage();
            assertTrue(
                    message,
                    message.contains("PREFILL_DECODE")
                            && message.contains(SyntheticLoweringProvider.ID.name())
                            && message.contains("F16")
                            && message.contains("tornado"));
        }
    }

    /** Two providers for one identity are refused, naming both classes. */
    @Test
    public void duplicateProvidersFailNamingBothClasses() {
        var duplicate = new SyntheticLoweringProvider();
        try {
            TornadoBackendSupport.index(List.of(new SyntheticLoweringProvider(), duplicate));
            fail("two providers claiming one identity must be refused");
        } catch (IllegalStateException expected) {
            String message = expected.getMessage();
            assertTrue(
                    "must name the identity: " + message,
                    message.contains(SyntheticLoweringProvider.ID.name()));
            assertTrue(
                    "must name the class: " + message,
                    message.contains(SyntheticLoweringProvider.class.getName()));
            assertTrue("must name the backend: " + message, message.contains("tornado"));
        }
    }

    /**
     * The index is what refuses duplicates, so discovery order cannot change the outcome.
     *
     * <p>Both orders produce the same message. A registry that took the first provider it saw would
     * pass one of these and fail the other, and a wrong-architecture lowering produces plausible
     * output rather than a crash.
     */
    @Test
    public void discoveryOrderCannotChangeTheResult() {
        var a = new SyntheticLoweringProvider();
        var b = new SyntheticLoweringProvider();
        String first = messageOf(() -> TornadoBackendSupport.index(List.of(a, b)));
        String second = messageOf(() -> TornadoBackendSupport.index(List.of(b, a)));
        assertEquals("the same message whichever order they arrived in", first, second);
    }

    private static String messageOf(Runnable body) {
        try {
            body.run();
            return "no failure";
        } catch (RuntimeException e) {
            return e.getMessage();
        }
    }

    /** An unsupported mode names the architecture, the dtype, the mode and the backend. */
    @Test
    public void anUnsupportedModeNamesAllFour() {
        try {
            TornadoBackendSupport.lowering(
                    ArchitectureId.of("granite"),
                    DataType.Q8_0,
                    ExecutionMode.BATCH_PREFILL_DECODE,
                    OPTIONS,
                    CAPABILITIES);
            fail("granite has no batch prefill plan; asking for one must be refused");
        } catch (UnsupportedOperationException expected) {
            String message = expected.getMessage();
            assertTrue("architecture: " + message, message.contains("granite"));
            assertTrue("dtype: " + message, message.contains("Q8_0"));
            assertTrue("mode: " + message, message.contains("BATCH_PREFILL_DECODE"));
            assertTrue("backend: " + message, message.contains("tornado"));
        }
    }

    /** An unsupported dtype names what the architecture does lower. */
    @Test
    public void anUnsupportedDtypeSaysWhatIsLowered() {
        try {
            TornadoBackendSupport.lowering(
                    ArchitectureId.of("llama"),
                    DataType.F32,
                    ExecutionMode.STANDARD,
                    OPTIONS,
                    CAPABILITIES);
            fail("F32 is not lowered; asking must be refused");
        } catch (UnsupportedOperationException expected) {
            String message = expected.getMessage();
            assertTrue(message, message.contains("F32") && message.contains("llama"));
            assertTrue(
                    "must say what it does lower: " + message,
                    message.contains("F16") && message.contains("Q8_0"));
        }
    }

    /** The alias resolves to its own identity's lowering, not the family it delegates to. */
    @Test
    public void anAliasResolvesUnderItsOwnIdentity() {
        FamilyLowering mistral =
                TornadoBackendSupport.lowering(
                        ArchitectureId.of("mistral"),
                        DataType.Q8_0,
                        ExecutionMode.STANDARD,
                        OPTIONS,
                        CAPABILITIES);
        assertTrue(
                "the lowering must validate mistral programs, not llama ones",
                mistral.architecture().equals(ArchitectureId.of("mistral")));
    }
}
