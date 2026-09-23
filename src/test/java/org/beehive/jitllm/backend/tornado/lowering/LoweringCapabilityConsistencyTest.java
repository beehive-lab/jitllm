package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.EnumSet;
import java.util.Set;
import java.util.TreeSet;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.junit.Test;

/**
 * <b>They did not.</b> {@code LlamaLoweringProvider} and {@code Qwen3LoweringProvider} declared
 * {@code EVERY_MODE} — all three of {@code STANDARD}, {@code PREFILL_DECODE} and {@code
 * BATCH_PREFILL_DECODE} — while {@code LoweredPlanSelection} rejected every phase strategy but
 * single-token. Six declared combinations had therefore never executed a lowered graph, and a
 * qualification matrix built from the declarations would have claimed all six.
 *
 * <p>That is the failure this test exists to make impossible: <b>capability metadata that nothing
 * checks is documentation, and documentation drifts.</b>
 */
public class LoweringCapabilityConsistencyTest {

    /** Every mode any provider declares must be one selection can actually lower. */
    @Test
    public void everyDeclaredModeIsSelectable() {
        Set<String> offenders = new TreeSet<>();
        for (TornadoLoweringProvider provider : TornadoBackendSupport.discover()) {
            for (ExecutionMode mode : provider.supportedModes()) {
                if (!LoweredPlanSelection.SELECTABLE_MODES.contains(mode)) {
                    offenders.add(provider.getClass().getSimpleName() + " declares " + mode);
                }
            }
        }
        assertTrue(
                "a provider may not advertise a mode that LoweredPlanSelection refuses to"
                        + " select — that combination can never execute a lowered graph, and a matrix built"
                        + " from the declaration would claim it anyway. Offenders: "
                        + offenders,
                offenders.isEmpty());
    }

    /** No provider claims a prefill mode. */
    @Test
    public void noProviderClaimsAPrefillMode() {
        Set<ExecutionMode> prefill =
                EnumSet.of(ExecutionMode.PREFILL_DECODE, ExecutionMode.BATCH_PREFILL_DECODE);
        Set<String> offenders = new TreeSet<>();
        for (TornadoLoweringProvider provider : TornadoBackendSupport.discover()) {
            for (ExecutionMode mode : provider.supportedModes()) {
                if (prefill.contains(mode)) {
                    offenders.add(provider.getClass().getSimpleName() + " -> " + mode);
                }
            }
        }
        assertTrue(
                "no lowered prefill graph has ever executed; declaring one is a capability"
                        + " contract the implementation does not honour. Offenders: "
                        + offenders,
                offenders.isEmpty());
    }

    /** Selection lowers exactly one mode today, and that is the fact the matrix rests on. */
    @Test
    public void selectionLowersStandardOnly() {
        assertEquals(EnumSet.of(ExecutionMode.STANDARD), LoweredPlanSelection.SELECTABLE_MODES);
    }

    /**
     * Every qualified combination is declared by a provider.
     *
     * <p>The other direction of the contract: qualification is evidence about something that
     * exists, so a table row naming an architecture no provider lowers would be evidence about
     * nothing.
     */
    @Test
    public void everyQualifiedCombinationIsDeclared() {
        for (LoweringQualification.Combination combination : LoweringQualification.qualified()) {
            TornadoLoweringProvider provider =
                    TornadoBackendSupport.discover().stream()
                            .filter(p -> p.architecture().equals(combination.architecture()))
                            .findFirst()
                            .orElse(null);
            assertFalse(
                    "qualified combination " + combination + " has no lowering provider",
                    provider == null);
            assertTrue(
                    "qualified combination "
                            + combination
                            + " declares a dtype its provider"
                            + " does not support",
                    provider.supportedDataTypes().contains(combination.dtype()));
            assertTrue(
                    "qualified combination "
                            + combination
                            + " declares a mode its provider"
                            + " does not support",
                    provider.supportedModes().contains(combination.mode()));
        }
    }
}
