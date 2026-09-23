package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Test;

/**
 * What decides whether a model reserves the shared key/value pool: whether the lowered path can be
 * selected for it, not whether lowering is merely enabled. {@code auto} is enabled for every model
 * and lowers only the qualified set, so answering {@link LoweredPlanSelection#enabled()} here gave
 * every legacy session a pool sized for sessions it never shares with.
 */
public class LoweredPoolSelectionTest {

    private static final ArchitectureId LLAMA = ArchitectureId.of("llama");
    private static final ArchitectureId QWEN3 = ArchitectureId.of("qwen3");

    private static final ExecutionPolicy SINGLE_TOKEN =
            ExecutionPolicy.builder().phaseStrategy(PhaseStrategy.SINGLE_TOKEN).build();
    private static final ExecutionPolicy PREFILL_DECODE =
            ExecutionPolicy.builder().phaseStrategy(PhaseStrategy.PREFILL_DECODE).build();

    private final String previous = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);

    @After
    public void restore() {
        if (previous == null) System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        else System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, previous);
    }

    @Test
    public void autoPredictsLoweringOnlyForAQualifiedArchitectureAndMode() {
        System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        assertTrue("auto is enabled for every model", LoweredPlanSelection.enabled());
        assertTrue(LoweredPlanSelection.mayHandle(LLAMA, DataType.F16, SINGLE_TOKEN));
        assertFalse(LoweredPlanSelection.mayHandle(LLAMA, DataType.Q8_0, SINGLE_TOKEN));
        assertFalse(LoweredPlanSelection.mayHandle(QWEN3, DataType.F16, SINGLE_TOKEN));
        assertFalse(LoweredPlanSelection.mayHandle(LLAMA, DataType.F16, PREFILL_DECODE));
        // An unknown weight type assumes the lowered path, so the pool is over- not under-counted.
        assertTrue(LoweredPlanSelection.mayHandle(LLAMA, null, SINGLE_TOKEN));
    }

    @Test
    public void autoRejectsAnUnselectablePolicyBeforeLookingAtTheModel() {
        System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        // A prefill/decode policy never lowers, whatever the model: answered from the policy alone.
        assertFalse(LoweredPlanSelection.mayHandle(null, PREFILL_DECODE));
    }

    @Test
    public void offNeverLowersAndOnAlwaysNeedsThePool() {
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
        assertFalse(LoweredPlanSelection.mayHandle(null, SINGLE_TOKEN));
        assertFalse(LoweredPlanSelection.mayHandle(LLAMA, DataType.F16, SINGLE_TOKEN));

        // ON fails loudly later for an unimplemented combination; until then it requires the pool.
        System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "on");
        assertTrue(LoweredPlanSelection.mayHandle(null, PREFILL_DECODE));
        assertTrue(LoweredPlanSelection.mayHandle(QWEN3, DataType.Q8_0, PREFILL_DECODE));
    }
}
