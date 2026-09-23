package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.junit.Test;

/**
 * The factory chose between three plan classes on two {@code static final} fields read from system
 * properties. Asserting the default still works would prove nothing about the migration — the
 * default is what a property-derived constant already gave. So this asserts the <b>non-default</b>
 * too: a policy naming prefill/decode must produce the prefill/decode plan, in the same JVM as one
 * that does not.
 */
public class PhaseStrategyAccelTest {

    private static final int CONTEXT_LENGTH = 512;

    @Test
    public void bothPhaseStrategiesSelectTheirPlan() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousGpu = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Model loaded = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);

            TornadoVMMasterPlan singleToken = planFor(loaded, ExecutionPolicy.builder().build());
            try {
                assertTrue(
                        "a single-token policy must build the single-token plan, and it is the"
                                + " default so this half would pass either way",
                        singleToken instanceof TornadoVMMasterPlanSingleToken);
            } finally {
                singleToken.freeTornadoExecutionPlan();
            }

            TornadoVMMasterPlan prefillDecode =
                    planFor(
                            loaded,
                            ExecutionPolicy.builder()
                                    .phaseStrategy(PhaseStrategy.PREFILL_DECODE)
                                    .build());
            try {
                assertTrue(
                        "a prefill/decode policy must build the prefill/decode plan — this is"
                                + " the half that shows the resolved value reached the factory,"
                                + " and it was impossible while the choice was a class constant",
                        prefillDecode instanceof TornadoVMMasterPlanPrefillDecode);
            } finally {
                prefillDecode.freeTornadoExecutionPlan();
            }
        } finally {
            if (previousGpu == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previousGpu);
            }
        }
    }

    private static TornadoVMMasterPlan planFor(Model model, ExecutionPolicy policy) {
        State state = model.createNewState();
        state.resolveExecutionPolicy(policy);
        return TornadoVMMasterPlan.initializeTornadoVMPlan(state, model, MetricsSink.disabled());
    }
}
