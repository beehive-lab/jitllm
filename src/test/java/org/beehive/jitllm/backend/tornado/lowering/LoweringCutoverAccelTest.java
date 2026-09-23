package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * This is the only test that asserts the <i>default</i>. Every other lowering test sets the
 * property, so all of them would still pass if the default silently reverted — which is exactly how
 * a cutover regresses without anyone noticing.
 *
 * <p><b>The counter, not the property.</b> Reading {@code jitllm.lowering} back proves the path was
 * asked for; {@code loweredPlanCount()} proves a lowered plan was built. This project has already
 * recorded one accelerator gate that passed green while the flag never reached the JVM.
 */
public class LoweringCutoverAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    /** The qualified combination lowers by default — no property, no opt-in. */
    @Test
    public void llamaF16StandardLowersByDefault() throws Exception {
        assertLoweredByDefault(Fixture.LLAMA_3_2_1B_F16, true);
    }

    /** Llama Q8_0 does not, and that is the point of keying qualification on the dtype. */
    @Test
    public void llamaQ8_0StandardStaysLegacyByDefault() throws Exception {
        assertLoweredByDefault(Fixture.LLAMA_3_2_1B_Q8_0, false);
    }

    private static void assertLoweredByDefault(Fixture fixture, boolean expectLowered)
            throws Exception {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        // Cleared, deliberately: this test is about what happens when a user sets nothing at all.
        System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        try {
            assertEquals(
                    "an unset property must mean auto",
                    LoweringMode.AUTO,
                    LoweredPlanSelection.mode());
            long before = LoweredPlanSelection.loweredPlanCount();
            try (LocalModel loaded =
                    LocalModels.load(model, ModelOptions.builder().contextLength(256).build())) {
                TextGenerationModel generator = (TextGenerationModel) loaded;
                try (GenerationSession session = generator.newSession()) {
                    GenerationResult result =
                            session.generate(
                                    GenerationRequest.builder()
                                            .prompt("Name one colour.")
                                            .maxNewTokens(16)
                                            .temperature(0f)
                                            .seed(1234L)
                                            .build());
                    assertTrue(
                            "the run must actually generate",
                            result.text() != null && !result.text().isEmpty());
                }
            }
            long lowered = LoweredPlanSelection.loweredPlanCount() - before;
            if (expectLowered) {
                assertTrue(
                        "a qualified combination must lower with nothing set; it built "
                                + lowered
                                + " lowered plans",
                        lowered > 0);
            } else {
                assertEquals(
                        "an unqualified combination must select legacy under auto,"
                                + " deliberately and silently",
                        0,
                        lowered);
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
