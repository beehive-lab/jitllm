package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * Qwen3.8-27B through the public route on an accelerator: load, session, generate, reset, close.
 *
 * <p>Reset matters more here than for any other family. A key/value cache does not need clearing
 * between sequences — attention reads no further than the current position — but this stack is
 * three quarters recurrent, and a recurrent state has no such mask: whatever the previous sequence
 * summed into it conditions the next one. On the device that state is a buffer updated in place
 * across tokens and never read back, so a reset that cleared only the host copy would leave the
 * device continuing the old sequence, and the symptom would be a second answer that is fluent and
 * subtly wrong rather than an error.
 *
 * <p>Greedy at a fixed seed throughout, which is what makes token identity a property rather than a
 * hope.
 */
// @formatter:on
public class Qwen35LifecycleAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 512;
    private static final int MAX_NEW = 24;

    private static final String PROMPT = "What is the capital of France? Answer in one word.";

    @Test
    public void aSessionGeneratesResetsAndClosesOnTheDevice() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;
            // Thinking off: the answer has to land inside the token budget, and this family
            // spends a paragraph reasoning before it otherwise.
            GenerationSession session =
                    generation.newSession(
                            SessionOptions.builder().thinkingMode(ThinkingMode.DISABLED).build());
            String first;
            String repeated;
            String afterReset;
            try {
                GenerationRequest request = request();
                first = session.generate(request).text();
                assertFalse("nothing was generated, so the case proves nothing", first.isEmpty());

                // Proved, not assumed. A silent host fallback produces fluent text at a plausible
                // rate, and every assertion below would pass on it.
                var snapshot = org.beehive.jitllm.auxiliary.RunMetrics.snapshot();
                assertEquals("the plan that ran", "legacy", snapshot.executionPath());
                assertEquals(
                        "the tuple that ran",
                        "qwen35/Q4_0/STANDARD",
                        snapshot.executionCombination());
                assertTrue(
                        "the model answered '" + first + "'",
                        first.toLowerCase().contains("paris"));

                // Same session, same request, no reset: the conversation surface resets
                // transparently when the prefix does not extend, so this is the ordinary
                // second turn.
                repeated = session.generate(request).text();

                session.reset();
                afterReset = session.generate(request).text();
            } finally {
                session.close();
            }

            assertEquals(
                    "a second identical request answers the same way, so nothing carried over"
                            + " from the first — on the device, that means the recurrent state was"
                            + " cleared there and not only on the host",
                    first,
                    repeated);
            assertEquals("an explicit reset returns the session to a new one", first, afterReset);

            assertThrows(
                    "a closed session must refuse work rather than use freed device buffers",
                    IllegalStateException.class,
                    () -> session.generate(request()));
        } finally {
            restore(previous);
        }
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder()
                .prompt(PROMPT)
                .maxNewTokens(MAX_NEW)
                .temperature(0.0f)
                .seed(1234L)
                .build();
    }

    private static ModelOptions options() {
        return ModelOptions.builder().contextLength(CONTEXT).build();
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(GPU_PROPERTY);
        } else {
            System.setProperty(GPU_PROPERTY, previous);
        }
    }

    private static Path fixtureOrSkip() {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent: fixture " + Fixture.QWEN3_8_27B_Q4_0.fileName, false);
        }
        return modelPath;
    }
}
