package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Serialization is asserted the way it is observable — correct interleaved output — not by
 * inspecting a lock.
 */
public class MultiSessionAccelTest {

    private static final int MAX_TOKENS = 96;
    private static final String GPU_PROPERTY = "use.tornadovm";

    @Test
    public void sessionsAreIndependentAndResettable_cpu() throws Exception {
        assertSessionsAreIndependentAndResettable(false);
    }

    @Test
    public void sessionsAreIndependentAndResettable_gpu() throws Exception {
        assertSessionsAreIndependentAndResettable(true);
    }

    private void assertSessionsAreIndependentAndResettable(boolean gpu) throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0),
                    false);
        }
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, Boolean.toString(gpu));
        try (LocalModel loaded =
                LocalModels.load(
                        model,
                        ModelOptions.builder()
                                .contextLength(512)
                                .maxConcurrentSessions(2)
                                .build())) {
            TextGenerationModel generator = (TextGenerationModel) loaded;

            try (GenerationSession alice = generator.newSession();
                    GenerationSession bob = generator.newSession()) {

                // Turn 1, interleaved: each session learns a different name.
                say(alice, "Remember this: my name is Alice. Reply with just: OK");
                say(bob, "Remember this: my name is Bob. Reply with just: OK");

                assertTrue("alice consumed context", alice.position() > 0);
                assertTrue("bob consumed context", bob.position() > 0);

                // Turn 2: recall only works from each session's own retained KV.
                String aliceAnswer = say(alice, "What is my name? Answer with the name only.");
                String bobAnswer = say(bob, "What is my name? Answer with the name only.");

                assertTrue(
                        "session A should recall Alice, got: " + aliceAnswer,
                        aliceAnswer.toLowerCase().contains("alice"));
                assertTrue(
                        "session A must not see session B's context, got: " + aliceAnswer,
                        !aliceAnswer.toLowerCase().contains("bob"));
                assertTrue(
                        "session B should recall Bob, got: " + bobAnswer,
                        bobAnswer.toLowerCase().contains("bob"));
                assertTrue(
                        "session B must not see session A's context, got: " + bobAnswer,
                        !bobAnswer.toLowerCase().contains("alice"));

                // Reset is session-local: it clears the sequence it is called on and nothing else.
                bob.reset();
                assertEquals("reset returns the sequence to an empty context", 0, bob.position());

                String aliceAfterBobReset =
                        say(alice, "What is my name? Answer with the name only.");
                assertTrue(
                        "a neighbour's reset must not clear this session, got: "
                                + aliceAfterBobReset,
                        aliceAfterBobReset.toLowerCase().contains("alice"));
            }
        } finally {
            restore(previous);
        }
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(GPU_PROPERTY);
        } else {
            System.setProperty(GPU_PROPERTY, previous);
        }
    }

    private static String say(GenerationSession session, String prompt) {
        GenerationResult result =
                session.generate(
                        GenerationRequest.builder()
                                .prompt(prompt)
                                .maxNewTokens(MAX_TOKENS)
                                .temperature(0.0f)
                                .build());
        return result.text();
    }
}
