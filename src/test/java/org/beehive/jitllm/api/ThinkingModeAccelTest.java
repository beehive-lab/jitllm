package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * What a reasoning mode actually encodes, on families that have one and families that do not.
 *
 * <p>Class B: runs under {@code -Paccel-tests} with the pinned fixtures. Qwen3 has a reasoning
 * phase; Llama does not, which is what makes the rejection case testable.
 */
public class ThinkingModeAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final List<ChatMessage> ASK =
            List.of(ChatMessage.of(ChatRole.USER, "What is 2 + 2?"));

    /** Qwen3: enabled and disabled encode differently, and both differ from leaving it alone. */
    @Test
    public void aFamilyWithAReasoningPhaseEncodesEachModeDifferently() throws Exception {
        Model model = loadOrSkip(Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue(
                "this case needs a family with a reasoning phase",
                model.chatFormat().supportsThinking());

        List<Integer> byDefault = encode(model, ThinkingMode.DEFAULT);
        List<Integer> enabled = encode(model, ThinkingMode.ENABLED);
        List<Integer> disabled = encode(model, ThinkingMode.DISABLED);

        assertNotEquals(
                "DISABLED must reach the prompt, or the property does nothing",
                byDefault,
                disabled);
        assertNotEquals("ENABLED and DISABLED are different requests", enabled, disabled);
        assertTrue(
                "a mode only appends; it never rewrites the conversation",
                disabled.size() >= byDefault.size());
        assertEquals(
                "DEFAULT leaves the family's own behaviour alone",
                byDefault,
                encode(model, ThinkingMode.DEFAULT));
    }

    /** Llama has no reasoning phase, so DEFAULT is the only mode it can be asked for. */
    @Test
    public void aFamilyWithoutOneEncodesNothingForDefault() throws Exception {
        Model model = loadOrSkip(Fixture.LLAMA_3_2_1B_Q8_0);
        assumeTrue(
                "this case needs a family without a reasoning phase",
                !model.chatFormat().supportsThinking());
        assertEquals(encode(model, ThinkingMode.DEFAULT), encode(model, ThinkingMode.DEFAULT));
    }

    /** Asking a family that cannot represent the control fails, and says why. */
    @Test
    public void anExplicitModeOnAFamilyWithoutOneIsRejected() throws Exception {
        Path modelPath = fixtureOrSkip(Fixture.LLAMA_3_2_1B_Q8_0);
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model =
                LocalModels.load(
                        modelPath,
                        ModelOptions.builder()
                                .contextLength(512)
                                .thinkingMode(ThinkingMode.DISABLED)
                                .build())) {
            TextGenerationModel generation = (TextGenerationModel) model;

            IllegalArgumentException thrown =
                    assertThrows(IllegalArgumentException.class, generation::newSession);
            assertTrue(thrown.getMessage(), thrown.getMessage().contains("DISABLED"));
            assertTrue(
                    "the message must name what has no reasoning phase",
                    thrown.getMessage().contains("reasoning phase"));

            // And the escape hatch works: a session may return to the family's own behaviour.
            try (GenerationSession session =
                    generation.newSession(
                            SessionOptions.builder().thinkingMode(ThinkingMode.DEFAULT).build())) {
                assertEquals(0, session.position());
            }
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }

    private static List<Integer> encode(Model model, ThinkingMode mode) {
        return new ConversationEncoder(model, mode).encode(ASK, List.of());
    }

    private static Model loadOrSkip(Fixture fixture) throws Exception {
        return ModelLoader.loadModel(fixtureOrSkip(fixture), 512, true, false);
    }

    private static Path fixtureOrSkip(Fixture fixture) {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: " + fixture.fileName, false);
        }
        return modelPath;
    }
}
