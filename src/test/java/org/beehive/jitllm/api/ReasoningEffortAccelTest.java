package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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
 * A reasoning effort on the real vocabularies, and when one is refused.
 *
 * <p>Class B: runs under {@code -Paccel-tests} with the pinned fixtures, on the CPU. Qwen3.8 has
 * the control; Qwen3 has a reasoning phase but no effort control, which makes the rejection
 * testable. {@link ReasoningEffortTemplateTest} checks the rendering itself against the template.
 */
public class ReasoningEffortAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final List<ChatMessage> ASK =
            List.of(ChatMessage.of(ChatRole.USER, "What is 2 + 2?"));

    /** Qwen3.8: the template's default is xhigh, and each effort writes its own instruction. */
    @Test
    public void qwen38WritesTheInstructionOfEachEffort() throws Exception {
        Model model = loadOrSkip(Fixture.QWEN3_8_27B_Q4_0);

        String byDefault = decode(model, ThinkingMode.DEFAULT, ReasoningEffort.DEFAULT);
        assertTrue(byDefault, byDefault.contains("Reasoning effort is set to xhigh."));
        assertEquals(byDefault, decode(model, ThinkingMode.DEFAULT, ReasoningEffort.XHIGH));
        assertTrue(
                decode(model, ThinkingMode.DEFAULT, ReasoningEffort.LOW)
                        .contains("Reasoning effort is set to low."));
        String medium = decode(model, ThinkingMode.DEFAULT, ReasoningEffort.MEDIUM);
        assertFalse(medium, medium.contains("Reasoning effort"));
        String disabled = decode(model, ThinkingMode.DISABLED, ReasoningEffort.DEFAULT);
        assertFalse(disabled, disabled.contains("Reasoning effort"));
    }

    /** With thinking off there is nothing for an effort to apply to, so asking for one fails. */
    @Test
    public void anExplicitEffortWithThinkingDisabledIsRejected() throws Exception {
        Path modelPath = fixtureOrSkip(Fixture.QWEN3_8_27B_Q4_0);
        onCpu(
                () -> {
                    try (LocalModel model =
                            LocalModels.load(
                                    modelPath,
                                    ModelOptions.builder()
                                            .contextLength(512)
                                            .reasoningEffort(ReasoningEffort.LOW)
                                            .build())) {
                        TextGenerationModel generation = (TextGenerationModel) model;
                        assertTrue(model.info().capabilities().reasoningEffortControl());

                        IllegalArgumentException thrown =
                                assertThrows(
                                        IllegalArgumentException.class,
                                        () ->
                                                generation.newSession(
                                                        SessionOptions.builder()
                                                                .thinkingMode(ThinkingMode.DISABLED)
                                                                .build()));
                        assertTrue(thrown.getMessage(), thrown.getMessage().contains("DISABLED"));

                        // The model's own effort, with thinking left on, opens.
                        try (GenerationSession session = generation.newSession()) {
                            assertEquals(0, session.position());
                        }
                    }
                });
    }

    /** Asking a family whose template has no effort control fails, and says why. */
    @Test
    public void anExplicitEffortOnAFamilyWithoutTheControlIsRejected() throws Exception {
        Path modelPath = fixtureOrSkip(Fixture.QWEN3_0_6B_Q8_0);
        onCpu(
                () -> {
                    try (LocalModel model =
                            LocalModels.load(
                                    modelPath,
                                    ModelOptions.builder()
                                            .contextLength(512)
                                            .reasoningEffort(ReasoningEffort.XHIGH)
                                            .build())) {
                        TextGenerationModel generation = (TextGenerationModel) model;
                        assertFalse(model.info().capabilities().reasoningEffortControl());

                        IllegalArgumentException thrown =
                                assertThrows(
                                        IllegalArgumentException.class, generation::newSession);
                        assertTrue(thrown.getMessage(), thrown.getMessage().contains("XHIGH"));
                        assertTrue(
                                thrown.getMessage(),
                                thrown.getMessage().contains("no reasoning-effort control"));

                        // The escape hatch: a session may return to the family's own behaviour.
                        try (GenerationSession session =
                                generation.newSession(
                                        SessionOptions.builder()
                                                .reasoningEffort(ReasoningEffort.DEFAULT)
                                                .build())) {
                            assertEquals(0, session.position());
                        }
                    }
                });
    }

    private interface Body {
        void run() throws Exception;
    }

    private static void onCpu(Body body) throws Exception {
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try {
            body.run();
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }

    private static String decode(Model model, ThinkingMode thinking, ReasoningEffort effort) {
        return model.tokenizer()
                .decode(new ConversationEncoder(model, thinking, effort).encode(ASK, List.of()));
    }

    private static Model loadOrSkip(Fixture fixture) throws Exception {
        return ModelLoader.loadModel(fixtureOrSkip(fixture), 512, false, false);
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
