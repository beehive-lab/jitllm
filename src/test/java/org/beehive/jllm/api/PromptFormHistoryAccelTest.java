package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * The prompt form of a multi-turn session — each request sends only the new user text, the way
 * {@code jllm chat} does — must leave the cache holding the history the model's template would
 * render.
 *
 * <p>Two defects made later turns end early. The previous response sat in the cache without its
 * terminator, so each new user turn continued an assistant message that was never closed; and a
 * reasoning model's finished {@code <think>} block stayed in the history, which its template never
 * shows, so Qwen3 closed its next turn straight after reasoning, with no answer.
 */
public class PromptFormHistoryAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    private static final String[] TURNS = {
        "hello my name is orion. who are you?",
        "what is my name?",
        "what is the weather in Athens greece right now?"
    };

    @Test
    public void everyReasoningTurnAnswers_cpu() throws Exception {
        everyReasoningTurnAnswers(false);
    }

    @Test
    public void everyReasoningTurnAnswers_gpu() throws Exception {
        everyReasoningTurnAnswers(true);
    }

    private void everyReasoningTurnAnswers(boolean gpu) throws Exception {
        Path model = fixtureOrSkip(Fixture.QWEN3_0_6B_F16);
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, Boolean.toString(gpu));
        try (LocalModel loaded =
                        LocalModels.load(
                                model, ModelOptions.builder().contextLength(4096).build());
                GenerationSession session = ((TextGenerationModel) loaded).newSession()) {
            for (String turn : TURNS) {
                GenerationResult result = session.generate(greedy(turn, 400));
                String text = result.text();
                assertEquals(turn, FinishReason.STOP_TOKEN, result.finishReason());
                int end = text.indexOf("</think>");
                assertTrue("reasoning never closed for: " + turn + "\n" + text, end >= 0);
                assertFalse(
                        "no answer after reasoning for: " + turn + "\n" + text,
                        text.substring(end + "</think>".length()).isBlank());
            }
        } finally {
            restore(previous);
        }
    }

    /** A family without reasoning needs only the closed turn to keep what it was told. */
    @Test
    public void aLaterTurnSeesTheEarlierOne() throws Exception {
        Path model = fixtureOrSkip(Fixture.LLAMA_3_2_1B_Q8_0);
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel loaded =
                        LocalModels.load(
                                model, ModelOptions.builder().contextLength(1024).build());
                GenerationSession session = ((TextGenerationModel) loaded).newSession()) {
            session.generate(greedy("My name is Orion. Please remember it.", 48));
            GenerationResult recall =
                    session.generate(greedy("What is my name? Answer with just the name.", 16));
            assertEquals(FinishReason.STOP_TOKEN, recall.finishReason());
            assertTrue(recall.text(), recall.text().contains("Orion"));
        } finally {
            restore(previous);
        }
    }

    private static GenerationRequest greedy(String prompt, int maxNewTokens) {
        return GenerationRequest.builder()
                .prompt(prompt)
                .maxNewTokens(maxNewTokens)
                .temperature(0f)
                .seed(1L)
                .build();
    }

    private static Path fixtureOrSkip(Fixture fixture) {
        Path model = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), model != null);
        return model;
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(GPU_PROPERTY);
        } else {
            System.setProperty(GPU_PROPERTY, previous);
        }
    }
}
