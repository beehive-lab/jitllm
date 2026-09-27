package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * {@link DecisionSession} must be a pure optimisation: scoring several questions off one prefilled
 * prefix has to give the logits each question gets when it is scored on its own from position 0.
 * A wrong rewind position or a stale seed shows up here as a different winner or drifted logits.
 */
public class DecisionSessionAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 1024;

    private static final String STATE =
            "Ticket from a conference attendee: 'The wifi in room 7 has been down for an hour and my"
                    + " talk starts in ten minutes. I need the slides online NOW.'";

    private static final String[] QUESTIONS = {
        "Is this urgent?\nA) Yes\nB) No\nAnswer with the letter only.",
        "Which team should handle it?\nA) Venue and network\nB) Registration\nC) Catering\n"
                + "D) Speakers desk\nAnswer with the letter only.",
        "How stressed is the attendee?\nA) Calm\nB) Somewhat stressed\nC) Very stressed\n"
                + "Answer with the letter only.",
    };

    @Test
    public void branchedScoresMatchFreshScoresOnTheHost() throws Exception {
        compare(false, 1e-3);
    }

    @Test
    public void branchedScoresMatchFreshScoresOnTheGpu() throws Exception {
        compare(true, 0.15);
    }

    private static void compare(boolean gpu, double tolerance) throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, Boolean.toString(gpu));
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;
            try (DecisionSession decisions = DecisionSession.open(generation)) {
                List<List<ChatMessage>> conversations = new ArrayList<>();
                int[][] candidates = new int[QUESTIONS.length][];
                for (int q = 0; q < QUESTIONS.length; q++) {
                    conversations.add(
                            List.of(ChatMessage.of(ChatRole.USER, STATE + "\n\n" + QUESTIONS[q])));
                    candidates[q] = letters(decisions, 4);
                }

                DecisionSession.Scored branched =
                        decisions.scoreConversations(conversations, candidates);
                assertTrue("the prefix must actually be shared", branched.sharedTokens() > 20);

                for (int q = 0; q < QUESTIONS.length; q++) {
                    float[] alone =
                            decisions
                                    .scoreConversations(
                                            List.of(conversations.get(q)), new int[][] {candidates[q]})
                                    .logits()[0];
                    float[] shared = branched.logits()[q];
                    assertEquals(
                            "question " + q + ": the winning option must not depend on branching",
                            argmax(alone),
                            argmax(shared));
                    for (int i = 0; i < alone.length; i++) {
                        assertEquals(
                                "question " + q + ", candidate " + i + " (gpu=" + gpu + ")",
                                alone[i],
                                shared[i],
                                tolerance);
                    }
                }
            }
        } finally {
            restore(previous);
        }
    }

    private static int[] letters(DecisionSession decisions, int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) {
            ids[i] = decisions.model().tokenizer().encodeAsList(String.valueOf((char) ('A' + i))).get(0);
        }
        return ids;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
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
        String override = System.getProperty("jitllm.decision.model");
        if (override != null && !override.isBlank()) {
            return Path.of(override);
        }
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent: fixture " + Fixture.LLAMA_3_2_1B_Q8_0.fileName, false);
        }
        return modelPath;
    }
}
