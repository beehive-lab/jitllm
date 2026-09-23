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
 * Class B: runs under {@code -Paccel-tests} with the pinned fixture, skipping when it is absent.
 *
 * <p>Everything here is greedy (temperature 0) with a fixed seed, which is what makes
 * token-identity a property rather than a wish [A2]: prefix reuse and {@code reset()} change how
 * long a request takes, and the amendment is explicit that they preserve <i>output</i> only under
 * greedy or a fixed seed with a deterministic execution tuple.
 */
public class ConversationAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 512;
    private static final int MAX_NEW = 24;

    private static final ChatMessage FIRST_USER =
            ChatMessage.of(ChatRole.USER, "Name one colour. Answer with just the word.");
    private static final ChatMessage SECOND_USER =
            ChatMessage.of(ChatRole.USER, "Name a different one. Just the word.");

    /**
     * Acceptance: a conversation continued on one session equals the same conversation sent whole
     * to a fresh session.
     *
     * <p>This is the whole claim of "the session is a cache, not a memory". If reuse ever changed
     * the answer, these two would differ.
     */
    @Test
    public void aReusedPrefixProducesWhatAFreshSessionProduces() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;

            String reused;
            List<ChatMessage> whole = new ArrayList<>();
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                whole.add(FIRST_USER);
                GenerationResult first = session.generate(request(whole));
                // The assistant turn joins the conversation, exactly as a caller would replay it.
                whole.add(ChatMessage.of(ChatRole.ASSISTANT, first.text()));
                whole.add(SECOND_USER);
                reused = session.generate(request(whole)).text();
            }

            String fresh;
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                fresh = session.generate(request(whole)).text();
            }

            assertTrue("the comparison is worthless if nothing was generated", !reused.isEmpty());
            assertEquals("prefix reuse is a timing property, never a semantic one", fresh, reused);
        } finally {
            restore(previous);
        }
    }

    /**
     * Acceptance: an edited earlier turn diverges, resets transparently, and answers the
     * conversation that was actually sent.
     */
    @Test
    public void divergenceResetsAndAnswersTheConversationThatWasSent() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;

            List<ChatMessage> edited =
                    List.of(
                            ChatMessage.of(
                                    ChatRole.USER, "Name one animal. Answer with just the word."));

            String afterDivergence;
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                session.generate(request(List.of(FIRST_USER))); // retains a colour prefix
                afterDivergence = session.generate(request(edited)).text();
            }

            String fresh;
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                fresh = session.generate(request(edited)).text();
            }

            assertEquals(
                    "a diverged conversation is re-encoded, so it must equal a fresh session",
                    fresh,
                    afterDivergence);
        } finally {
            restore(previous);
        }
    }

    /**
     * Acceptance A2: an explicit {@code reset()} changes nothing about the answer, at greedy and a
     * fixed seed.
     *
     * <p>The ADR's first draft claimed this unconditionally. It is only true under these
     * conditions, because discarding a retained prefix cannot restore draws a random source has
     * already made — which is exactly why the test states them.
     */
    @Test
    public void explicitResetDoesNotChangeTheAnswerAtGreedyAndAFixedSeed() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;

            String withoutReset;
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                withoutReset = session.generate(request(List.of(FIRST_USER))).text();
            }

            String withReset;
            try (GenerationSession session =
                    generation.newSession(SessionOptions.builder().build())) {
                session.generate(request(List.of(FIRST_USER)));
                session.reset();
                withReset = session.generate(request(List.of(FIRST_USER))).text();
            }

            assertEquals(
                    "reset never changes the encoded input or the sampling initialization",
                    withoutReset,
                    withReset);
        } finally {
            restore(previous);
        }
    }

    /**
     * A defect this milestone found rather than caused: {@code reset()} followed by {@code
     * generate()} threw on the CPU path.
     *
     * <p>{@code LegacySessionRuntime.reset()} set {@code latestToken = -1} — {@code State}'s "not
     * yet set" constructor value — but every family's {@code createNewState} seeds a real start
     * token over it, and {@code PromptIngestion} reads that seed as the first token to feed. After
     * a reset it read {@code -1}, and {@code -1} reached the embedding table as an index: an {@code
     * AssertionError} from {@code Q8_0FloatTensor.getFloat} with assertions on, and an
     * out-of-bounds read without them.
     *
     * <p>It is written with the <b>prompt</b> form deliberately. The conversation surface reached
     * this through a transparent reset, which is how it was found, but the defect is older than
     * that surface and has always been reachable through an explicit {@code reset()}.
     */
    @Test
    public void resetThenGenerateWorksOnThePromptForm() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;
            try (GenerationSession session = generation.newSession()) {
                GenerationRequest request =
                        GenerationRequest.builder()
                                .prompt("Name one colour. Answer with just the word.")
                                .maxNewTokens(MAX_NEW)
                                .temperature(0.0f)
                                .seed(1234L)
                                .build();
                String before = session.generate(request).text();
                session.reset();
                String after = session.generate(request).text();
                assertTrue("nothing was generated, so the case proves nothing", !before.isEmpty());
                assertEquals(
                        "reset returns the session to a new one, so the same request answers"
                                + " the same way",
                        before,
                        after);
            }
        } finally {
            restore(previous);
        }
    }

    /**
     * The count equality is the one worth having end to end: the stub-based unit tests can prove
     * the stream drops a terminal stop token, but only a real run proves the number it drops
     * matches what the result reports as generated.
     */
    @Test
    public void everyEmittedTokenIsOneEventAndTheCountsAgree() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;
            List<GenerationEvent> events = new ArrayList<>();
            List<String> streamedText = new ArrayList<>();

            try (GenerationSession session = generation.newSession()) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .messages(
                                                List.of(
                                                        ChatMessage.of(
                                                                ChatRole.USER,
                                                                "Count from one to five, in words.")))
                                        .maxNewTokens(48)
                                        .temperature(0.0f)
                                        .seed(99L)
                                        .onEvent(events::add)
                                        .onToken(streamedText::add)
                                        .build());

                assertTrue(
                        "nothing was generated, so the invariants prove nothing",
                        result.generatedTokens() > 0);
                assertEquals(
                        "one event per emitted completion token",
                        result.generatedTokens(),
                        events.size());
                assertEquals(
                        "concatenated event text equals the result text",
                        String.join("", events.stream().map(GenerationEvent::text).toList()),
                        result.text());
                assertEquals(
                        "onToken sees exactly the non-empty texts, in the same order",
                        events.stream()
                                .map(GenerationEvent::text)
                                .filter(t -> !t.isEmpty())
                                .toList(),
                        streamedText);
                assertTrue(
                        "no terminal stop token reaches the stream",
                        events.stream().map(GenerationEvent::tokenId).noneMatch(id -> id < 0));
            }
        } finally {
            restore(previous);
        }
    }

    private static GenerationRequest request(List<ChatMessage> messages) {
        return GenerationRequest.builder()
                .messages(messages)
                .maxNewTokens(MAX_NEW)
                .temperature(0.0f) // greedy: A2's first condition
                .seed(1234L) // and its second, for good measure
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
