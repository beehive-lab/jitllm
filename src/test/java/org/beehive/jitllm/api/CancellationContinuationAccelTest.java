package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/** Cancellation must retain the same continuation state as an equal token-budget cutoff. */
@RunWith(Parameterized.class)
public class CancellationContinuationAccelTest {
    @Parameterized.Parameters(name = "gpu={0}, batch={1}, fixture={2}")
    public static Object[][] modes() {
        return new Object[][] {
            {false, 0, Fixture.LLAMA_3_2_1B_Q8_0},
            {false, 1, Fixture.LLAMA_3_2_1B_Q8_0},
            {false, 8, Fixture.LLAMA_3_2_1B_Q8_0},
            {true, 0, Fixture.LLAMA_3_2_1B_Q8_0},
            {true, 1, Fixture.LLAMA_3_2_1B_Q8_0},
            {true, 8, Fixture.LLAMA_3_2_1B_Q8_0},
            {true, 0, Fixture.LLAMA_3_2_1B_F16}
        };
    }

    private final boolean gpu;
    private final int batch;
    private final Fixture fixture;

    public CancellationContinuationAccelTest(boolean gpu, int batch, Fixture fixture) {
        this.gpu = gpu;
        this.batch = batch;
        this.fixture = fixture;
    }

    @Test
    public void cancelledTurnContinuesLikeTheSameTokenBudget() throws Exception {
        Path path = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), path != null);
        if (gpu) {
            assumeTrue("GPU cases need TORNADOVM_HOME", System.getenv("TORNADOVM_HOME") != null);
        }
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", Boolean.toString(gpu));
        ExecutionPolicy policy =
                ExecutionPolicy.builder()
                        .phaseStrategy(
                                batch == 0
                                        ? PhaseStrategy.SINGLE_TOKEN
                                        : PhaseStrategy.PREFILL_DECODE)
                        .prefillBatchSize(Math.max(1, batch))
                        .build();
        ModelOptions options =
                ModelOptions.builder()
                        .contextLength(512)
                        .executionPolicy(policy)
                        .storageOptions(StorageOptions.fp32())
                        .build();
        try (LocalModel model = LocalModels.load(path, options);
                GenerationSession session = ((TextGenerationModel) model).newSession()) {
            String prompt = "Count from one to one hundred, in words, separated by commas.";
            List<Integer> expectedTokens = new ArrayList<>();
            GenerationResult limited =
                    session.generate(
                            request(prompt, 6)
                                    .onEvent(event -> expectedTokens.add(event.tokenId()))
                                    .build());
            assertEquals(FinishReason.MAX_TOKENS, limited.finishReason());
            assertTrue("enough tokens to cancel before the budget", expectedTokens.size() > 1);
            int expectedSeed = seed(session);
            int expectedPosition = session.position();
            List<Integer> expectedNext = new ArrayList<>();
            session.generate(
                    request("Continue counting from where you stopped.", 24)
                            .onEvent(event -> expectedNext.add(event.tokenId()))
                            .build());

            session.reset();
            CancellationToken token = new CancellationToken();
            List<Integer> actualTokens = new ArrayList<>();
            GenerationResult cancelled =
                    session.generate(
                            request(prompt, 64)
                                    .cancellation(token)
                                    .onEvent(
                                            event -> {
                                                actualTokens.add(event.tokenId());
                                                // The stream holds one token back; cancel when the
                                                // loop has the same count.
                                                if (actualTokens.size()
                                                        == expectedTokens.size() - 1) {
                                                    token.cancel();
                                                }
                                            })
                                    .build());
            assertEquals(FinishReason.CANCELLED, cancelled.finishReason());
            assertEquals(expectedTokens, actualTokens);
            assertEquals(expectedPosition, session.position());
            assertEquals(
                    "the retained final token must seed the next turn",
                    expectedSeed,
                    seed(session));
            assertEquals((int) actualTokens.getLast(), seed(session));
            List<Integer> actualNext = new ArrayList<>();
            session.generate(
                    request("Continue counting from where you stopped.", 24)
                            .onEvent(event -> actualNext.add(event.tokenId()))
                            .build());
            assertTrue("the next turn generated tokens", !actualNext.isEmpty());
            assertEquals(
                    "same retained history must produce the same next turn",
                    expectedNext,
                    actualNext);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static GenerationRequest.Builder request(String prompt, int maxTokens) {
        return GenerationRequest.builder()
                .prompt(prompt)
                .maxNewTokens(maxTokens)
                .temperature(0f)
                .seed(42);
    }

    // Check the continuation invariant directly: fluent output alone can hide a stale seed.
    private static int seed(GenerationSession session) throws Exception {
        Field field = DelegatingSession.class.getDeclaredField("runtime");
        field.setAccessible(true);
        SessionRuntime runtime = (SessionRuntime) field.get(session);
        return runtime instanceof LoweredSessionRuntime lowered
                ? lowered.cursor().seed()
                : runtime.executionState().latestToken;
    }
}
