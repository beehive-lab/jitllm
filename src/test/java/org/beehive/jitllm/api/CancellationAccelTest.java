package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;

/**
 * Cancelling a generation through {@link CancellationToken}, on the CPU loop and on the GPU loop.
 * Needs the Llama 3.2 1B Q8_0 fixture; the GPU cases also need a TornadoVM device.
 */
public class CancellationAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 512;
    private static final String PROMPT =
            "Count from one to one hundred, in words, separated by commas.";

    @Test
    public void cancellingAtTheFifthEventStopsAfterTheSixthTokenOnTheCpu() throws Exception {
        stopsRightAfterTheCancellationAndTheSessionGoesOn(false);
    }

    @Test
    public void cancellingAtTheFifthEventStopsAfterTheSixthTokenOnTheGpu() throws Exception {
        stopsRightAfterTheCancellationAndTheSessionGoesOn(true);
    }

    @Test
    public void anAlreadyCancelledRequestLeavesTheSessionUntouchedOnTheCpu() throws Exception {
        alreadyCancelledReturnsAtOnce(false);
    }

    @Test
    public void anAlreadyCancelledRequestLeavesTheSessionUntouchedOnTheGpu() throws Exception {
        alreadyCancelledReturnsAtOnce(true);
    }

    @Test
    public void cancellingFromAnotherThreadStopsTheGpuLoop() throws Exception {
        withSession(
                true,
                session -> {
                    CancellationToken token = new CancellationToken();
                    CountDownLatch firstToken = new CountDownLatch(1);
                    Thread canceller =
                            new Thread(
                                    () -> {
                                        try {
                                            if (firstToken.await(60, TimeUnit.SECONDS)) {
                                                token.cancel();
                                            }
                                        } catch (InterruptedException e) {
                                            Thread.currentThread().interrupt();
                                        }
                                    });
                    canceller.start();
                    GenerationResult result =
                            session.generate(
                                    request(256)
                                            .cancellation(token)
                                            .onToken(text -> firstToken.countDown())
                                            .build());
                    canceller.join();
                    assertEquals(FinishReason.CANCELLED, result.finishReason());
                    assertTrue(
                            "cancelled well before the budget: " + result.generatedTokens(),
                            result.generatedTokens() < 256);
                });
    }

    @Test
    public void cancellationInTheFinalBufferedEventKeepsTheBudgetReasonOnTheCpu() throws Exception {
        finalBufferedEventKeepsTheBudgetReason(false);
    }

    @Test
    public void cancellationInTheFinalBufferedEventKeepsTheBudgetReasonOnTheGpu() throws Exception {
        finalBufferedEventKeepsTheBudgetReason(true);
    }

    private static void finalBufferedEventKeepsTheBudgetReason(boolean gpu) throws Exception {
        withSession(
                gpu,
                session -> {
                    CancellationToken token = new CancellationToken();
                    AtomicInteger delivered = new AtomicInteger();
                    GenerationResult result =
                            session.generate(
                                    request(1)
                                            .cancellation(token)
                                            .onEvent(
                                                    event -> {
                                                        delivered.incrementAndGet();
                                                        token.cancel();
                                                    })
                                            .build());
                    assertEquals(1, delivered.get());
                    assertEquals(1, result.generatedTokens());
                    assertTrue(token.isCancelled());
                    // The only event is flushed after the loop exhausts its one-token budget.
                    assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
                });
    }

    private static void stopsRightAfterTheCancellationAndTheSessionGoesOn(boolean gpu)
            throws Exception {
        withSession(
                gpu,
                session -> {
                    CancellationToken token = new CancellationToken();
                    AtomicInteger delivered = new AtomicInteger();
                    GenerationResult cancelled =
                            session.generate(
                                    request(64)
                                            .cancellation(token)
                                            .onEvent(
                                                    event -> {
                                                        if (delivered.incrementAndGet() == 5) {
                                                            token.cancel();
                                                        }
                                                    })
                                            .build());
                    assertEquals(FinishReason.CANCELLED, cancelled.finishReason());
                    // Events are one token behind the loop (the stream holds a token back in case
                    // it
                    // is a stop token): the fifth event is delivered while the sixth token is. The
                    // loop stops right after that token, which the result still reports.
                    assertEquals(
                            "the loop stops after the token being delivered when cancel() ran",
                            6,
                            cancelled.generatedTokens());
                    assertFalse(cancelled.text().isEmpty());

                    // The session is still usable: the next turn runs to a natural end.
                    GenerationResult next =
                            session.generate(
                                    GenerationRequest.builder()
                                            .prompt("Now say just: done.")
                                            .temperature(0f)
                                            .seed(42)
                                            .maxNewTokens(16)
                                            .build());
                    assertNotEquals(FinishReason.CANCELLED, next.finishReason());
                    assertTrue(next.generatedTokens() > 0);
                });
    }

    private static void alreadyCancelledReturnsAtOnce(boolean gpu) throws Exception {
        withSession(
                gpu,
                session -> {
                    int before = session.position();
                    CancellationToken token = new CancellationToken();
                    token.cancel();
                    GenerationResult result =
                            session.generate(request(64).cancellation(token).build());
                    assertEquals(FinishReason.CANCELLED, result.finishReason());
                    assertEquals(0, result.generatedTokens());
                    assertEquals("", result.text());
                    assertEquals("nothing was ingested", before, session.position());
                });
    }

    private static GenerationRequest.Builder request(int maxNewTokens) {
        return GenerationRequest.builder()
                .prompt(PROMPT)
                .temperature(0f)
                .seed(42)
                .maxNewTokens(maxNewTokens);
    }

    private interface SessionBody {
        void run(GenerationSession session) throws Exception;
    }

    private static void withSession(boolean gpu, SessionBody body) throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent: fixture " + Fixture.LLAMA_3_2_1B_Q8_0.fileName, false);
        }
        if (gpu) {
            assumeTrue(
                    "GPU cases need a TornadoVM SDK (TORNADOVM_HOME)",
                    System.getenv("TORNADOVM_HOME") != null);
        }
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, Boolean.toString(gpu));
        // FP32 key/value storage: accepted on every backend (FP16 is not verified off CUDA).
        ModelOptions options =
                ModelOptions.builder()
                        .contextLength(CONTEXT)
                        .storageOptions(StorageOptions.fp32())
                        .build();
        try (LocalModel model = LocalModels.load(modelPath, options);
                GenerationSession session = ((TextGenerationModel) model).newSession()) {
            body.run(session);
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }
}
