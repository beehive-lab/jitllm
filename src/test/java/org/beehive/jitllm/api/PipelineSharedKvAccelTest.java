package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * A model split across devices, its sessions leasing slots of a shared key/value pool: each stage
 * owns its cache and a one-sequence block table, so a chunk's prefill must address the stage's
 * slot, not the session's. Two sessions open together — the second at a non-zero pool slot — and a
 * reopened one must generate what a session of a private cache generates, through a prompt that
 * spans several prefill chunks and key/value blocks and the decode after it.
 *
 * <p>Runs a one-stage split (the ordinary prefill path) and a two-stage split on the same device
 * (the overlapped-chunk path).
 */
public class PipelineSharedKvAccelTest {

    private static final ExecutionPolicy BATCHED =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(128)
                    .build();

    /** About 300 tokens: three chunks of 128, nineteen blocks of 16. */
    private static final String PROMPT =
            "Read the list and answer the question at the end. ".repeat(2)
                    + "Item: the river is long and the mountain is high. ".repeat(24)
                    + "Question: what is high? Answer in one word.";

    @Test
    public void sessionsOfASharedPoolMatchAPrivateCacheOnOneStage() throws Exception {
        sessionsMatch("0:0", false);
    }

    @Test
    public void sessionsOfASharedPoolMatchAPrivateCacheOnTwoStages() throws Exception {
        // A later stage of Llama Q4_0 batched prefill runs its projections through cuBLAS.
        sessionsMatch("0:0,0:0", true);
    }

    private static void sessionsMatch(String devices, boolean nativeLibraries) throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_Q4_0);
        withSplit(
                devices,
                () -> {
                    String expected;
                    try (LocalModel loaded =
                            LocalModels.load(llama, options(false, 1, nativeLibraries))) {
                        try (GenerationSession only = ((TextGenerationModel) loaded).newSession()) {
                            expected = generate(only);
                        }
                    }
                    assertFalse("the private-cache control produced no text", expected.isBlank());

                    try (LocalModel loaded =
                            LocalModels.load(llama, options(true, 2, nativeLibraries))) {
                        TextGenerationModel model = (TextGenerationModel) loaded;
                        try (GenerationSession first = model.newSession()) {
                            try (GenerationSession second = model.newSession()) {
                                assertEquals("first session", expected, generate(first));
                                assertEquals("second session", expected, generate(second));
                            }
                            // The second's slot is free again; a new session reuses it.
                            try (GenerationSession reopened = model.newSession()) {
                                assertEquals("reopened session", expected, generate(reopened));
                            }
                        }
                    }
                });
    }

    private static ModelOptions options(boolean sharedPool, int sessions, boolean nativeLibraries) {
        return ModelOptions.builder()
                .contextLength(512)
                .maxConcurrentSessions(sessions)
                .executionPolicy(
                        ExecutionPolicy.Overrides.builder()
                                .nativeLibraries(nativeLibraries)
                                .build()
                                .applyTo(BATCHED))
                .storageOptions(new StorageOptions(DataType.F16, sharedPool))
                .build();
    }

    private static String generate(GenerationSession session) throws Exception {
        return session.generate(
                        GenerationRequest.builder()
                                .prompt(PROMPT)
                                .maxNewTokens(12)
                                .temperature(0.0f)
                                .seed(42L)
                                .build())
                .text();
    }

    private static Path locate(Fixture fixture) {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        return file;
    }

    private static void withSplit(String devices, ThrowingRunnable body) throws Exception {
        withProperty(
                "use.tornadovm",
                "true",
                () ->
                        withProperty(
                                DeviceSplit.DEVICES_PROPERTY,
                                devices,
                                () -> withProperty(DeviceSplit.TRANSPORT_PROPERTY, "host", body)));
    }

    private static void withProperty(String key, String value, ThrowingRunnable body)
            throws Exception {
        String previous = System.getProperty(key);
        System.setProperty(key, value);
        try {
            body.run();
        } finally {
            if (previous == null) System.clearProperty(key);
            else System.setProperty(key, previous);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
