package org.beehive.jitllm.api;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * {@link EmbeddingSession} on the GPU, with an embedding model given by {@code
 * -Djitllm.embedding.model=<gguf>} (e.g. Qwen3-Embedding-0.6B Q8_0); skipped without one.
 *
 * <p>Checks what a retrieval system relies on: unit vectors, the same text giving the same vector,
 * a paraphrase scoring above an unrelated text, and the vector not depending on whether the prompt
 * was ingested one token at a time or in a batched prefill.
 */
public class EmbeddingSessionAccelTest {

    private static final String QUERY =
            "Instruct: Given a question, retrieve passages that answer it\nQuery:How do virtual threads work?";
    private static final String RELATED =
            "Virtual threads are lightweight threads implemented by the JDK rather than the OS. Many virtual"
                    + " threads are multiplexed onto a small number of platform carrier threads.";
    private static final String UNRELATED =
            "The garbage collector reclaims heap memory occupied by objects that are no longer reachable.";

    @Test
    public void vectorsAreUnitDeterministicAndRankAParaphraseFirst() throws Exception {
        Path gguf = modelOrSkip();
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try (LocalModel model =
                        LocalModels.load(
                                gguf,
                                ModelOptions.builder()
                                        .contextLength(512)
                                        .maxConcurrentSessions(2)
                                        .build());
                EmbeddingSession embeddings = EmbeddingSession.open((TextGenerationModel) model)) {
            float[] q = embeddings.embed(QUERY);
            float[] related = embeddings.embed(RELATED);
            float[] unrelated = embeddings.embed(UNRELATED);

            assertEquals(model.info().name(), embeddings.dimension(), q.length);
            assertEquals("unit length", 1.0, dot(q, q), 1e-4);
            assertArrayEquals("the same text, the same vector", q, embeddings.embed(QUERY), 1e-6f);

            try (EmbeddingSession batched =
                    EmbeddingSession.open(
                            (TextGenerationModel) model,
                            SessionOptions.builder()
                                    .executionPolicy(
                                            ExecutionPolicy.Overrides.builder()
                                                    .phaseStrategy(
                                                            ExecutionPolicy.PhaseStrategy
                                                                    .PREFILL_DECODE)
                                                    .prefillBatchSize(32)
                                                    .build())
                                    .build())) {
                double agreement = dot(related, batched.embed(RELATED));
                System.out.printf(
                        "[embedding] %s dim %d: query-related %.4f, query-unrelated %.4f,"
                                + " single-token vs batched prefill %.6f%n",
                        model.info().name(),
                        q.length,
                        dot(q, related),
                        dot(q, unrelated),
                        agreement);
                assertTrue(
                        "single-token and batched prefill agree: " + agreement, agreement > 0.999);
            }

            assertTrue(
                    "related "
                            + dot(q, related)
                            + " must score above unrelated "
                            + dot(q, unrelated),
                    dot(q, related) > dot(q, unrelated) + 0.05);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    private static Path modelOrSkip() {
        String path = System.getProperty("jitllm.embedding.model");
        boolean present = path != null && !path.isBlank() && Files.isRegularFile(Path.of(path));
        if (!present) {
            System.out.println(
                    "[SKIP] environment absent: set -Djitllm.embedding.model=<embedding gguf>");
        }
        assumeTrue("environment absent: -Djitllm.embedding.model", present);
        return Path.of(path);
    }
}
