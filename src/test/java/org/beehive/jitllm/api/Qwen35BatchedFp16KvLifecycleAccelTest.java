package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * The batched plan with a <b>half-precision</b> key/value cache.
 *
 * <p>Its own class because it is a different configuration, and it exists because that
 * configuration was silently broken. The decode activation graph is the only link between the
 * batch-prefill layers and the decode layers, and it bound the FP32 key/value carriers
 * unconditionally while both sides of it bound whichever store the session actually holds. A
 * half-precision session therefore prefilled one pair of arrays and decoded from another, and
 * because the FP32 arrays are allocated either way nothing failed — the model simply answered as
 * though the prompt had never been read. Asked to explain matrix multiplication it explained what a
 * matrix is.
 *
 * <p>So this asserts the thing that broke: that the answer after a batched half-precision prefill
 * is the answer to the question asked. Same prompt, same greedy seed and same assertions as the
 * FP32 case beside it.
 */
// @formatter:on
public class Qwen35BatchedFp16KvLifecycleAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int CONTEXT = 512;
    private static final int MAX_NEW = 140;

    // Long enough that losing the prompt shows. Asked with a short prompt the model
    // answers the same way either way, which is how this defect survived until now.
    private static final String PROMPT =
            "Explain what a matrix multiplication is in one paragraph.";

    /**
     * What this session's own plan must be built with. Nothing here; the tensor-core subclass
     * overrides it.
     */
    protected void verifyDispatch(DelegatingSession session, int batch, int dim) {}

    @Test
    public void aHalfPrecisionKeyValueBatchedSessionCarriesItsPromptIntoDecode() throws Exception {
        Path modelPath = fixtureOrSkip();
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", "32");
        System.setProperty("jitllm.kvcache.fp16", "true");
        try (LocalModel model = LocalModels.load(modelPath, options())) {
            TextGenerationModel generation = (TextGenerationModel) model;
            // Thinking off: the answer has to land inside the token budget, and this family
            // spends a paragraph reasoning before it otherwise.
            GenerationSession session =
                    // Thinking left on, deliberately: the failure this covers shows in a long
                    // decode, and a short one answers correctly either way.
                    generation.newSession(SessionOptions.builder().build());
            String first;
            String repeated;
            String afterReset;
            try {
                GenerationRequest request = request();
                first = session.generate(request).text();
                assertFalse("nothing was generated, so the case proves nothing", first.isEmpty());

                // Proved, not assumed. A silent host fallback produces fluent text at a plausible
                // rate, and every assertion below would pass on it.
                var snapshot = org.beehive.jitllm.auxiliary.RunMetrics.snapshot();
                assertEquals("the plan that ran", "legacy", snapshot.executionPath());
                assertEquals(
                        "the tuple that ran",
                        "qwen35/Q4_0/BATCH_PREFILL_DECODE",
                        snapshot.executionCombination());
                // The defect's signature: with the prefilled history lost, the model answers a
                // question it was not asked -- it explains what a matrix is.
                assertTrue(
                        "the answer does not mention multiplication, so the prompt did not reach"
                                + " decode: '"
                                + first
                                + "'",
                        first.toLowerCase().contains("multiplication")
                                || first.toLowerCase().contains("multiply"));

                // The plan this session generated with, before close frees it.
                verifyDispatch((DelegatingSession) session, 32, model.configuration().dimension());

                // Same session, same request, no reset: the conversation surface resets
                // transparently when the prefix does not extend, so this is the ordinary
                // second turn.
                repeated = session.generate(request).text();

                session.reset();
                afterReset = session.generate(request).text();
            } finally {
                session.close();
            }

            assertEquals(
                    "a second identical request answers the same way, so nothing carried over"
                            + " from the first — on the device, that means the recurrent state was"
                            + " cleared there and not only on the host",
                    first,
                    repeated);
            assertEquals("an explicit reset returns the session to a new one", first, afterReset);

            assertThrows(
                    "a closed session must refuse work rather than use freed device buffers",
                    IllegalStateException.class,
                    () -> session.generate(request()));
        } finally {
            System.clearProperty("jitllm.withPrefillDecode");
            System.clearProperty("jitllm.prefillBatchSize");
            restore(previous);
        }
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder()
                .prompt(PROMPT)
                .maxNewTokens(MAX_NEW)
                .temperature(0.0f)
                .seed(1234L)
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
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent: fixture " + Fixture.QWEN3_8_27B_Q4_0.fileName, false);
        }
        return modelPath;
    }
}
