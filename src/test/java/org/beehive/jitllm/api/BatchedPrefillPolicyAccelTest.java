package org.beehive.jllm.api;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * A batched-prefill policy set through {@link ModelOptions} must reach <b>allocation</b>, not only
 * plan construction.
 *
 * <p>The prefill workspace is sized from the batch width when the state is built. The facade
 * resolves an {@code ExecutionPolicy} onto the finished state, which is after that, so the width
 * had come from the {@code jllm.prefillBatchSize} system property alone. A caller that set the
 * width through the facade got a plan built for a batch whose arrays were never allocated, and the
 * failure arrived from inside TornadoVM as {@code null object passed into streamIn() in schedule
 * prefillActivation} — naming neither the policy nor the property.
 *
 * <p>The property is deliberately left unset here: setting it would allocate the arrays for the
 * wrong reason and the test would pass with the defect present.
 */
public class BatchedPrefillPolicyAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String BATCH_PROPERTY = "jllm.prefillBatchSize";
    private static final int BATCH = 32;

    @Test
    public void aBatchedPrefillPolicyFromTheFacadeGenerates() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousBatch = System.getProperty(BATCH_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.clearProperty(BATCH_PROPERTY);
        try {
            ModelOptions options =
                    ModelOptions.builder()
                            .contextLength(512)
                            .executionPolicy(
                                    ExecutionPolicy.builder()
                                            .phaseStrategy(
                                                    ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                            .prefillBatchSize(BATCH)
                                            .build())
                            .build();

            try (LocalModel loaded = LocalModels.load(model, options)) {
                TextGenerationModel generator = (TextGenerationModel) loaded;
                try (GenerationSession session = generator.newSession()) {
                    // Long enough that prefill runs more than one batch.
                    String prompt =
                            "Answer with one word. The capital of France is a city whose name"
                                    + " every schoolchild in Europe learns. What is that city"
                                    + " called?";
                    GenerationResult result =
                            session.generate(
                                    GenerationRequest.builder()
                                            .prompt(prompt)
                                            .maxNewTokens(16)
                                            .seed(42L)
                                            .build());
                    assertTrue(
                            "batched prefill through the facade produced no text",
                            !result.text().isBlank());
                    assertTrue(
                            "batched prefill produced the wrong answer, got: " + result.text(),
                            result.text().toLowerCase().contains("paris"));
                }
            }
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(BATCH_PROPERTY, previousBatch);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
