package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.policy.StorageOptions;
import org.junit.Test;

/**
 * FP16 is the default key/value cache through the library, explicitly — a configuration whose
 * kernels do not implement it is refused before anything is allocated, and FP32 stays available.
 *
 * <p>Uses Phi-3 as a refused family while it has no FP16 cache path on the GPU, Llama F16
 * single-token as the supported neighbour, and Llama Q4_0 — single-token only — for a session
 * override into a mode it does not have.
 */
public class Fp16KeyValueDefaultAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    @Test
    public void aSupportedConfigurationGetsFp16ByDefault() throws Exception {
        Path file = fixtureOrSkip(Fixture.LLAMA_3_2_1B_F16);
        onGpu(
                () -> {
                    try (LocalModel model =
                                    LocalModels.load(
                                            file,
                                            ModelOptions.builder().contextLength(256).build());
                            GenerationSession session =
                                    ((TextGenerationModel) model).newSession()) {
                        assertEquals("FP16", session.prepare().kvCache());
                        assertFalse(session.generate(greedy()).text().isBlank());
                    }
                });
    }

    @Test
    public void anUnsupportedConfigurationIsRefusedAtLoadAndFp32IsTheWayOut() throws Exception {
        Path file = fixtureOrSkip(Fixture.PHI3_MINI_4K_Q8_0);
        onGpu(
                () -> {
                    IllegalArgumentException refused =
                            assertThrows(
                                    IllegalArgumentException.class,
                                    () ->
                                            LocalModels.load(
                                                    file,
                                                    ModelOptions.builder()
                                                            .contextLength(256)
                                                            .build()));
                    String message = refused.getMessage();
                    assertTrue(message, message.contains("GPUL-CFG-002"));
                    assertTrue(message, message.contains("phi3"));
                    assertTrue(message, message.contains("--fp32-kv-cache"));
                    assertTrue(message, message.contains("StorageOptions.fp32()"));

                    try (LocalModel model =
                                    LocalModels.load(
                                            file,
                                            ModelOptions.builder()
                                                    .contextLength(256)
                                                    .storageOptions(StorageOptions.fp32())
                                                    .build());
                            GenerationSession session =
                                    ((TextGenerationModel) model).newSession()) {
                        assertEquals("FP32", session.prepare().kvCache());
                        assertFalse(session.generate(greedy()).text().isBlank());
                    }
                });
    }

    /**
     * A session's policy override can select layers the model's own policy did not, so it is
     * checked again — before the session takes a lease — and the model stays usable.
     */
    @Test
    public void aSessionOverrideIntoAnUnsupportedModeIsRefused() throws Exception {
        Path file = fixtureOrSkip(Fixture.LLAMA_3_2_1B_Q4_0);
        onGpu(
                () -> {
                    try (LocalModel model =
                            LocalModels.load(
                                    file, ModelOptions.builder().contextLength(256).build())) {
                        TextGenerationModel text = (TextGenerationModel) model;
                        SessionOptions sequential =
                                SessionOptions.builder()
                                        .executionPolicy(
                                                ExecutionPolicy.Overrides.builder()
                                                        .phaseStrategy(
                                                                ExecutionPolicy.PhaseStrategy
                                                                        .PREFILL_DECODE)
                                                        .prefillBatchSize(1)
                                                        .build())
                                        .build();
                        IllegalArgumentException refused =
                                assertThrows(
                                        IllegalArgumentException.class,
                                        () -> text.newSession(sequential));
                        assertTrue(refused.getMessage(), refused.getMessage().contains("PREFILL"));
                        // Nothing was leased: the one session this model allows is still free.
                        try (GenerationSession session = text.newSession()) {
                            assertFalse(session.generate(greedy()).text().isBlank());
                        }
                    }
                });
    }

    private interface Body {
        void run() throws Exception;
    }

    private static void onGpu(Body body) throws Exception {
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
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

    private static GenerationRequest greedy() {
        return GenerationRequest.builder()
                .prompt("Name one colour.")
                .maxNewTokens(8)
                .temperature(0f)
                .seed(1L)
                .build();
    }

    private static Path fixtureOrSkip(Fixture fixture) {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        return file;
    }
}
