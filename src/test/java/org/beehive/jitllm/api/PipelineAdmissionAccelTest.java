package org.beehive.jitllm.api;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;

/**
 * A model split across devices ({@code --devices}) has no sequential prefill/decode plan: that
 * policy is refused by name at load, with either key/value cache, rather than built as a
 * single-token pipeline the generation loop then casts to the wrong plan. A split the family's
 * stages cannot build is refused the same way, before any stage or transport exists. Single-token
 * and batched pipelines still generate, as does sequential prefill without a split.
 *
 * <p>One device suffices: a split over the host transport takes the same path, with one stage or
 * with two on the same device.
 */
public class PipelineAdmissionAccelTest {

    private static final ExecutionPolicy SEQUENTIAL =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(1)
                    .build();

    private static final ExecutionPolicy BATCHED =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(128)
                    .build();

    @Test
    public void sequentialPrefillIsRefusedOnASplitWithEitherCache() throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_Q8_0);
        for (StorageOptions storage :
                new StorageOptions[] {StorageOptions.fp16(), StorageOptions.fp32()}) {
            withSplit(
                    () -> {
                        UnsupportedOperationException refused =
                                assertThrows(
                                        UnsupportedOperationException.class,
                                        () ->
                                                LocalModels.load(
                                                        llama, options(SEQUENTIAL, storage)));
                        String message = refused.getMessage();
                        assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
                        assertTrue(message, message.contains("--devices"));
                        assertTrue(message, message.contains("sequential prefill/decode"));
                    });
        }
    }

    @Test
    public void aSessionOverrideToSequentialPrefillIsRefusedOnASplit() throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_Q4_0);
        withSplit(
                () -> {
                    try (LocalModel loaded =
                            LocalModels.load(llama, options(BATCHED, StorageOptions.fp16()))) {
                        SessionOptions sequential =
                                SessionOptions.builder()
                                        .executionPolicy(
                                                ExecutionPolicy.Overrides.builder()
                                                        .prefillBatchSize(1)
                                                        .build())
                                        .build();
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> ((TextGenerationModel) loaded).newSession(sequential));
                    }
                });
    }

    @Test
    public void aBatchedSplitTheStagesCannotBuildIsRefusedAtLoad() throws Exception {
        // Llama Q8_0 has no batched stage layers, with one stage as with two.
        assertRefusedAtLoad(
                locate(Fixture.LLAMA_3_2_1B_Q8_0), BATCHED, "0:0", "--batch-prefill-size");
        // Llama Q4_0 stages that hold part of the model run their projections through cuBLAS.
        assertRefusedAtLoad(
                locate(Fixture.LLAMA_3_2_1B_Q4_0), BATCHED, "0:0,0:0", "--with-native-libraries");
    }

    private static void assertRefusedAtLoad(
            Path file, ExecutionPolicy policy, String devices, String remedy) throws Exception {
        withSplit(
                devices,
                () -> {
                    UnsupportedOperationException refused =
                            assertThrows(
                                    UnsupportedOperationException.class,
                                    () ->
                                            LocalModels.load(
                                                    file, options(policy, StorageOptions.fp16())));
                    String message = refused.getMessage();
                    assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
                    assertTrue(message, message.contains(remedy));
                });
    }

    @Test
    public void singleTokenAndBatchedPipelinesStillGenerate() throws Exception {
        // Llama Q4_0 splits its batched prefill too; Q8_0 splits single-token only.
        generatesOnASplit(locate(Fixture.LLAMA_3_2_1B_Q8_0), ExecutionPolicy.builder().build());
        generatesOnASplit(locate(Fixture.LLAMA_3_2_1B_Q4_0), BATCHED);
    }

    private static void generatesOnASplit(Path file, ExecutionPolicy policy) throws Exception {
        withSplit(
                () -> {
                    try (LocalModel loaded =
                            LocalModels.load(file, options(policy, StorageOptions.fp16()))) {
                        assertFalse(generate((TextGenerationModel) loaded).isBlank());
                    }
                });
    }

    @Test
    public void sequentialPrefillWithoutASplitStillGenerates() throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_Q8_0);
        withGpu(
                () -> {
                    try (LocalModel loaded =
                            LocalModels.load(llama, options(SEQUENTIAL, StorageOptions.fp16()))) {
                        assertFalse(generate((TextGenerationModel) loaded).isBlank());
                    }
                });
    }

    private static ModelOptions options(ExecutionPolicy policy, StorageOptions storage) {
        return ModelOptions.builder()
                .contextLength(256)
                .executionPolicy(policy)
                .storageOptions(storage)
                .build();
    }

    private static String generate(TextGenerationModel model) throws Exception {
        try (GenerationSession session = model.newSession()) {
            return session.generate(
                            GenerationRequest.builder()
                                    .prompt("The capital of France is")
                                    .maxNewTokens(8)
                                    .temperature(0.0f)
                                    .seed(42L)
                                    .build())
                    .text();
        }
    }

    private static Path locate(Fixture fixture) {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        return file;
    }

    /** A one-stage split on device 0:0 over the host transport, on the GPU. */
    private static void withSplit(ThrowingRunnable body) throws Exception {
        withSplit("0:0", body);
    }

    private static void withSplit(String devices, ThrowingRunnable body) throws Exception {
        withGpu(
                () ->
                        withProperty(
                                DeviceSplit.DEVICES_PROPERTY,
                                devices,
                                () -> withProperty(DeviceSplit.TRANSPORT_PROPERTY, "host", body)));
    }

    private static void withGpu(ThrowingRunnable body) throws Exception {
        withProperty("use.tornadovm", "true", body);
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
