package org.beehive.jllm.api;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.NativePrefillSupport;
import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * {@code --with-native-libraries} through the library facade: off by default, honoured where a
 * native path exists (Qwen3 F16 batched prefill), and refused at load or session creation
 * everywhere else — never quietly run on the JIT kernels.
 *
 * <p>The refusals happen before any plan is built, so they upload nothing; the one generating model
 * is Qwen3 0.6B F16. Numerics of the native path are {@code NativePrefillNumericalAccelTest}'s.
 */
public class NativeLibrariesAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int BATCH = 128;

    @Test
    public void nativeLibrariesAreOffByDefault() {
        assertFalse(ModelOptions.builder().build().executionPolicy().nativeLibraries());
    }

    @Test
    public void qwen3F16BatchedPrefillRunsWithThem() throws Exception {
        Path file = locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "the native path needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        withGpu(
                () -> {
                    try (LocalModel loaded = LocalModels.load(file, options(true, true))) {
                        String text = generate((TextGenerationModel) loaded);
                        assertFalse("the native path produced no text", text.isBlank());
                    }
                });
    }

    @Test
    public void everyOtherConfigurationIsRefusedAtLoad() throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_F16);
        Path qwen3 = locate(Fixture.QWEN3_0_6B_F16);
        withGpu(
                () -> {
                    assertRefused(() -> LocalModels.load(llama, options(true, true)), "llama");
                    // Qwen3 F16, but not batched prefill: nothing native would run.
                    assertRefused(() -> LocalModels.load(qwen3, options(true, false)), "STANDARD");
                });
    }

    @Test
    public void aSessionOverrideIsCheckedToo() throws Exception {
        Path llama = locate(Fixture.LLAMA_3_2_1B_F16);
        withGpu(
                () -> {
                    try (LocalModel loaded = LocalModels.load(llama, options(false, true))) {
                        SessionOptions session =
                                SessionOptions.builder()
                                        .executionPolicy(
                                                ExecutionPolicy.Overrides.builder()
                                                        .nativeLibraries(true)
                                                        .build())
                                        .build();
                        assertRefused(
                                () -> ((TextGenerationModel) loaded).newSession(session), "llama");
                        // The refused session took no lease: the model still has its one.
                        try (GenerationSession ok = ((TextGenerationModel) loaded).newSession()) {
                            assertFalse(generate(ok).isBlank());
                        }
                    }
                });
    }

    private static ModelOptions options(boolean nativeLibraries, boolean batched) {
        ExecutionPolicy.Builder policy = ExecutionPolicy.builder().nativeLibraries(nativeLibraries);
        if (batched) {
            policy.phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(BATCH);
        }
        return ModelOptions.builder().contextLength(512).executionPolicy(policy.build()).build();
    }

    private static String generate(TextGenerationModel model) throws Exception {
        try (GenerationSession session = model.newSession()) {
            return generate(session);
        }
    }

    private static String generate(GenerationSession session) throws Exception {
        return session.generate(
                        GenerationRequest.builder()
                                .prompt("The capital of France is")
                                .maxNewTokens(16)
                                .temperature(0.0f)
                                .seed(42L)
                                .build())
                .text();
    }

    private static void assertRefused(ThrowingRunnable action, String combination) {
        UnsupportedOperationException e =
                assertThrows(UnsupportedOperationException.class, action::run);
        String message = e.getMessage();
        assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
        assertTrue(message, message.contains("--with-native-libraries"));
        assertTrue(message, message.contains(combination));
    }

    private static Path locate(Fixture fixture) {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        return file;
    }

    private static void withGpu(ThrowingRunnable body) throws Exception {
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            body.run();
        } finally {
            if (previous == null) System.clearProperty(GPU_PROPERTY);
            else System.setProperty(GPU_PROPERTY, previous);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
