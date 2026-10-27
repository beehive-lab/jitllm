package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * A plan tiles its projections for its own execution policy, given through the API, not for the
 * process's properties: Gemma 4 Q4_0 with native projections chosen in {@link ModelOptions} keeps
 * the blocks cuBLAS's decode reads, with no packing property set. A session that asks for a policy
 * tiles cannot serve builds its plan without them, over the same loaded weights.
 */
public class PackedWeightsPolicyAccelTest {

    private static final String WIDTH = "jitllm.prefillBatchSize";
    private static final String PREFILL_DECODE = "jitllm.withPrefillDecode";
    private static final String NATIVE = "jitllm.nativeLibraries";

    @Test
    public void nativePrefillChosenThroughTheApiGenerates() throws Exception {
        Path gemma = locate(Fixture.GEMMA_4_E2B_Q4_0);
        assumeNativeLibraries();
        withGpu(
                () -> {
                    try (LocalModel loaded = LocalModels.load(gemma, options(128, true))) {
                        assertFalse(generate((TextGenerationModel) loaded).isBlank());
                    }
                });
    }

    @Test
    public void modelsLoadedWithDifferentPoliciesKeepTheirOwnRepresentation() throws Exception {
        Path gemma = locate(Fixture.GEMMA_4_E2B_Q4_0);
        assumeNativeLibraries();
        withGpu(
                () -> {
                    try (LocalModel jit = LocalModels.load(gemma, options(128, false));
                            LocalModel nativeLibraries =
                                    LocalModels.load(gemma, options(128, true))) {
                        assertFalse(generate((TextGenerationModel) jit).isBlank());
                        assertFalse(generate((TextGenerationModel) nativeLibraries).isBlank());
                    }
                });
    }

    @Test
    public void conflictingPropertiesDoNotDecideTheRepresentation() throws Exception {
        Path gemma = locate(Fixture.GEMMA_4_E2B_Q4_0);
        withGpu(
                () ->
                        // The properties ask for a width the packed GEMM tiles and no native
                        // libraries; the API asks for a width it does not.
                        withProperties(
                                () -> {
                                    try (LocalModel loaded =
                                            LocalModels.load(gemma, options(64, false))) {
                                        assertFalse(
                                                generate((TextGenerationModel) loaded).isBlank());
                                    }
                                },
                                PREFILL_DECODE,
                                "true",
                                WIDTH,
                                "128",
                                NATIVE,
                                "false"));
    }

    @Test
    public void aSessionTilesForItsOwnPolicy() throws Exception {
        Path gemma = locate(Fixture.GEMMA_4_E2B_Q4_0);
        assumeTrue("needs int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        withGpu(
                () -> {
                    String untiled;
                    try (LocalModel narrow = LocalModels.load(gemma, options(64, false))) {
                        untiled = generate((TextGenerationModel) narrow);
                    }
                    try (LocalModel loaded = LocalModels.load(gemma, options(128, false))) {
                        TextGenerationModel model = (TextGenerationModel) loaded;
                        String tiled = generate(model);
                        // A width the int8 GEMM does not tile: no tiles, over the same weights a
                        // tiled plan has repacked on its own device copy.
                        assertEquals(
                                untiled,
                                generate(
                                        model,
                                        ExecutionPolicy.Overrides.builder().prefillBatchSize(64)));
                        assertEquals(tiled, generate(model));
                    }
                });
    }

    private static ModelOptions options(int width, boolean nativeLibraries) {
        return ModelOptions.builder()
                .contextLength(256)
                .executionPolicy(
                        ExecutionPolicy.builder()
                                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                .prefillBatchSize(width)
                                .nativeLibraries(nativeLibraries)
                                .build())
                .build();
    }

    private static String generate(TextGenerationModel model) throws Exception {
        try (GenerationSession session = model.newSession()) {
            return generate(session);
        }
    }

    private static String generate(
            TextGenerationModel model, ExecutionPolicy.Overrides.Builder override)
            throws Exception {
        SessionOptions options = SessionOptions.builder().executionPolicy(override.build()).build();
        try (GenerationSession session = model.newSession(options)) {
            return generate(session);
        }
    }

    private static String generate(GenerationSession session) throws Exception {
        return session.generate(
                        GenerationRequest.builder()
                                .prompt("What colour is the sky on a clear day?")
                                .maxNewTokens(8)
                                .temperature(0.0f)
                                .seed(42L)
                                .build())
                .text();
    }

    private static void assumeNativeLibraries() {
        assumeTrue("needs tensor cores", TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
    }

    private static Path locate(Fixture fixture) {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        return file;
    }

    private static void withGpu(ThrowingRunnable body) throws Exception {
        withProperties(body, "use.tornadovm", "true");
    }

    private static void withProperties(ThrowingRunnable body, String... keyValues)
            throws Exception {
        String[] previous = new String[keyValues.length / 2];
        for (int i = 0; i < previous.length; i++) {
            previous[i] = System.getProperty(keyValues[2 * i]);
            System.setProperty(keyValues[2 * i], keyValues[2 * i + 1]);
        }
        try {
            body.run();
        } finally {
            for (int i = 0; i < previous.length; i++) {
                if (previous[i] == null) System.clearProperty(keyValues[2 * i]);
                else System.setProperty(keyValues[2 * i], previous[i]);
            }
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
