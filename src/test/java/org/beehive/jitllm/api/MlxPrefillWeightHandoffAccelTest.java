package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * The MLX prefill must not change what the model decodes: the decode graphs bind from the prefill
 * graphs every weight those graphs still hold, and upload the ones MLX replaced.
 *
 * <p>Which ones those are depends on the representation, which is why each is exercised:
 *
 * <ul>
 *   <li>F16 — the prefill stacks Q/K/V and gate/up into copies, so it stops binding {@code wq},
 *       {@code wk}, {@code wv}, {@code w1} and {@code w3} and still binds {@code wo}, {@code w2}
 *       and the norms, which decode consumes;
 *   <li>Q8_0, affine or FP16 copies — the prefill binds none of the projections, so decode uploads
 *       all seven and consumes only the norms.
 * </ul>
 *
 * <p>A decode graph that consumed a weight the prefill graph never uploaded would read zeros — the
 * failure {@code NativePrefillWeightHandoffAccelTest} describes on the cuBLAS path — and the greedy
 * text would diverge at once. The prompt is longer than the width, so it spans several chunks and
 * the last is partial. The numerical side is {@code *MlxPrefillParityAccelTest}'s, against the CPU.
 */
// @formatter:on
public class MlxPrefillWeightHandoffAccelTest {

    private static final int BATCH = 16;

    @Test
    public void qwen3F16DecodesIdenticallyAfterAnMlxPrefill() throws Exception {
        assertSameText(Fixture.QWEN3_0_6B_F16, ExecutionPolicy.DecodedProjections.BACKEND);
    }

    @Test
    public void llamaQ8AffineDecodesIdenticallyAfterAnMlxPrefill() throws Exception {
        assertSameText(Fixture.LLAMA_3_2_1B_Q8_0, ExecutionPolicy.DecodedProjections.PER_CHUNK);
    }

    @Test
    public void llamaQ8Fp16CopiesDecodeIdenticallyAfterAnMlxPrefill() throws Exception {
        assertSameText(Fixture.LLAMA_3_2_1B_Q8_0, ExecutionPolicy.DecodedProjections.RESIDENT);
    }

    private static void assertSameText(Fixture fixture, ExecutionPolicy.DecodedProjections decoded)
            throws Exception {
        Path model = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), model != null);
        assumeTrue(
                "the MLX prefill runs on Metal only",
                BackendId.METAL.equals(TornadoDevices.current().backend()));
        assumeTrue("MLX's mlx.metallib is not loadable here", MlxPrefill.available());
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            String jit = generate(model, false, decoded);
            String mlx = generate(model, true, decoded);
            assertFalse("the JIT prefill produced no text", jit.isBlank());
            assertEquals(
                    fixture
                            + " "
                            + decoded
                            + ": the MLX prefill changed what the model decodes; the usual cause is a"
                            + " decode graph binding a weight the prefill graph no longer uploads —"
                            + " see MlxPrefillProjections.notBound",
                    jit,
                    mlx);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static String generate(
            Path model, boolean nativeLibraries, ExecutionPolicy.DecodedProjections decoded)
            throws Exception {
        ModelOptions options =
                ModelOptions.builder()
                        .contextLength(512)
                        .executionPolicy(
                                ExecutionPolicy.builder()
                                        .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                        .prefillBatchSize(BATCH)
                                        .nativeLibraries(nativeLibraries)
                                        .decodedProjections(decoded)
                                        .build())
                        .build();
        try (LocalModel loaded = LocalModels.load(model, options)) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                String prompt =
                        "Answer with one word. The capital of France is a city whose name every"
                                + " schoolchild in Europe learns. What is that city called?";
                return session.generate(
                                GenerationRequest.builder()
                                        .prompt(prompt)
                                        .maxNewTokens(24)
                                        .temperature(0.0f)
                                        .seed(42L)
                                        .build())
                        .text();
            }
        }
    }
}
