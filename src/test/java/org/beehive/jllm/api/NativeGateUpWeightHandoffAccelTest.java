package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * The native gate/up projection must not change what the model generates.
 *
 * <p>{@code -Djllm.projection.cublas.gateUp=true} replaces the batch-prefill {@code gemmMMAGateUp}
 * kernel with one cuBLAS GEMM over a stacked {@code [gate|up]} weight. That leaves {@code w1} and
 * {@code w3} as arguments to no task in {@code batchPrefillLayer_<i>} — and the decode layer graphs
 * bind their weights with {@code consumeFromDevice("batchPrefillLayer_<i>", …)}. TornadoVM builds a
 * graph from its tasks' argument lists, so a weight no task reads is never uploaded, and decode
 * consumed a buffer nothing had written: {@code W1 = W3 = 0}, every decode FFN contributing exactly
 * zero, logits off by relL2 ≈ 7 while prefill's own output stayed bit-identical at every layer.
 *
 * <p>The regression is invisible to a prefill-only check, and invisible to anything that inspects
 * only the buffer the GEMM writes. It shows up in the first decoded token, which is what this
 * asserts: the same prompt, the same seed, greedy, with the flag off and on, must decode to the
 * same text. Off and on are compared in one process on one file, so no recorded golden is needed.
 */
public class NativeGateUpWeightHandoffAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String GATE_UP_PROPERTY = "jllm.projection.cublas.gateUp";
    private static final int BATCH = 128;

    @Test
    public void nativeGateUpDecodesIdenticallyToTheJitPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousGateUp = System.getProperty(GATE_UP_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            String jit = generate(model, false);
            String nativeGateUp = generate(model, true);

            assertTrue("the JIT gate/up path produced no text", !jit.isBlank());
            assertEquals(
                    "the native gate/up projection changed what the model decodes; the usual cause"
                        + " is the batch-prefill graph no longer uploading a weight the decode"
                        + " graphs consume from it",
                    jit,
                    nativeGateUp);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(GATE_UP_PROPERTY, previousGateUp);
        }
    }

    /** One session, greedy, with the native gate/up flag set as given. */
    private static String generate(Path model, boolean nativeGateUp) throws Exception {
        System.setProperty(GATE_UP_PROPERTY, Boolean.toString(nativeGateUp));
        ModelOptions options =
                ModelOptions.builder()
                        .contextLength(512)
                        .executionPolicy(
                                ExecutionPolicy.builder()
                                        .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                        .prefillBatchSize(BATCH)
                                        .build())
                        .build();
        try (LocalModel loaded = LocalModels.load(model, options)) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                // Long enough that prefill runs as a batch, short enough to stay one chunk.
                String prompt =
                        "Answer with one word. The capital of France is a city whose name every"
                                + " schoolchild in Europe learns. What is that city called?";
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .prompt(prompt)
                                        .maxNewTokens(24)
                                        .temperature(0.0f)
                                        .seed(42L)
                                        .build());
                return result.text();
            }
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
