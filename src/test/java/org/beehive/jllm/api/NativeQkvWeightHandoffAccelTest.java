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
 * The native packed QKV projection must not change what the model generates.
 *
 * <p>{@code -Djllm.projection.cublas.qkv=true} replaces the batch-prefill {@code gemmMMAQKV}
 * kernel with one cuBLAS GEMM over a stacked {@code [q|k|v]} weight, which leaves {@code wq},
 * {@code wk} and {@code wv} as arguments to no task in {@code batchPrefillLayer_<i>}. The decode
 * layer graphs bind their weights from that graph, and TornadoVM never uploads a weight no task
 * reads, so all three have to be taken over by the consumer — see {@link
 * org.beehive.jllm.backend.tornado.layers.type.fp16.Qwen3FP16FFNLayers#weightsNotProvidedBySource}.
 * Exactly the failure the gate/up projection hit first, with three weights instead of two.
 *
 * <p>This runs the flag alone against the flag off. The combination with native gate/up — where a
 * decode layer graph has to take ownership of five weights at once — is covered by the numerical
 * matrix rather than here, because a second pair of model loads in one JVM is not worth the device
 * memory. Two loads per class, one JVM per class ({@code reuseForks=false}).
 */
public class NativeQkvWeightHandoffAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String QKV_PROPERTY = "jllm.projection.cublas.qkv";
    private static final int BATCH = 128;

    @Test
    public void nativeQkvDecodesIdenticallyToTheJitPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousQkv = System.getProperty(QKV_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            String jit = generate(model, false);
            String nativeQkv = generate(model, true);

            assertTrue("the JIT QKV path produced no text", !jit.isBlank());
            assertEquals(
                    "the native QKV projection changed what the model decodes; the usual cause is"
                        + " the batch-prefill graph no longer uploading wq/wk/wv, which the decode"
                        + " graphs consume from it",
                    jit,
                    nativeQkv);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(QKV_PROPERTY, previousQkv);
        }
    }

    private static String generate(Path model, boolean nativeQkv) throws Exception {
        System.setProperty(QKV_PROPERTY, Boolean.toString(nativeQkv));
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

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
