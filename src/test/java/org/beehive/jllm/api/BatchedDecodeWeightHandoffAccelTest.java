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
 * The batched prefill/decode plan must decode what the single-token plan decodes.
 *
 * <p>This guards one specific, silent failure mode. The batched decode layer graphs do not upload
 * their own weights: they bind them from the matching batch-prefill layer graph with {@code
 * consumeFromDevice("batchPrefillLayer_<i>", …)}, to avoid a second device copy. TornadoVM builds a
 * graph from its tasks' argument lists, so a weight the producer declares in {@code
 * transferToDevice} but hands to no task is never allocated and never uploaded — and the consumer
 * then binds a buffer nothing ever wrote. It reads zeros. Nothing throws.
 *
 * <p>That is not hypothetical: replacing the batch-prefill {@code gateUpProj} kernel with a single
 * cuBLAS GEMM over a stacked {@code [gate|up]} weight took {@code w1} and {@code w3} out of every
 * prefill task's argument list, decode's FFN then contributed exactly zero at every layer, and the
 * model emitted {@code ",,,,"}. Prefill's own output was bit-identical at every layer throughout.
 *
 * <p>The single-token plan is immune — its layer graphs upload every weight themselves ({@code
 * weightSourceGraphName()} returns null) — which is what makes it a usable reference here. Two
 * things this deliberately does <b>not</b> do, because both were tried and neither works:
 *
 * <ul>
 *   <li>Inspect the source for weights that reach no task. The defect kept the JIT task in an
 *       {@code else} branch, so the reference was still there to find.
 *   <li>Compare total bytes copied to the device. The defect substituted 336 MiB of stacked weights
 *       for the 336 MiB of {@code w1}/{@code w3} it stopped uploading; the totals matched to within
 *       12 MiB.
 * </ul>
 *
 * <p>The two plans run numerically different kernels, so this asserts token-level agreement over a
 * short greedy generation, not bit equality. If that ever proves brittle for a model, the fix is a
 * shorter generation or a logit tolerance — not deleting the check.
 */
public class BatchedDecodeWeightHandoffAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final int BATCH = 128;
    private static final int NEW_TOKENS = 24;

    /** Long enough to prefill as a batch, short enough to stay inside one chunk. */
    private static final String PROMPT =
            "Answer with one word. The capital of France is a city whose name every schoolchild"
                    + " in Europe learns. What is that city called?";

    @Test
    public void batchedPrefillDecodeAgreesWithTheSingleTokenPlan() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            String batched = generate(model, ExecutionPolicy.PhaseStrategy.PREFILL_DECODE);
            String singleToken = generate(model, ExecutionPolicy.PhaseStrategy.SINGLE_TOKEN);

            assertTrue("the single-token plan produced no text", !singleToken.isBlank());
            assertEquals(
                    "the batched prefill/decode plan decoded something else. If a batch-prefill"
                            + " task stopped reading a weight, that weight is no longer uploaded and"
                            + " the decode graphs consuming it from that graph are reading zeros;"
                            + " Qwen3FP16FFNLayers.weightsNotProvidedBySource is where the consumer"
                            + " takes ownership of such a weight",
                    singleToken,
                    batched);
        } finally {
            if (previousGpu == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previousGpu);
            }
        }
    }

    private static String generate(Path model, ExecutionPolicy.PhaseStrategy strategy)
            throws Exception {
        ExecutionPolicy.Builder policy = ExecutionPolicy.builder().phaseStrategy(strategy);
        if (strategy == ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            policy.prefillBatchSize(BATCH);
        }
        ModelOptions options =
                ModelOptions.builder().contextLength(512).executionPolicy(policy.build()).build();
        try (LocalModel loaded = LocalModels.load(model, options)) {
            TextGenerationModel generator = (TextGenerationModel) loaded;
            try (GenerationSession session = generator.newSession()) {
                return session.generate(
                                GenerationRequest.builder()
                                        .prompt(PROMPT)
                                        .maxNewTokens(NEW_TOKENS)
                                        .temperature(0.0f)
                                        .seed(42L)
                                        .build())
                        .text();
            }
        }
    }
}
