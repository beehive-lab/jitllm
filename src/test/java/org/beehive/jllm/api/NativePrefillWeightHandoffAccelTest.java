package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.NativePrefillSupport;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * The native prefill path must not change what the model generates.
 *
 * <p>Turning it on replaces the batch-prefill {@code qkvProj} and {@code gateUpProj} MMA kernels
 * with single cuBLAS GEMMs over stacked {@code [q|k|v]} and {@code [gate|up]} weights. That leaves
 * {@code wq}, {@code wk}, {@code wv}, {@code w1} and {@code w3} as arguments to <b>no task</b> in
 * {@code batchPrefillLayer_<i>}. The decode layer graphs bind their weights from that graph, and
 * TornadoVM builds a graph out of its tasks' argument lists — so a weight the producer declares in
 * {@code transferToDevice} but hands to no task is never allocated there, never uploaded, and the
 * consumer binds a buffer nothing ever wrote. It reads zeros. Nothing throws.
 *
 * <p>That is not hypothetical. It is how this path failed twice:
 *
 * <ul>
 *   <li><b>gate/up first.</b> Decode's FFN contributed exactly zero at every layer and the model
 *       emitted {@code ",,,,"}. Prefill's own output was bit-identical at every layer throughout.
 *   <li><b>then QKV,</b> the same failure with three weights instead of two.
 * </ul>
 *
 * <p>Both are fixed by {@code Qwen3FP16FFNLayersDecode.weightsNotProvidedBySource}, which takes
 * ownership of exactly those five. This test is what proves it still does. Two things it
 * deliberately does <b>not</b> do, because both were tried and neither works:
 *
 * <ul>
 *   <li>Inspect the source for weights that reach no task. The gate/up defect kept the JIT task in
 *       an {@code else} branch, so the reference was still there to find.
 *   <li>Compare total bytes copied to the device. The defect substituted 336 MiB of stacked weights
 *       for the 336 MiB of {@code w1}/{@code w3} it stopped uploading; the totals matched to within
 *       12 MiB.
 * </ul>
 *
 * <p>The two configurations run numerically different kernels, so this asserts token-level
 * agreement over a short greedy generation, not bit equality. If that ever proves brittle for a
 * model, the fix is a shorter generation or a logit tolerance — not deleting the check.
 *
 * <p>Replaces the two earlier per-flag tests (one for gate/up, one for QKV). The projections are no
 * longer independently selectable — {@code NativePrefillSupport} answers for all four at once — so
 * flipping that single answer exercises all five weights together, which is both strictly more
 * coverage and the configuration that actually ships.
 *
 * <p><b>It proves it ran the path it claims.</b> Selection is a capability decision, so on a host
 * without the tensor-core backend or without cuBLAS both halves of the comparison quietly select
 * the generated kernels and agree perfectly while covering nothing. The resolved answer is asserted
 * to differ between the two halves, and a host that cannot select the native path at all is an
 * explicit skip rather than a green tick.
 */
public class NativePrefillWeightHandoffAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String NATIVE_PROPERTY = "jllm.prefill.native";
    private static final int BATCH = 128;

    /**
     * Qwen3-0.6B's attention shape at {@link #BATCH}, for the diagnostic in the skip message only.
     * Nothing here depends on the fused attention: the handoff is about the projections.
     */
    private static final NativePrefillSupport.SdpaShape PROBE_SHAPE =
            new NativePrefillSupport.SdpaShape(1, 16, BATCH, BATCH, 128, 0.088388f, true);

    @Test
    public void nativePrefillDecodesIdenticallyToTheJitPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousNative = System.getProperty(NATIVE_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            System.setProperty(NATIVE_PROPERTY, "true");
            assumeTrue(
                    "this host does not select the native prefill projections, so there is no"
                            + " weight handoff to exercise: "
                            + NativePrefillSupport.describe(true, PROBE_SHAPE),
                    NativePrefillSupport.nativeProjections());
            System.setProperty(NATIVE_PROPERTY, "false");
            assertFalse(
                    "the off-switch did not switch anything off, so both halves of this"
                            + " comparison would run the same path and prove nothing",
                    NativePrefillSupport.nativeProjections());

            String jit = generate(model, false);
            String nativePrefill = generate(model, true);

            assertTrue("the JIT prefill path produced no text", !jit.isBlank());
            assertEquals(
                    "the native prefill path changed what the model decodes; the usual cause is the"
                            + " batch-prefill graph no longer uploading wq/wk/wv/w1/w3, which the"
                            + " decode graphs consume from it — see"
                            + " Qwen3FP16FFNLayersDecode.weightsNotProvidedBySource",
                    jit,
                    nativePrefill);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(NATIVE_PROPERTY, previousNative);
        }
    }

    private static String generate(Path model, boolean nativePrefill) throws Exception {
        System.setProperty(NATIVE_PROPERTY, Boolean.toString(nativePrefill));
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
