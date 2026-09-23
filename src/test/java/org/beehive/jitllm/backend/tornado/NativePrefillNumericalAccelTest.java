package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import org.beehive.jitllm.Options;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

// @formatter:off
/**
 * The native prefill path against the generated-kernel path it replaces, over the chunk transitions
 * that actually broke during its development.
 *
 * <p>A maintained, compact form of the external harness the implementation was validated with. Two
 * requests of different lengths per case, separated by {@code resetSequenceState()} and built from
 * different tokens, each followed by decode steps — so reset, replay and prefill-to-decode
 * continuity are inside every figure rather than being separate cases.
 *
 * <h2>Why these two cases</h2>
 *
 * <ul>
 *   <li><b>360 then 200 at width 128, context 368.</b> Three chunks: cuDNN takes the first, the
 *       batched fallback takes a full chunk and then a partial one of 104 rows. The context is
 *       deliberately smaller than two chunks, so the key/value capacity is tight and a padded row
 *       that addressed an unallocated page would scatter over live keys. Decode here follows the
 *       fallback attention.
 *   <li><b>300 then 128 at width 512, context 320.</b> A <b>partial first chunk</b> — 300 real rows
 *       in a 512-wide graph — with a capacity smaller than the chunk width, which is the exact
 *       shape that once made the two attention paths disagree by relative L2 around 1.0. One chunk,
 *       so the fallback must not engage, and decode here follows the native attention.
 * </ul>
 *
 * <h2>What is asserted, and in what order</h2>
 *
 * <p>Non-finite values and a shape mismatch are rejected <b>before</b> any error metric, because a
 * comparison that reduces first can report a small number for a dump that is entirely NaN. Only
 * then the relative L2 over every decode step, at the tolerance the implementation was validated
 * at.
 *
 * <p>And each run's dispatch is read off its own grid scheduler first. Without that this test would
 * pass just as well on a host where both runs quietly selected the generated kernels, and would be
 * reporting coverage it does not have.
 */
// @formatter:on
public class NativePrefillNumericalAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String KV_FP16_PROPERTY = "jitllm.kvcache.fp16";

    /** The tolerance the implementation was validated at; measured cases land near 6e-04. */
    private static final double REL_L2_TOLERANCE = 2e-2;

    private static final int DECODE_STEPS = 4;

    @Test
    public void multiChunkWithTightCapacityAgreesWithTheGeneratedKernels() throws Exception {
        assertCaseAgrees(
                "360->200 @128, ctx 368 (native + JIT full + JIT partial)", 128, 360, 200, 368);
    }

    @Test
    public void aPartialFirstChunkWithTightCapacityAgreesWithTheGeneratedKernels()
            throws Exception {
        assertCaseAgrees(
                "300->128 @512, ctx 320 (partial first chunk, no fallback)", 512, 300, 128, 320);
    }

    private void assertCaseAgrees(String label, int batch, int len1, int len2, int context)
            throws Exception {
        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                model != null);

        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousKv = System.getProperty(KV_FP16_PROPERTY);
        String previousNative = System.getProperty(NativePrefillSupport.PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(KV_FP16_PROPERTY, "true");
        try {
            System.setProperty(NativePrefillSupport.PROPERTY, "true");
            Run candidate = run(model, batch, len1, len2, context);
            assumeTrue(
                    "this host does not select the native prefill path, so there is nothing to"
                            + " compare against: "
                            + candidate.evidence().describe(),
                    candidate.evidence().nativeProjections());

            System.setProperty(NativePrefillSupport.PROPERTY, "false");
            Run reference = run(model, batch, len1, len2, context);

            assertTrue(
                    label
                            + ": the reference run selected the native projections too, so this"
                            + " case compared the native path with itself — "
                            + reference.evidence().describe(),
                    !reference.evidence().nativeProjections());
            assertTrue(
                    label
                            + ": the reference run must carry the generated attention for every"
                            + " chunk — "
                            + reference.evidence().describe(),
                    reference.evidence().jitAttentionInPrimary()
                            && !reference.evidence().cudnnAttention());

            assertAgrees(label, reference, candidate);
        } finally {
            restore(NativePrefillSupport.PROPERTY, previousNative);
            restore(KV_FP16_PROPERTY, previousKv);
            restore(GPU_PROPERTY, previousGpu);
        }
    }

    /** One run's decode logits and the dispatch the plan was built with. */
    private record Run(
            PlanDispatchEvidence.NativePrefillEvidence evidence, int vocabulary, float[][] steps) {}

    private static Run run(Path model, int batch, int len1, int len2, int context)
            throws Exception {
        Options options =
                new Options(
                        model,
                        "numerical",
                        null,
                        null,
                        false,
                        0.0f,
                        1.0f,
                        42,
                        context,
                        false,
                        false,
                        true,
                        true,
                        batch);
        Model loaded = ModelLoader.loadModel(options);
        State state = loaded.createNewState();
        TornadoVMMasterPlanBatchPrefillDecode plan =
                (TornadoVMMasterPlanBatchPrefillDecode)
                        TornadoVMMasterPlan.initializeTornadoVMPlan(state, loaded);
        try {
            PlanDispatchEvidence.NativePrefillEvidence evidence =
                    PlanDispatchEvidence.qwen3NativePrefill(
                            PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
            int vocabulary = loaded.configuration().vocabularySize();
            // Different tokens per request, so a plan that replayed the first request's state
            // instead of resetting would not accidentally agree.
            Random random = new Random(2026);
            float[][] steps = new float[2 * DECODE_STEPS][];
            int at = 0;
            for (int len : new int[] {len1, len2}) {
                int[] tokens = new int[len + DECODE_STEPS];
                for (int i = 0; i < tokens.length; i++) {
                    tokens[i] = random.nextInt(vocabulary);
                }
                plan.resetSequenceState();
                for (int start = 0; start < len; start += batch) {
                    int n = Math.min(batch, len - start);
                    TornadoBatchPrefillPass.batchPrefill(
                            loaded,
                            state,
                            Arrays.copyOfRange(tokens, start, start + n),
                            start,
                            n,
                            plan);
                }
                for (int step = 0; step < DECODE_STEPS; step++) {
                    Logits logits =
                            TornadoBatchPrefillPass.decode(
                                    loaded, state, tokens[len + step], len + step, plan);
                    float[] row = new float[vocabulary];
                    for (int i = 0; i < vocabulary; i++) {
                        row[i] = logits.get(i);
                    }
                    steps[at++] = row;
                }
            }
            return new Run(evidence, vocabulary, steps);
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    // @formatter:off
    /**
     * Shape, then finiteness, then the metric.
     *
     * <p>That order is the point. {@code max(0, NaN)} is {@code 0} and a sum containing NaN divided
     * by a sum containing NaN can be anything, so a comparison that computes first can report a
     * passing relative L2 for a dump that is entirely non-finite. This rejects both before it
     * reduces anything.
     */
    // @formatter:on
    private static void assertAgrees(String label, Run reference, Run candidate) {
        assertEquals(
                label + ": vocabulary differs between the two runs",
                reference.vocabulary(),
                candidate.vocabulary());
        assertEquals(
                label + ": decode step count differs between the two runs",
                reference.steps().length,
                candidate.steps().length);
        for (int step = 0; step < reference.steps().length; step++) {
            float[] want = reference.steps()[step];
            float[] got = candidate.steps()[step];
            assertEquals(
                    label + ": logit count differs at decode step " + step,
                    want.length,
                    got.length);
            assertAllFinite(label + " reference, decode step " + step, want);
            assertAllFinite(label + " native, decode step " + step, got);
        }

        double numerator = 0.0;
        double denominator = 0.0;
        double worst = 0.0;
        for (int step = 0; step < reference.steps().length; step++) {
            float[] want = reference.steps()[step];
            float[] got = candidate.steps()[step];
            for (int i = 0; i < want.length; i++) {
                double difference = (double) got[i] - want[i];
                numerator += difference * difference;
                denominator += (double) want[i] * want[i];
                worst = Math.max(worst, Math.abs(difference));
            }
        }
        double relativeL2 = denominator == 0.0 ? 0.0 : Math.sqrt(numerator / denominator);
        assertTrue(
                label
                        + ": the native prefill path disagrees with the generated kernels —"
                        + " relative L2 "
                        + relativeL2
                        + " over "
                        + reference.steps().length
                        + " decode steps (tolerance "
                        + REL_L2_TOLERANCE
                        + "), largest absolute difference "
                        + worst,
                relativeL2 < REL_L2_TOLERANCE);
    }

    private static void assertAllFinite(String what, float[] values) {
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                throw new AssertionError(
                        what + ": logit " + i + " is " + values[i] + ", not a finite number");
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
