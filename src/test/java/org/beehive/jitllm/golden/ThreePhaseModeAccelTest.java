package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * <b>This runs all three in one JVM</b>, which is itself the point: the mode used to be a {@code
 * static final} read at class initialization, so a test could exercise exactly one per process. It
 * is policy now.
 *
 * <h2>What the three modes actually do, measured</h2>
 *
 * <p>What this test therefore pins for the batched mode is that it <b>runs and produces a full
 * response</b> through the unified loop — including the generated-token budget and the Qwen2-MoE
 * seed handling that only its copy of the decode loop used to have.
 */
public class ThreePhaseModeAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final String PREFILL_DECODE = "jitllm.withPrefillDecode";
    private static final String PREFILL_BATCH = "jitllm.prefillBatchSize";

    /**
     * The two modes every backend must run. No capability gate: unlike batched prefill, sequential
     * prefill/decode uses the same non-MMA kernels single-token does.
     */
    @Test
    public void singleTokenAndSequentialPrefillDecodeAgree() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousMode = System.getProperty(PREFILL_DECODE);
        String previousBatch = System.getProperty(PREFILL_BATCH);
        try {
            System.clearProperty(PREFILL_DECODE);
            System.clearProperty(PREFILL_BATCH);
            GoldenCapture.Result singleToken = GoldenCapture.capture(model, true);

            System.setProperty(PREFILL_DECODE, "true");
            System.clearProperty(PREFILL_BATCH);
            GoldenCapture.Result prefillDecode = GoldenCapture.capture(model, true);

            assertTrue("the capture must actually produce tokens", singleToken.tokenIds.size() > 1);
            assertEquals(
                    "sequential prefill/decode must emit what single-token emits — it did not"
                            + " once, because its loop never received the seed fix",
                    singleToken.tokenIds,
                    prefillDecode.tokenIds);
        } finally {
            restore(PREFILL_DECODE, previousMode);
            restore(PREFILL_BATCH, previousBatch);
        }
    }

    /**
     * Batched prefill/decode. Runs on every backend — deliberately <b>not</b> gated on {@link
     * TensorCoreSupport#isTensorCoreCapableBackend()}.
     *
     * <p>It was gated for Metal parity task 9, on the reading that TornadoVM lowers the MMA
     * batch-prefill kernels only on CUDA. That reading does not apply to <em>this</em> path: the
     * plan components pick the MMA layer class only when the device advertises {@code
     * TENSOR_CORE_MMA} (see {@code LlamaQ8_0PlanComponents#batchPrefillTransformerLayers}), so a
     * Metal device takes the non-MMA class and never reaches an MMA kernel. What actually failed
     * here was TornadoVM 5.2.0's interpreter — {@code consumeFromDevice} resolving against the
     * previously executed graph rather than the named one, surfacing as {@code
     * NullPointerException: XPUDeviceBufferState.getXPUBuffer() is null}. Fixed upstream in PR #996
     * and released in 6.0.0, which this project now pins; with that pin this test passes on Metal,
     * teacher-forced parity assertion included.
     *
     * <p>The MMA gap is real, but it belongs to the multi-sequence <em>engine</em> path, where
     * {@code BatchedEngineAccelTest} still gates on the capability and still bails on {@code
     * gemmMMAQKV} — "MMA instructions only supported for the CUDA backend". Keep that gate; this
     * one was measuring a different thing.
     */
    @Test
    public void batchedPrefillDecodeAgreesWithSingleToken() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        String previousMode = System.getProperty(PREFILL_DECODE);
        String previousBatch = System.getProperty(PREFILL_BATCH);
        try {
            System.clearProperty(PREFILL_DECODE);
            System.clearProperty(PREFILL_BATCH);
            GoldenCapture.Result singleToken = GoldenCapture.capture(model, true);

            // Teacher-forced along single-token's own tokens, so every row compares the same
            // context and a difference is arithmetic rather than history.
            System.setProperty(PREFILL_DECODE, "true");
            System.setProperty(PREFILL_BATCH, "8");
            GoldenCapture.Result batched = GoldenCapture.capture(model, true, singleToken.tokenIds);

            assertEquals(
                    "batched prefill must emit exactly what single-token emits",
                    singleToken.tokenIds,
                    batched.tokenIds);
            assertFalse(
                    "and it must produce real tokens, not repeats of one",
                    batched.tokenIds.stream().distinct().count() < 3);

            assertLogitsAgree(singleToken, batched);
        } finally {
            restore(PREFILL_DECODE, previousMode);
            restore(PREFILL_BATCH, previousBatch);
        }
    }

    /** FP16's bounds, copied from {@code CpuGpuParity} so the criterion is the same one. */
    private static final double ATOL = 1.5e-2;

    private static final double RTOL = 1e-2;

    /**
     * This is how the 27.801057 was measured, and it is now how the fix is held. The bounds are
     * unchanged; what changed is the sequence the batched path feeds. Measured after the fix, the
     * worst excess is <b>0.0</b> — the two paths agree bit-for-bit, not merely within tolerance.
     *
     * <p>Compares the decode rows the two runs have in common.
     *
     * <p>The tail alignment is kept. It is a no-op now that both runs emit the same number of rows,
     * and keeping it costs nothing while making the comparison independent of that fact — this
     * method's job is to compare decode steps against decode steps, not to assume a row count.
     */
    private static void assertLogitsAgree(
            GoldenCapture.Result expected, GoldenCapture.Result actual) {
        int rows = Math.min(expected.rows.size(), actual.rows.size());
        assertTrue("there must be decode rows to compare", rows > 8);
        int expectedOffset = expected.rows.size() - rows;
        int actualOffset = actual.rows.size() - rows;
        double worst = 0;
        for (int r = 0; r < rows; r++) {
            float[] a = expected.rows.get(expectedOffset + r);
            float[] b = actual.rows.get(actualOffset + r);
            assertEquals("row " + r + " width", a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                double bound = ATOL + RTOL * Math.abs(a[i]);
                worst = Math.max(worst, Math.abs(b[i] - a[i]) - bound);
            }
        }
        assertTrue(
                "batched prefill's logits leave the FP16 parity bounds by "
                        + worst
                        + "; the paths run different kernels, but not that different",
                worst <= 0);
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
