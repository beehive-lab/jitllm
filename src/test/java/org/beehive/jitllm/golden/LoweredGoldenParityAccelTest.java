package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * {@code GoldenLogitsAccelTest} and {@code CpuGpuParity} build their plan through {@code
 * GoldenCapture}, which called {@code TornadoVMMasterPlan.initializeTornadoVMPlan} directly — so
 * they never reached the selection branch, and setting the opt-in would have changed nothing while
 * appearing to. {@code GoldenCapture} now routes through the lowering when it applies, and <b>every
 * test here asserts on {@link LoweredPlanSelection#loweredPlanCount()}</b>, not on the property. A
 * run where the lowering did not execute fails.
 */
public class LoweredGoldenParityAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    /** Bounds copied from {@code CpuGpuParity} so the criterion is the same one. */
    private static final double F16_ATOL = 1.5e-2;

    private static final double F16_RTOL = 1e-2;

    /**
     * Q8_0 tracks the host far more closely than F16 does — it keeps FP32 activations — so it gets
     * the tighter bounds rather than F16's, which it would pass trivially.
     */
    private static final double Q8_0_ATOL = 5e-4;

    private static final double Q8_0_RTOL = 1e-2;

    /**
     * The lowered path reproduces the legacy path's logits <b>bit for bit</b>.
     *
     * <p>Stronger than the recorded golden, and the right comparison for this slice: the question
     * is not "does the model still work" but "does lowering change anything". Both captures run in
     * one process against one file, so a difference can only come from the plan construction.
     */
    @Test
    public void theLoweredPathReproducesTheLegacyLogitsExactly() throws Exception {
        assertLoweredLogitsMatchLegacy(Fixture.LLAMA_3_2_1B_F16);
    }

    @Test
    public void theLoweredPathReproducesTheLegacyLogitsExactly_q8_0() throws Exception {
        assertLoweredLogitsMatchLegacy(Fixture.LLAMA_3_2_1B_Q8_0);
    }

    private void assertLoweredLogitsMatchLegacy(Fixture fixture) throws Exception {
        Path model = fixtureOrSkip(fixture);

        long beforeLegacy = LoweredPlanSelection.loweredPlanCount();
        GoldenCapture.Result legacy = capture(model, false);
        assertEquals(
                "the legacy capture must not reach the lowering",
                beforeLegacy,
                LoweredPlanSelection.loweredPlanCount());

        long beforeLowered = LoweredPlanSelection.loweredPlanCount();
        GoldenCapture.Result lowered = capture(model, true);
        assertLoweringRan(beforeLowered);

        assertEquals("emitted token ids", legacy.tokenIds, lowered.tokenIds);
        assertEquals("number of captured rows", legacy.rows.size(), lowered.rows.size());
        for (int r = 0; r < legacy.rows.size(); r++) {
            float[] a = legacy.rows.get(r);
            float[] b = lowered.rows.get(r);
            assertEquals("row " + r + " width", a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                    assertEquals(
                            "logits row "
                                    + r
                                    + ", element "
                                    + i
                                    + " differs between the legacy and lowered paths",
                            a[i],
                            b[i],
                            0f);
                }
            }
        }
    }

    /**
     * CPU/GPU parity on the lowered path, against the same bounds the existing gate uses.
     *
     * <p>The CPU side cannot reach the lowering — there is no plan — so only the GPU capture
     * asserts the counter.
     */
    @Test
    public void theLoweredPathAgreesWithTheHostWithinTheParityBounds() throws Exception {
        assertParityAgainstHost(Fixture.LLAMA_3_2_1B_F16, F16_ATOL, F16_RTOL);
    }

    @Test
    public void theLoweredPathAgreesWithTheHostWithinTheParityBounds_q8_0() throws Exception {
        assertParityAgainstHost(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0_ATOL, Q8_0_RTOL);
    }

    private void assertParityAgainstHost(Fixture fixture, double atol, double rtol)
            throws Exception {
        Path model = fixtureOrSkip(fixture);
        GoldenCapture.assertHostLogitsAvailable();

        GoldenCapture.Result cpu = GoldenCapture.capture(model, false);

        long before = LoweredPlanSelection.loweredPlanCount();
        GoldenCapture.Result gpu = capture(model, true);
        assertLoweringRan(before);

        assertEquals("row count", cpu.rows.size(), gpu.rows.size());
        double worst = 0;
        for (int r = 0; r < cpu.rows.size(); r++) {
            float[] host = cpu.rows.get(r);
            float[] device = gpu.rows.get(r);
            for (int i = 0; i < host.length; i++) {
                double bound = atol + rtol * Math.abs(host[i]);
                double delta = Math.abs(device[i] - host[i]);
                worst = Math.max(worst, delta - bound);
                assertTrue(
                        "row "
                                + r
                                + ", element "
                                + i
                                + ": |gpu-cpu| = "
                                + delta
                                + " exceeds "
                                + bound,
                        delta <= bound);
            }
        }
        assertTrue("worst margin " + worst, worst <= 0);
    }

    private static GoldenCapture.Result capture(Path model, boolean lowered) throws Exception {
        String previous = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        if (lowered) {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "on");
        } else {
            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
        }
        try {
            return GoldenCapture.capture(model, true);
        } finally {
            if (previous == null) {
                System.clearProperty(LoweredPlanSelection.ENABLE_PROPERTY);
            } else {
                System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, previous);
            }
        }
    }

    private static void assertLoweringRan(long before) {
        assertTrue(
                "the lowered path did not run — this capture fell back to legacy construction,"
                        + " so the result says nothing about the lowering",
                LoweredPlanSelection.loweredPlanCount() > before);
    }

    private static Path fixtureOrSkip(Fixture fixture) {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), false);
        }
        return model;
    }
}
