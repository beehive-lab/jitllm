package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Separate from the Q8_0 case because the comparison loads the model <b>twice</b> — once per path —
 * and Phi3-mini at F16 is 7.6 GiB, so the pair does not fit beside anything else in a JVM whose
 * device memory a closed session does not return. One class, one fork, one pair.
 *
 * <p>The widths are what a device check earns here. A split that read a different width than its
 * projection produced would address past the end of the buffer, and the halves would still look
 * like halves — the tokens are the only thing that shows it.
 *
 * <p>The comparison is <b>bit-identical logits</b> between the lowered and legacy paths, taken in
 * one process against one file — a stronger statement than a recorded golden, and it needs no
 * committed data. Every case asserts {@link LoweredPlanSelection#loweredPlanCount()} moved: a
 * family whose description or validation was wrong would fall back to the legacy plan and produce a
 * perfect match that proves nothing.
 *
 * <p><b>One family per class</b>, because the accel suite forks one JVM per class and a closed
 * session's device memory does not come back. Two families of models in one JVM exhausts the
 * device; it did, which is why this is three classes rather than one.
 */
public class LoweredPhi3F16ParityAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void phi3LowersAndReproducesTheLegacyLogitsExactly_f16() throws Exception {
        assertLoweredMatchesLegacy(Fixture.PHI3_MINI_4K_F16);
    }

    private void assertLoweredMatchesLegacy(Fixture fixture) throws Exception {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), false);
        }

        long beforeLegacy = LoweredPlanSelection.loweredPlanCount();
        GoldenCapture.Result legacy = capture(model, false);
        assertEquals(
                "the legacy capture must not reach the lowering",
                beforeLegacy,
                LoweredPlanSelection.loweredPlanCount());

        long beforeLowered = LoweredPlanSelection.loweredPlanCount();
        GoldenCapture.Result lowered = capture(model, true);
        assertTrue(
                "the lowered path did not run — the comparison would be legacy against legacy",
                LoweredPlanSelection.loweredPlanCount() > beforeLowered);

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
}
