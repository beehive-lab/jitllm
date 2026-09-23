package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * The comparison is <b>bit-identical logits</b> between the lowered and legacy paths, taken in one
 * process against one file — a stronger statement than a recorded golden, and it needs no committed
 * data. Every case asserts {@link LoweredPlanSelection#loweredPlanCount()} moved: a family whose
 * description or validation was wrong would fall back to the legacy plan and produce a perfect
 * match that proves nothing.
 *
 * <p><b>One family per class</b>, because the accel suite forks one JVM per class and a closed
 * session's device memory does not come back. Two families of models in one JVM exhausts the
 * device; it did, which is why this is three classes rather than one.
 */
public class LoweredMistralParityAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void mistralLowersAndReproducesTheLegacyLogitsExactly() throws Exception {
        assertLoweredMatchesLegacy(Fixture.MISTRAL_7B_Q8_0);
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
