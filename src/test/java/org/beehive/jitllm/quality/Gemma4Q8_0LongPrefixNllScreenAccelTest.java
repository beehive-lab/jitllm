package org.beehive.jitllm.quality;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * {@link Gemma4BatchedPrefillNllScreenAccelTest} on the Q8_0 file with a prefix past the sliding
 * window, at the width the benchmark runs.
 *
 * <p>The short screen's 320-token prefix sits inside the 512-position window, so a sliding layer
 * never drops a key there and a mistake in where a chunk's window starts cannot show. This one
 * ingests 1100 positions in chunks of 512 — two whole chunks and a partial one — so every chunk
 * after the first attends a window that begins inside an earlier chunk, and then scores decode
 * positions whose window has moved past everything the first chunk wrote.
 *
 * <p>Same limit as the short screen, against the same host reference. Its own class, and so its own
 * JVM: device memory a closed session frees returns to TornadoVM's buffer provider rather than to
 * the driver.
 */
// @formatter:on
public class Gemma4Q8_0LongPrefixNllScreenAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static final Gemma4BatchedPrefillNllScreenAccelTest.Schedule LONG =
            new Gemma4BatchedPrefillNllScreenAccelTest.Schedule(512, 1100, 48, 9000);

    @Test
    public void aPrefixPastTheWindowDoesNotMakeHeldOutTextLessLikely() throws Exception {
        Gemma4BatchedPrefillNllScreenAccelTest.screen(Fixture.GEMMA_4_E2B_Q8_0, LONG);
    }
}
