package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * The reset property for Gemma 4's Q4_0 file on the batched-prefill path, which is what the
 * benchmark runs: a fresh sequence, a longer different one, a reset, and the first sequence again,
 * compared bit for bit.
 *
 * <p>A width of seven, so the prompt is ingested as whole chunks and a partial final one, and the
 * polluting sequence leaves key/value entries beyond every position the replay reads.
 *
 * <p>Its own class, and therefore its own JVM: device memory a closed session frees returns to
 * TornadoVM's buffer provider rather than to the driver.
 */
public class Gemma4Q4_0SequenceResetBatchedAccelTest extends Qwen35SequenceReset {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Override
    Fixture fixture() {
        return Fixture.GEMMA_4_E2B_Q4_0;
    }

    @Test
    public void gemma4E2bQ40ResetRestoresTheSequenceBatched() throws Exception {
        assertResetRestoresTheSequence(7);
    }
}
