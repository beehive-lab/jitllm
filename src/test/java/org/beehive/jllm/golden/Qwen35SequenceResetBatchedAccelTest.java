package org.beehive.jllm.golden;

import org.junit.Test;

/**
 * The reset property on the batched-prefill path, which is what the benchmark runs.
 *
 * <p>Its own class, and therefore its own JVM: this fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process runs out partway through.
 */
public class Qwen35SequenceResetBatchedAccelTest extends Qwen35SequenceReset {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceBatched() throws Exception {
        assertResetRestoresTheSequence(32);
    }
}
