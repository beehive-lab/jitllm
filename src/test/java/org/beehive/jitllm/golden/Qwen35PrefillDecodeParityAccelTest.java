package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B with the prompt ingested as its own sequential phase, against the CPU reference.
 *
 * <p>Its own class, and therefore its own JVM: this fixture holds 15.5 GiB on the device, and
 * device memory a closed session frees returns to TornadoVM's buffer provider rather than to the
 * driver — two of these in one process exhausts the card partway through the second, which is what
 * {@link CpuGpuParity} says and what a shared class demonstrated.
 *
 * <p>Same bounds as the single-token path, because it is the same computation. A mode that needed
 * looser bounds would be computing something else, and that would be the finding rather than the
 * configuration.
 */
public class Qwen35PrefillDecodeParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen3_8_27b_q4_0_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.QWEN3_8_27B_Q4_0, Q8_0_FULLY_PACKED);
    }
}
