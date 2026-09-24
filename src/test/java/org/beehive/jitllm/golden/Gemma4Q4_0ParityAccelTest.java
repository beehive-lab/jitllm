package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's Q4_0 file against the CPU reference.
 *
 * <p>Its own class, because surefire forks per class and this family's three fixtures cannot be
 * loaded in one JVM: 9.3 GB in BF16 and 5.0 GB in Q8_0, and device memory a closed session frees
 * returns to TornadoVM's buffer provider rather than to the driver, so the third load exhausts a 24
 * GB card and the failure lands on whichever fixture ran last rather than on whichever is wrong.
 *
 * <p>The device runs this file's projections <b>as Q4_0</b>, retained from the file, with the
 * packed-integer kernels that quantize their activation to eight bits; the host reference decodes
 * the same blocks in FP32. Hence {@link CpuGpuParity#Q4_0_PACKED_ACTIVATION} rather than the Q8_0
 * bounds.
 */
public class Gemma4Q4_0ParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ4_0CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_Q4_0, Q4_0_PACKED_ACTIVATION);
    }
}
