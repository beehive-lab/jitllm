package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Granite's logits against the CPU reference. See {@link CpuGpuParity}. */
public class GraniteCpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void granite_3_2_2b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.GRANITE_3_2_2B_F16, FP16);
    }

    @Test
    public void granite_3_2_2b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.GRANITE_3_2_2B_Q8_0, Q8_0);
    }
}
