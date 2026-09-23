package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Phi3's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Phi3CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void phi3_mini_4k_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.PHI3_MINI_4K_F16, FP16);
    }

    @Test
    public void phi3_mini_4k_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.PHI3_MINI_4K_Q8_0, Q8_0);
    }
}
