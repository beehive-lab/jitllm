package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Qwen2's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Qwen2CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen2_5_0_5b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN2_5_0_5B_F16, FP16);
    }

    @Test
    public void qwen2_5_0_5b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN2_5_0_5B_Q8_0, Q8_0);
    }
}
