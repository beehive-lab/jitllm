package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Qwen3's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Qwen3CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen3_0_6b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_0_6B_F16, FP16);
    }

    @Test
    public void qwen3_0_6b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_0_6B_Q8_0, Q8_0);
    }
}
