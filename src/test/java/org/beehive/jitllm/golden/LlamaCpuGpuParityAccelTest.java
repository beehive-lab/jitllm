package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Llama's logits against the CPU reference. See {@link CpuGpuParity}. */
public class LlamaCpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void llama3_2_1b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0);
    }

    @Test
    public void llama3_2_1b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_F16, FP16);
    }
}
