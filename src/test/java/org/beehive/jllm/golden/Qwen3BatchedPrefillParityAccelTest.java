package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3's logits against the CPU reference with the accelerator in batched prefill.
 *
 * <p>Llama and Qwen3 are the two families with a GPU batched-prefill implementation; every other
 * family refuses the mode by name, so there is nothing to compare for them.
 */
public class Qwen3BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    private static final int BATCH = 128;

    @Test
    public void qwen3_0_6b_q8_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q8_0, Q8_0, BATCH);
    }

    @Test
    public void qwen3_0_6b_f16_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_F16, FP16, BATCH);
    }
}
