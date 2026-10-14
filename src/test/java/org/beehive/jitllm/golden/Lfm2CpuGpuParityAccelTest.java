package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * LFM2.5-24B-A2B's logits against the CPU reference. See {@link CpuGpuParity}.
 *
 * <p>The gate on three things no other fixture exercises: the short-convolution window carrying
 * from one token to the next, the sigmoid router selecting the same experts on the device as on the
 * host, and expert projections that read only the selected experts' rows out of stacked Q4_K, Q5_K
 * and Q6_K tensors. A router that picks a different expert, or a kernel that reads the wrong
 * expert's rows, produces fluent text and logits this comparison cannot mistake.
 *
 * <p>The device keeps FP32 activations and decodes every weight in its own representation, the
 * arithmetic the host does, so this is held to the Q8_0 bounds.
 */
public class Lfm2CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void lfm2_5_24b_a2b_cpuGpuParity() throws Exception {
        assertParity(Fixture.LFM2_5_24B_A2B_Q4_K_XL, Q8_0);
    }

    @Test
    public void lfm2_5_24b_a2b_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.LFM2_5_24B_A2B_Q4_K_XL, Q8_0);
    }
}
