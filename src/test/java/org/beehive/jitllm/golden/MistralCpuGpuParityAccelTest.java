package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Mistral's logits against the CPU reference. See {@link CpuGpuParity}. */
public class MistralCpuGpuParityAccelTest extends CpuGpuParity {

    /**
     * The packed-integer path off, as in {@link Qwen3CpuGpuParityAccelTest}, which says what it
     * computes instead and why these bounds do not describe it.
     */
    static {
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void mistral_7b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.MISTRAL_7B_Q8_0, Q8_0);
    }

    /** A Q4_K_M file kept as it is: Q4_K and Q6_K read per tensor. */
    @Test
    public void mistral_7b_q4_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.MISTRAL_7B_Q4_K_M, Q8_0);
    }
}
