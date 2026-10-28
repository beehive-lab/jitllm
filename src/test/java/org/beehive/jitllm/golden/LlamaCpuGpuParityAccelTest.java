package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Llama's logits against the CPU reference. See {@link CpuGpuParity}. */
public class LlamaCpuGpuParityAccelTest extends CpuGpuParity {

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
    public void llama3_2_1b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0);
    }

    @Test
    public void llama3_2_1b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_F16, FP16);
    }

    /** A Q4_0 file kept as it is: Q4_0 and Q4_1 read per tensor. */
    @Test
    public void llama3_2_1b_unsloth_q4_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_0, Q8_0);
    }

    /** A Q4_K_M file kept as it is: Q4_K and Q6_K read per tensor. */
    @Test
    public void llama3_2_1b_unsloth_q4_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_M, Q8_0);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_k_m_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_M, Q8_0);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_1_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_1, Q8_0);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_k_s_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_S, Q8_0);
    }

    @Test
    public void llama3_2_1b_unsloth_q5_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q5_K_M, Q8_0);
    }

    @Test
    public void llama3_2_1b_unsloth_q6_k_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_Q6_K, Q8_0);
    }

    /** Q8_0 and Q6_K projections in one layer. */
    @Test
    public void llama3_2_1b_unsloth_ud_q6_k_xl_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_UNSLOTH_UD_Q6_K_XL, Q8_0);
    }
}
