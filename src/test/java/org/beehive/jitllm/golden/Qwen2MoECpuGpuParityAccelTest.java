package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen2-MoE's logits against the CPU reference, single-token. See {@link CpuGpuParity}.
 *
 * <p>Each fixture is a 14B-parameter model: run one method per JVM ({@code -Dtest=Class#method}),
 * or the device memory a closed session leaves to TornadoVM's buffer provider runs out partway.
 */
public class Qwen2MoECpuGpuParityAccelTest extends CpuGpuParity {

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
    public void qwen1_5_moe_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN1_5_MOE_A2_7B_Q8_0, Q8_0);
    }

    /** Every projection and expert kept as Q4_0. */
    @Test
    public void qwen1_5_moe_q4_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN1_5_MOE_A2_7B_Q4_0, Q8_0);
    }

    /** Q6_K kept, with Q8_0 routed down projections in the same layers. */
    @Test
    public void qwen1_5_moe_q6_k_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN1_5_MOE_A2_7B_Q6_K, Q8_0);
    }
}
