package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Qwen3's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Qwen3CpuGpuParityAccelTest extends CpuGpuParity {

    // @formatter:off
    /**
     * The packed-integer path off, through its exact-comparison switch, so the GPU computes what
     * the CPU reference computes and the {@link #Q8_0} bounds hold on every device.
     *
     * <p>Where the path is on (CUDA, by default), the projections quantize their activation to
     * eight bits per block of 32, as llama.cpp's do. That is a different arithmetic, not a defect,
     * and these bounds do not describe it. Measured on Qwen3-0.6B Q8_0 over these 63 rows: the
     * packed CUDA path against the CPU reference, relative L2 0.048 and cosine 0.99902; the CPU
     * reference itself with eight-bit activations ({@code -Djitllm.quantizeActivation=true})
     * against the floating-point GPU path, relative L2 0.042 and cosine 0.99912. The quantization
     * alone accounts for the distance. The packed kernels are checked against the host in {@code
     * KQuantDp4aDecodeProjectionAccelTest}, {@code Q6_KDp4aAccelTest} and {@code
     * Q8_0PackedDecodeAccelTest}.
     */
    // @formatter:on
    static {
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

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

    /** A Q4_K_M file kept as it is: Q4_K and Q6_K read per tensor. */
    @Test
    public void qwen3_0_6b_q4_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_0_6B_Q4_K_M, Q8_0);
    }

    @Test
    public void qwen3_1_7b_q4_k_m_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_1_7B_Q4_K_M, Q8_0);
    }

    @Test
    public void qwen3_0_6b_q4_k_m_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.QWEN3_0_6B_Q4_K_M, Q8_0);
    }
}
