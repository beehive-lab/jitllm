package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen2-MoE's logits against the CPU reference with the accelerator in batched prefill. One method
 * per JVM, as {@link Qwen2MoECpuGpuParityAccelTest} says.
 */
public class Qwen2MoEBatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final int BATCH = 128;

    /**
     * The packed-integer path off, as in {@link Qwen3CpuGpuParityAccelTest}, which says what it
     * computes instead and why these bounds do not describe it.
     */
    static {
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    // @formatter:off
    /**
     * The bounds for the prompt rows. On tensor cores (CUDA) the experts are int8 GEMMs over the
     * activation quantized to eight bits per block of 32, at every prompt position, whatever the
     * file stores: {@link #QWEN2_MOE_INT8_BATCHED}. Elsewhere the prefill keeps FP32 activations
     * and {@link #Q8_0} applies.
     */
    // @formatter:on
    private static Bounds prefill() {
        return TensorCoreSupport.isTensorCoreCapableBackend() ? QWEN2_MOE_INT8_BATCHED : Q8_0;
    }

    @Test
    public void qwen1_5_moe_q8_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN1_5_MOE_A2_7B_Q8_0, prefill(), BATCH);
    }

    @Test
    public void qwen1_5_moe_q4_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN1_5_MOE_A2_7B_Q4_0, prefill(), BATCH);
    }

    @Test
    public void qwen1_5_moe_q6_k_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN1_5_MOE_A2_7B_Q6_K, prefill(), BATCH);
    }
}
