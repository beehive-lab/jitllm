package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Llama's logits against the CPU reference with the accelerator in batched prefill.
 *
 * <p>Its own class, not more cases on {@link LlamaCpuGpuParityAccelTest}: surefire forks per class
 * and device memory returns to TornadoVM's buffer provider rather than to the driver, so loading
 * two modes of the same fixture in one JVM is what the split into per-family classes already exists
 * to avoid.
 */
public class LlamaBatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    /** A whole tensor-core tile, so the GEMM path runs unpadded. */
    private static final int BATCH = 128;

    /**
     * The packed-integer path off, as in {@link Qwen3CpuGpuParityAccelTest}, which says what it
     * computes instead and why these bounds do not describe it.
     */
    static {
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    /**
     * The bounds for a quantized file's prompt rows: {@link #FP16} where the batched prefill runs
     * on tensor cores (CUDA), whose GEMMs take FP16 operands, and {@link #Q8_0} elsewhere. {@link
     * Qwen3BatchedPrefillParityAccelTest} has the measurements, this file's Q8_0 among them.
     */
    private static Bounds quantizedPrefill() {
        return TensorCoreSupport.isTensorCoreCapableBackend() ? FP16 : Q8_0;
    }

    @Test
    public void llama3_2_1b_q8_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_Q8_0, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_f16_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_F16, FP16, BATCH);
    }

    /**
     * The Q4_0 layers' projections are packed where the device allows ({@link #LLAMA_Q4_0_PACKED}).
     */
    @Test
    public void llama3_2_1b_unsloth_q4_0_batchedPrefillParity() throws Exception {
        assertParityBatched(
                Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_0,
                PackedActivationTasks.supported() ? LLAMA_Q4_0_PACKED : quantizedPrefill(),
                BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_k_m_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_M, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_1_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_1, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_q4_k_s_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_S, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_q5_k_m_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_Q5_K_M, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_q6_k_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_Q6_K, quantizedPrefill(), BATCH);
    }

    @Test
    public void llama3_2_1b_unsloth_ud_q6_k_xl_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_UNSLOTH_UD_Q6_K_XL, quantizedPrefill(), BATCH);
    }
}
