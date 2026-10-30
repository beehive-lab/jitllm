package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
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
     * The bounds for a quantized file's prompt rows. Where the batched prefill runs on tensor cores
     * (CUDA), its GEMMs take FP16 operands, the weights decoded to FP16 and the activations rounded
     * to it: the arithmetic {@link #FP16} describes, whatever the file stores. Elsewhere the
     * prefill keeps FP32 activations and {@link #Q8_0} applies.
     *
     * <p>Measured with the packed path off on CUDA over these 63 rows, against {@link #Q8_0}: Q8_0
     * 0.28% elementwise violations, relative L2 7.6e-4; Q4_K_M 0.6B 0.30% and 1.25e-3; Q4_K_M 1.7B
     * 0.21% and 9.7e-4; cosine at least 0.9999992 and no argmax disagreement on any. The F16 file,
     * through the same GEMMs, measures 1.18e-3. Llama-3.2-1B Q8_0 measures the same on its batched
     * path (0.25%, 5.5e-4), so this is the prefill arithmetic, not Qwen3's layers.
     */
    // @formatter:on
    private static Bounds quantizedPrefill() {
        return TensorCoreSupport.isTensorCoreCapableBackend() ? FP16 : Q8_0;
    }

    @Test
    public void qwen3_0_6b_q8_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q8_0, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_f16_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_F16, FP16, BATCH);
    }

    @Test
    public void qwen3_0_6b_q4_k_m_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q4_K_M, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_1_7b_q4_k_m_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_1_7B_Q4_K_M, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q4_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q4_0, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q4_1_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q4_1, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q4_k_s_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q4_K_S, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q5_k_m_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q5_K_M, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q5_k_s_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q5_K_S, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_q6_k_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_Q6_K, quantizedPrefill(), BATCH);
    }

    @Test
    public void qwen3_0_6b_ud_q6_k_xl_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_UD_Q6_K_XL, quantizedPrefill(), BATCH);
    }
}
