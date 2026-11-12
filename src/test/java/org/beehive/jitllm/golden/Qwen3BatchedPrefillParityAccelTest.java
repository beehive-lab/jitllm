package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
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
     * A quantized file's prompt rows against the host. Where the batched prefill runs on tensor
     * cores (CUDA), its GEMMs take FP16 operands, the weights decoded to FP16 and the activations
     * rounded to it: the arithmetic {@link #FP16} describes, whatever the file stores. On Metal's
     * SIMD-group matrices, a GEMM with enough workgroups stages its weights as FP16 and keeps the
     * activations and sums FP32 ({@code ...HalfTiles}). {@link #Q8_0} then applies against the host
     * with those weights rounded the same way, and {@link #FP16} against the exact host, because
     * this rounds less than the tensor-core GEMMs do. Elsewhere the prefill keeps FP32 weights and
     * activations, and {@link #Q8_0} applies against the exact host.
     *
     * <p>Measured with the packed path off on CUDA over these 63 rows, against {@link #Q8_0}: Q8_0
     * 0.28% elementwise violations, relative L2 7.6e-4; Q4_K_M 0.6B 0.30% and 1.25e-3; Q4_K_M 1.7B
     * 0.21% and 9.7e-4; cosine at least 0.9999992 and no argmax disagreement on any. The F16 file,
     * through the same GEMMs, measures 1.18e-3. Llama-3.2-1B Q8_0 measures the same on its batched
     * path (0.25%, 5.5e-4), so this is the prefill arithmetic, not Qwen3's layers.
     */
    // @formatter:on
    private GoldenCapture.Result assertQuantizedPrefill(Fixture fixture) throws Exception {
        return assertQuantizedPrefill(fixture, BATCH);
    }

    private GoldenCapture.Result assertQuantizedPrefill(Fixture fixture, int width)
            throws Exception {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return assertParityBatched(fixture, FP16, width);
        }
        return assertParityBatchedStagedWeights(fixture, Q8_0, FP16, width);
    }

    /**
     * The Q8_0 GEMMs every layer ran on Metal's SIMD-group matrices; nothing to check elsewhere.
     * {@code residualHalfTiles} says whether the projections into the model width staged FP16 too.
     */
    private static void assertQ8_0Selection(GoldenCapture.Result gpu, boolean residualHalfTiles) {
        if (!BatchPrefillGemmPolicy.simdgroup()) {
            return;
        }
        String residual =
                residualHalfTiles
                        ? "batchedGemmQ8WithResidualHalfTiles"
                        : "batchedGemmQ8WithResidual";
        assertEquals("layers", 28, gpu.prefillProjectionKernels.size());
        for (java.util.Map<String, String> layer : gpu.prefillProjectionKernels.values()) {
            assertEquals("batchedGemmQKVQ8HalfTiles", layer.get("batch_qkv"));
            assertEquals("batchedGemmRmsNormFFNGateUpQ8HalfTiles", layer.get("batch_ffn_gate_up"));
            assertEquals(residual, layer.get("batch_attn_out"));
            assertEquals(residual, layer.get("batch_ffn_down"));
        }
    }

    /**
     * On Metal at width 128 the q|k|v and gate|up GEMMs have enough workgroups to stage FP16, and
     * the projections into the model width, 32 workgroups, keep FP32 tiles: this case covers both.
     */
    @Test
    public void qwen3_0_6b_q8_0_batchedPrefillParity() throws Exception {
        assertQ8_0Selection(assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q8_0), false);
    }

    /**
     * Width 7: the same selection as at 128, but the 18 prefill positions take three chunks (7, 7
     * and 4), not one, before decode reads the cache they wrote.
     */
    @Test
    public void qwen3_0_6b_q8_0_batchedPrefillParityWidth7() throws Exception {
        assertQ8_0Selection(assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q8_0, 7), false);
    }

    /**
     * Width 512: all four GEMMs stage FP16, so all seven projection weights are rounded.
     *
     * <p>The narrow family is off. It would run this 18-token prompt on graphs 128 rows wide, so
     * the 512-wide graphs this case is about would not run; {@code NarrowChunkQwen3Q8AccelTest}
     * covers that family.
     */
    @Test
    public void qwen3_0_6b_q8_0_batchedPrefillParityWidth512() throws Exception {
        String property = BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY;
        String previous = System.getProperty(property);
        System.setProperty(property, "false");
        try {
            assertQ8_0Selection(assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q8_0, 512), true);
        } finally {
            if (previous == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, previous);
            }
        }
    }

    @Test
    public void qwen3_0_6b_f16_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_0_6B_F16, FP16, BATCH);
    }

    @Test
    public void qwen3_0_6b_q4_k_m_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q4_K_M);
    }

    @Test
    public void qwen3_1_7b_q4_k_m_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_1_7B_Q4_K_M);
    }

    @Test
    public void qwen3_0_6b_q4_0_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q4_0);
    }

    @Test
    public void qwen3_0_6b_q4_1_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q4_1);
    }

    @Test
    public void qwen3_0_6b_q4_k_s_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q4_K_S);
    }

    @Test
    public void qwen3_0_6b_q5_k_m_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q5_K_M);
    }

    @Test
    public void qwen3_0_6b_q5_k_s_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q5_K_S);
    }

    @Test
    public void qwen3_0_6b_q6_k_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_Q6_K);
    }

    @Test
    public void qwen3_0_6b_ud_q6_k_xl_batchedPrefillParity() throws Exception {
        assertQuantizedPrefill(Fixture.QWEN3_0_6B_UD_Q6_K_XL);
    }
}
