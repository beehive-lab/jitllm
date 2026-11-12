package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4 E2B Q8_0 against the CPU reference with its batched prefill projections on MLX; see
 * {@link MlxPrefillParity}.
 *
 * <p>MLX takes the query/key/value, output, gate/up and down projections; the attention, the norms
 * and the per-layer-embedding GEMMs stay on the Metal kernels. MLX's GEMMs read FP16 weights, from
 * the copies or the affine repack. The Metal SIMD-group prefill reads the Q8_0 weights exactly. So
 * {@link CpuGpuParity#GEMMA4_Q8_0_BATCHED}, the envelope that prefill is held to, applies against
 * the host with the projection weights rounded to FP16. {@link CpuGpuParity#FP16}, as for every
 * other family's MLX prefill, applies against the exact host.
 */
public class Gemma4Q8_0MlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    /** The FP16 products of the MLX projections, converted back to FP32 for the Metal kernels. */
    @Override
    String[] mlxTasks() {
        return new String[] {".mlx_qkv_float", ".mlx_wo_float", ".mlx_w2_float"};
    }

    @Test
    public void q8_0Fp16Copies() throws Exception {
        assertMlxParityStagedWeights(
                Fixture.GEMMA_4_E2B_Q8_0, GEMMA4_Q8_0_BATCHED, FP16, Weights.FP16_COPIES);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParityStagedWeights(
                Fixture.GEMMA_4_E2B_Q8_0, GEMMA4_Q8_0_BATCHED, FP16, Weights.AFFINE);
    }
}
