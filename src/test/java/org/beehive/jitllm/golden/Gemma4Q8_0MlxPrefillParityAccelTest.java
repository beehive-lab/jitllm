package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4 E2B Q8_0 against the CPU reference with its batched prefill projections on MLX; see
 * {@link MlxPrefillParity}.
 *
 * <p>Held to {@link CpuGpuParity#GEMMA4_Q8_0_BATCHED}, the envelope the Metal SIMD-group batched
 * prefill is held to: MLX takes the query/key/value, output, gate/up and down projections, and the
 * attention, the norms and the per-layer-embedding GEMMs stay on the Metal kernels.
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
        assertMlxParity(Fixture.GEMMA_4_E2B_Q8_0, GEMMA4_Q8_0_BATCHED, Weights.FP16_COPIES);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParity(Fixture.GEMMA_4_E2B_Q8_0, GEMMA4_Q8_0_BATCHED, Weights.AFFINE);
    }
}
