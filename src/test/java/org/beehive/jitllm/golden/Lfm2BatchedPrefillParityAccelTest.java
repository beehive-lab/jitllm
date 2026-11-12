package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * The {@code lfm2moe} batched prefill against the CPU reference, at three chunk widths: 7 (every
 * chunk partial, the expert tiles a few rows each), 64 and 512 (the default).
 *
 * <p>The prompt's projections are SIMD-group GEMMs, which at the wider widths stage the weights as
 * FP16, and the experts are grouped GEMMs over the chunk's assignments; decode then continues on
 * the single-token graphs from the key/value store and the convolution windows the chunks left.
 *
 * <p>Measured on an M4 Pro against the FP16 bounds: LFM2-8B-A1B relative L2 3.0e-4, 2.7e-4 and
 * 7.6e-4 at widths 7, 64 and 512; LFM2.5-24B-A2B 6.0e-5 and 1.6e-4 at 64 and 512; argmax 63/63
 * throughout.
 */
// @formatter:on
public class Lfm2BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void lfm2_8b_a1b_batchedPrefillParityAt7() throws Exception {
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 7);
    }

    @Test
    public void lfm2_8b_a1b_batchedPrefillParityAt64() throws Exception {
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 64);
    }

    @Test
    public void lfm2_8b_a1b_batchedPrefillParityAt512() throws Exception {
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 512);
    }

    @Test
    public void lfm2_5_24b_a2b_batchedPrefillParityAt64() throws Exception {
        assertParityBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16, 64);
    }

    @Test
    public void lfm2_5_24b_a2b_batchedPrefillParityAt512() throws Exception {
        assertParityBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16, 512);
    }
}
