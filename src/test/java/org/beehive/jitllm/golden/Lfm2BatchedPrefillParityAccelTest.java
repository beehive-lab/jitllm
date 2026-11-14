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
 * <p>Measured against the FP16 bounds. M4 Pro (experts on FP16-tile SIMD-group GEMMs at 512):
 * LFM2-8B-A1B relative L2 at most 5.5e-4, LFM2.5-24B-A2B at most 2.6e-4. RTX 5090 Laptop (experts
 * on tensor cores over FP16 inputs): 8B 1.6e-4 at every width, 24B at most 2.0e-4. Argmax 63/63
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
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 7);
    }

    @Test
    public void lfm2_8b_a1b_batchedPrefillParityAt64() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 64);
    }

    @Test
    public void lfm2_8b_a1b_batchedPrefillParityAt512() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16, 512);
    }

    @Test
    public void lfm2_5_24b_a2b_batchedPrefillParityAt64() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16, 64);
    }

    @Test
    public void lfm2_5_24b_a2b_batchedPrefillParityAt512() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16, 512);
    }

    /** The batched prefill runs on Metal and CUDA; the provider refuses OpenCL by name. */
    private static void assumeBatchedPrefillHere() {
        org.junit.Assume.assumeTrue(
                "the lfm2moe batched prefill does not run on OpenCL",
                !org.beehive.jitllm.runtime.backend.BackendId.OPENCL.equals(
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend()));
    }
}
