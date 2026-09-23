package org.beehive.jitllm.golden;

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

    @Test
    public void llama3_2_1b_q8_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0, BATCH);
    }

    @Test
    public void llama3_2_1b_f16_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.LLAMA_3_2_1B_F16, FP16, BATCH);
    }
}
