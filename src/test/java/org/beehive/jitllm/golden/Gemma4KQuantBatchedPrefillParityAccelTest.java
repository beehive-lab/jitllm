package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's Q4_K_M file, kept as it is, against the CPU reference with the prompt ingested through
 * the batched prefill graphs, then decoded.
 *
 * <p>The prompt rows go through FP16-operand GEMMs over the weights decoded per type (tensor cores
 * on CUDA, the {@code batchedGemmHalf…Strided} SIMD-group kernels on Metal), which is the
 * arithmetic {@link CpuGpuParity#GEMMA4_Q8_0_BATCHED} was measured for; the width of seven is
 * {@link Gemma4Q8_0BatchedPrefillParityAccelTest}'s. The packed-integer projections are off, as in
 * {@link Gemma4KQuantParityAccelTest}, so the decoded rows compute in floating point.
 *
 * <p>Its own class, and therefore its own JVM: device memory a closed session frees returns to
 * TornadoVM's buffer provider rather than to the driver.
 */
public class Gemma4KQuantBatchedPrefillParityAccelTest extends CpuGpuParity {

    static {
        System.setProperty("jitllm.gemma4.packedIntegerDot", "false");
    }

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ4_K_MBatchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.GEMMA_4_E2B_Q4_K_M, GEMMA4_Q8_0_BATCHED, 7);
    }
}
