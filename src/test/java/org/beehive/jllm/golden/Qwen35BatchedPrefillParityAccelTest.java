package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B with the prompt ingested in chunks, against the CPU reference.
 *
 * <p>Its own class, and therefore its own JVM: this fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so two of
 * these in one process exhaust the card partway through the second.
 *
 * <p>Same bounds as the single-token path. This family's batched projections accumulate in FP32
 * like its single-token ones — it does not use the shared Q8_0 tensor-core GEMM, whose FP16
 * accumulation is why batched prefill cannot meet these bounds for the families that do.
 */
public class Qwen35BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    static {
        // The scalar batched path, which the tensor-core default would otherwise replace: this
        // class is the scalar kernels' coverage; the Qwen35Mma* classes cover the tensor cores.
        System.setProperty("jllm.qwen35.tensorCores", "false");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 32);
    }
}
