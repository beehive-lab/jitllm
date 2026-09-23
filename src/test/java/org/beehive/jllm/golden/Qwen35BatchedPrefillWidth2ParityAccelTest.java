package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B ingested in chunks of 2, against the CPU reference.
 *
 * <p>A pair of tokens per chunk: the narrowest width at which a chunk is a chunk, and the one where
 * a scan that ignored its second row would still look plausible.
 *
 * <p>One width per class, and therefore per JVM. This fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process exhausts the card. The widths are separated rather than looped.
 */
public class Qwen35BatchedPrefillWidth2ParityAccelTest extends CpuGpuParity {

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
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt2() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 2);
    }
}
