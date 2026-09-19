package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B ingested in chunks of 7, against the CPU reference.
 *
 * <p>A width that divides nothing. The prompt is not a multiple of it, so the last chunk is
 * partially active and the padding rows have to contribute nothing.
 *
 * <p>One width per class, and therefore per JVM. This fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process exhausts the card. The widths are separated rather than looped.
 */
public class Qwen35BatchedPrefillWidth7ParityAccelTest extends CpuGpuParity {

    static {
        // The scalar batched path, which the tensor-core default would otherwise replace: this
        // class is the scalar kernels' coverage; the Qwen35Mma* classes cover the tensor cores.
        System.setProperty("jllm.qwen35.tensorCores", "false");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt7() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 7);
    }
}
