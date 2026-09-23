package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's logits against the CPU reference, one case per representation. See {@link
 * CpuGpuParity}.
 *
 * <p>This family reached an accelerator before it reached this gate: it was verified on the CPU and
 * on Apple-Silicon OpenCL, and never on CUDA. A GPU-versus-GPU comparison cannot see a defect that
 * moves the whole GPU, so the CPU is the only reference that can.
 */
public class Gemma4CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ8_0CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_Q8_0, Q8_0);
    }

    /** The file is BF16; the plan it selects is the FP16 one, so these are the FP16 bounds. */
    @Test
    public void gemma4E2bBf16CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_BF16, FP16);
    }

    // The Q4_0 fixture is a separate class, not a third case here: this model is 9.3 GB in BF16
    // and 5.0 GB in Q8_0, and device memory a closed session frees returns to TornadoVM's buffer
    // provider rather than to the driver, so a third load in one JVM exhausts a 24 GB card. See
    // Gemma4Q4_0ParityAccelTest.
}
