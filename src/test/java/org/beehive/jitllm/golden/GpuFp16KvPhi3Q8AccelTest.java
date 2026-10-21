package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvPhi3Q8AccelTest {

    // @formatter:off
    /**
     * The packed-integer decode off, through its exact-comparison switch, so this measures what the
     * half-precision cache does and not what an eight-bit activation quantizer does to it; see
     * {@link GpuFp16KvGemma4Q4AccelTest}.
     */
    // @formatter:on
    static {
        System.setProperty(
                org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks.DISABLE_PROPERTY,
                "false");
    }

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.PHI3_MINI_4K_Q8_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.checkBatchedOnMetal(Fixture.PHI3_MINI_4K_Q8_0);
    }
}
