package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * FP16 against FP32 key/value cache on the GPU for Gemma 4; see {@link GpuFp16KvPrecision}. The
 * family has no sequential prefill/decode plan, so its two modes are the single-token and the
 * batched one.
 */
public class GpuFp16KvGemma4Q8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.GEMMA_4_E2B_Q8_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(Fixture.GEMMA_4_E2B_Q8_0, GpuFp16KvPrecision.Mode.BATCHED);
    }
}
