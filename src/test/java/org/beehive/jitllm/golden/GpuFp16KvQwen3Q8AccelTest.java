package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvQwen3Q8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.QWEN3_0_6B_Q8_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void prefillDecode() throws Exception {
        GpuFp16KvPrecision.check(Fixture.QWEN3_0_6B_Q8_0, GpuFp16KvPrecision.Mode.PREFILL_DECODE);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(Fixture.QWEN3_0_6B_Q8_0, GpuFp16KvPrecision.Mode.BATCHED);
    }
}
