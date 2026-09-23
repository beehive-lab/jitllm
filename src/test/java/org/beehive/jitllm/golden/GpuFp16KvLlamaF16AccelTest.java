package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvLlamaF16AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.LLAMA_3_2_1B_F16, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(Fixture.LLAMA_3_2_1B_F16, GpuFp16KvPrecision.Mode.BATCHED);
    }

    @Test
    public void prefillDecode() throws Exception {
        GpuFp16KvPrecision.check(Fixture.LLAMA_3_2_1B_F16, GpuFp16KvPrecision.Mode.PREFILL_DECODE);
    }
}
