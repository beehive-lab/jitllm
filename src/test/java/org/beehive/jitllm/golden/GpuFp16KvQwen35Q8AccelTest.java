package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvQwen35Q8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check("Qwen3.5-0.8B-Q8_0.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void prefillDecode() throws Exception {
        GpuFp16KvPrecision.check("Qwen3.5-0.8B-Q8_0.gguf", GpuFp16KvPrecision.Mode.PREFILL_DECODE);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check("Qwen3.5-0.8B-Q8_0.gguf", GpuFp16KvPrecision.Mode.BATCHED);
    }
}
