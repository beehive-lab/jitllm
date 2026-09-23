package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU at 8B; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvQwen3_8BQ8AccelTest {

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check("Qwen3-8B-Q8_0.gguf", GpuFp16KvPrecision.Mode.BATCHED);
    }
}
