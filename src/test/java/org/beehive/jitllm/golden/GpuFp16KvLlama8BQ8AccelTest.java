package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU at 8B; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvLlama8BQ8AccelTest {

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(
                "meta-llama-3.1-8b-instruct.Q8_0.gguf", GpuFp16KvPrecision.Mode.BATCHED);
    }
}
