package org.beehive.jllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvDeepSeekQwen2Q8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(
                "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }
}
