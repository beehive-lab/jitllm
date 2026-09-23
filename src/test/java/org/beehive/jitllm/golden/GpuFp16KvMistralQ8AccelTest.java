package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvMistralQ8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(
                "Mistral-7B-Instruct-v0.3.Q8_0.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }
}
