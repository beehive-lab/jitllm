package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvDeepSeekQwen2Q8AccelTest {

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
        GpuFp16KvPrecision.check(
                "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.checkBatchedOnMetal("DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf");
    }
}
