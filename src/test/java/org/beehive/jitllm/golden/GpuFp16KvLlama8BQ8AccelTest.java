package org.beehive.jitllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU at 8B; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvLlama8BQ8AccelTest {

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
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(
                "meta-llama-3.1-8b-instruct.Q8_0.gguf", GpuFp16KvPrecision.Mode.BATCHED);
    }
}
