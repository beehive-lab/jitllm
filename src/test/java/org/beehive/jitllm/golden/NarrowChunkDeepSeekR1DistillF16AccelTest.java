package org.beehive.jitllm.golden;

import java.nio.file.Path;

/**
 * DeepSeek-R1-Distill-Qwen-1.5B F16: the narrow prefill family ({@link NarrowChunkPrefillParity}).
 */
public class NarrowChunkDeepSeekR1DistillF16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.modelsRoot().resolve("DeepSeek-R1-Distill-Qwen-1.5B-F16.gguf");
    }
}
