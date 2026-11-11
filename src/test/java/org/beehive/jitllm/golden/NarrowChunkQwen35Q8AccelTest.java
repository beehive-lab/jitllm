package org.beehive.jitllm.golden;

import java.nio.file.Path;

/** Qwen3.5-0.8B Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen35Q8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-Q8_0.gguf");
    }
}
