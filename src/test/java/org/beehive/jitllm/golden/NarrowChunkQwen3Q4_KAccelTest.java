package org.beehive.jitllm.golden;

import java.nio.file.Path;

/** Qwen3-0.6B Q4_K_M: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen3Q4_KAccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.modelsRoot().resolve("Qwen3-0.6B-Q4_K_M.gguf");
    }
}
