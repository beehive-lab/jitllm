package org.beehive.jitllm.golden;

import java.nio.file.Path;

/** Devstral-Small-2-24B Q4_K_M: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkDevstralQ4_KAccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.modelsRoot().resolve("Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf");
    }
}
