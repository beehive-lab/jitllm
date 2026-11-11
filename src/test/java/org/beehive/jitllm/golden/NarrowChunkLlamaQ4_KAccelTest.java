package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Llama-3.2-1B Q4_K_M: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkLlamaQ4_KAccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.LLAMA_3_2_1B_UNSLOTH_Q4_K_M);
    }
}
