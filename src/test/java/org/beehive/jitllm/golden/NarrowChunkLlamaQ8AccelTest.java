package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Llama-3.2-1B Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkLlamaQ8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
    }
}
