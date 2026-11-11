package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Llama-3.2-1B F16: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkLlamaF16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
    }
}
