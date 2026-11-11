package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Gemma-4-E2B Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkGemma4Q8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.GEMMA_4_E2B_Q8_0);
    }
}
