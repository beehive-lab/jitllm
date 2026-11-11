package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Qwen3-0.6B Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen3Q8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.QWEN3_0_6B_Q8_0);
    }
}
