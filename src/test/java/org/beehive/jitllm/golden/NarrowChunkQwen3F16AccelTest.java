package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Qwen3-0.6B F16: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen3F16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
    }
}
