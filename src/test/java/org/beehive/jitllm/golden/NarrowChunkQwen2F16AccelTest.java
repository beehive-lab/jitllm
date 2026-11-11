package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Qwen2.5-0.5B F16: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen2F16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.QWEN2_5_0_5B_F16);
    }
}
