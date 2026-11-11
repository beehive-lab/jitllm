package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Granite-3.2-2B Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkGraniteQ8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.GRANITE_3_2_2B_Q8_0);
    }
}
