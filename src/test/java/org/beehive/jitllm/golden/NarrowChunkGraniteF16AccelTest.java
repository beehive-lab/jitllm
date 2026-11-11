package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Granite-3.2-2B F16: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkGraniteF16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.GRANITE_3_2_2B_F16);
    }
}
