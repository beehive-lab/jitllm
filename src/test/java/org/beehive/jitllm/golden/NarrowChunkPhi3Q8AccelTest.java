package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Phi-3-mini Q8_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkPhi3Q8AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.PHI3_MINI_4K_Q8_0);
    }
}
