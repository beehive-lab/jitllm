package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Phi-3-mini F16: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkPhi3F16AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.PHI3_MINI_4K_F16);
    }
}
