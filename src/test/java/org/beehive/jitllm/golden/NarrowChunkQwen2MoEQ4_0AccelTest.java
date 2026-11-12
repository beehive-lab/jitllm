package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;

/** Qwen1.5-MoE-A2.7B Q4_0: the narrow prefill family ({@link NarrowChunkPrefillParity}). */
public class NarrowChunkQwen2MoEQ4_0AccelTest extends NarrowChunkPrefillParity {

    @Override
    protected Path model() {
        return GoldenFixture.locate(Fixture.QWEN1_5_MOE_A2_7B_Q4_0);
    }
}
