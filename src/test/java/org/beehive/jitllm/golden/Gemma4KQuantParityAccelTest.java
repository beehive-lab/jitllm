package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's Q4_K_M file, kept as it is, against the CPU reference.
 *
 * <p>Its own class for the reason {@link Gemma4Q4_0ParityAccelTest} gives: this family's fixtures
 * cannot share one JVM's device memory.
 *
 * <p>The packed-integer projections are off, through Gemma 4's exact-comparison switch for the
 * layers and the shared one for the vocabulary projection (a Q4_K or Q6_K output projection is
 * packed by the shared logits layer), so the device computes in floating point what the host
 * reference computes and the {@link #Q8_0} bounds hold on every device; {@link
 * Qwen3CpuGpuParityAccelTest} says what the packed path computes instead.
 */
public class Gemma4KQuantParityAccelTest extends CpuGpuParity {

    static {
        System.setProperty("jitllm.gemma4.packedIntegerDot", "false");
        System.setProperty(PackedActivationTasks.DISABLE_PROPERTY, "false");
    }

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ4_K_MCpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_Q4_K_M, Q8_0);
    }
}
