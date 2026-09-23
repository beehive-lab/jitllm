package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B's logits against the CPU reference. See {@link CpuGpuParity}.
 *
 * <p>The only fixture here that is genuinely mixed: Q4_0 projections and token embeddings, Q4_1
 * down projections on the first eight blocks, Q5_K recurrent outputs, a Q6_K vocabulary projection
 * and F32 norms and SSM parameters, none of them materialized. Every one of those is decoded by a
 * kernel selected from that tensor's own representation, and this is the gate that says the
 * selection is right: reading one block layout as another produces weights of plausible magnitude
 * and text that is fluent and wrong.
 *
 * <p>It is also the only fixture whose stack is three quarters recurrent, so it is the gate on the
 * delta-net state carrying correctly from one token to the next. The comparison is teacher-forced
 * along the CPU's tokens, which keeps both the key/value cache and the recurrent state identical at
 * every compared position.
 *
 * <p>Slow by the standards of the other families — the CPU reference runs at about a token a second
 * on a 27B — and worth it: nothing cheaper can see a defect that moves the whole GPU.
 */
public class Qwen35CpuGpuParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void qwen3_8_27b_q4_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_8_27B_Q4_0, Q8_0_FULLY_PACKED);
    }
}
