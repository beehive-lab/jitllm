package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}.
 *
 * <p>Wider bounds than the dense families: the routed experts make this model's logits move more
 * under the cache's rounding. On an M4 Pro, prompts of 144 to 296 tokens measured cosine 0.9964 to
 * 0.9992 and relative L2 0.041 to 0.085, top-1 agreement 100%; the CPU's FP16 cache, at 296 tokens,
 * the same 0.9992 and 0.041. On an RTX 5090 Laptop, 296 tokens: single-token 0.9975 and 0.070, the
 * tensor-core batched prefill 0.9944 and 0.107.
 */
public class GpuFp16KvQwen2MoeQ4AccelTest {

    private static final double MIN_COSINE = 0.99;
    private static final double MAX_RELATIVE_L2 = 0.15;

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(
                Fixture.QWEN1_5_MOE_A2_7B_Q4_0,
                GpuFp16KvPrecision.Mode.SINGLE_TOKEN,
                MIN_COSINE,
                MAX_RELATIVE_L2);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.checkBatchedOnMetal(
                Fixture.QWEN1_5_MOE_A2_7B_Q4_0, MIN_COSINE, MAX_RELATIVE_L2);
    }
}
