package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen2.5-0.5B — Q/K/V biases on the MLX QKV GEMM's output — against the CPU reference with its
 * batched prefill on MLX; see {@link MlxPrefillParity}.
 */
public class Qwen2MlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void f16() throws Exception {
        assertMlxParity(Fixture.QWEN2_5_0_5B_F16, FP16, Weights.AS_LOADED);
    }
}
