package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Llama-3.2-1B against the CPU reference with its batched prefill on MLX; see {@link
 * MlxPrefillParity}.
 */
public class LlamaMlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void f16() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_F16, FP16, Weights.AS_LOADED);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_Q8_0, FP16, Weights.AFFINE);
    }

    @Test
    public void q8_0Fp16Copies() throws Exception {
        assertMlxParity(Fixture.LLAMA_3_2_1B_Q8_0, FP16, Weights.FP16_COPIES);
    }
}
