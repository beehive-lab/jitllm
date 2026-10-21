package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Phi-3-mini — fused QKV and gate/up, multiplied as they are — against the CPU reference with its
 * batched prefill on MLX; see {@link MlxPrefillParity}.
 *
 * <p>Its Q8_0 projections are not exercised as FP16 copies: at 7.2 GB they exceed the backend's
 * resident allowance, so the residency never selects them, and decoding them on the host exceeds
 * the test JVM's 6 GiB of direct memory.
 */
public class Phi3MlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void f16() throws Exception {
        assertMlxParity(Fixture.PHI3_MINI_4K_F16, FP16, Weights.AS_LOADED);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParity(Fixture.PHI3_MINI_4K_Q8_0, FP16, Weights.AFFINE);
    }
}
