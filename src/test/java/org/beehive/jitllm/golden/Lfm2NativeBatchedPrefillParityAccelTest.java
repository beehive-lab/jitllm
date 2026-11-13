package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * The {@code lfm2moe} batched prefill with native libraries, CUDA's default: the attention and
 * convolution projections on cuBLAS and the experts on tensor cores, every GEMM over FP16 operands.
 * Against the CPU reference with an FP32 key/value cache on both sides.
 *
 * <p>Held to the FP16 cache's envelope ({@link Lfm2Fp16KvParityAccelTest#FP16_KEY_VALUE}), for the
 * same reason: the router's sigmoid scores have near-ties, the operands' FP16 rounding decides
 * which side of one a few prompt tokens fall on, and their keys, values and convolution windows
 * then carry another expert's output into later rows. Either GEMM alone stays within the FP16
 * bounds ({@code Lfm2BatchedPrefillParityAccelTest}); together they do not.
 *
 * <p>Measured on an RTX 5090 Laptop: LFM2.5-24B-A2B relative L2 at most 3.4e-4; LFM2-8B-A1B 4.0e-3
 * and 5.8e-2, cosine 0.9983, argmax 0/63 at every width.
 */
// @formatter:on
public class Lfm2NativeBatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void lfm2_8b_a1b_nativeBatchedPrefillParityAt64() throws Exception {
        assertNative(Fixture.LFM2_8B_A1B_Q4_K_XL, 64);
    }

    @Test
    public void lfm2_8b_a1b_nativeBatchedPrefillParityAt512() throws Exception {
        assertNative(Fixture.LFM2_8B_A1B_Q4_K_XL, 512);
    }

    @Test
    public void lfm2_5_24b_a2b_nativeBatchedPrefillParityAt64() throws Exception {
        assertNative(Fixture.LFM2_5_24B_A2B_Q4_K_XL, 64);
    }

    @Test
    public void lfm2_5_24b_a2b_nativeBatchedPrefillParityAt512() throws Exception {
        assertNative(Fixture.LFM2_5_24B_A2B_Q4_K_XL, 512);
    }

    private void assertNative(Fixture fixture, int width) throws Exception {
        assumeTrue(
                "the native path needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        String previous = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        try {
            assertParityBatched(fixture, Lfm2Fp16KvParityAccelTest.FP16_KEY_VALUE, width);
        } finally {
            if (previous == null) {
                System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
            } else {
                System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, previous);
            }
        }
    }
}
