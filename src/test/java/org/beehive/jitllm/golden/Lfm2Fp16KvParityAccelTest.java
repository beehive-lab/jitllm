package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * {@link Lfm2CpuGpuParityAccelTest} with the FP16 key/value cache, the default, on both sides.
 *
 * <p>Its own envelope, wider than the Q8_0 bounds, because the sigmoid router turns FP16 rounding
 * into a different expert where two scores nearly tie. The two sides round the same keys and
 * values; their FP32 noise, about 5e-5, decides which side of a near-tie a few tokens fall on, and
 * the token's keys and values then carry the other expert's output into later rows. Example, on the
 * 63-row trace of LFM2-8B-A1B on an M4 Pro: rows 0 to 2 agree to 5e-5, row 3 differs by 2.8e-2
 * after the host's FP16 run takes the other expert, and the host's own FP16 run differs from its
 * FP32 run at the same rows by the same amounts. The device's FP16 run stays within 3e-3 of the
 * host's FP32 run.
 *
 * <p>Measured on that trace, against the bounds: largest absolute difference 0.223 of the reference
 * RMS against 0.45, relative L2 0.0577 against 0.12, cosine 0.99834 against 0.996, elementwise
 * 21.7% against 45%, argmax 0/63, top-5 4.968/5.
 */
// @formatter:on
public class Lfm2Fp16KvParityAccelTest extends CpuGpuParity {

    static final Bounds FP16_KEY_VALUE = new Bounds(1.7e-4, 1e-2, 0.45, 0.12, 0.996, 0.45, 0.5);

    @Test
    public void lfm2_8b_a1b_cpuGpuParity() throws Exception {
        assertParity(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16_KEY_VALUE);
    }

    @Test
    public void lfm2_8b_a1b_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16_KEY_VALUE);
    }

    /** The batched prefill at its default width, over the same FP16 cache. */
    @Test
    public void lfm2_8b_a1b_batchedPrefillParity() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, FP16_KEY_VALUE, 512);
    }

    @Test
    public void lfm2_5_24b_a2b_batchedPrefillParity() throws Exception {
        assumeBatchedPrefillHere();
        assertParityBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16_KEY_VALUE, 512);
    }

    /**
     * The batched prefill with native libraries, CUDA's default: over the FP16 cache its attention
     * runs on tensor cores ({@code TensorCoreAttentionKernels.flashAttentionHead64}).
     */
    @Test
    public void lfm2_8b_a1b_nativeBatchedPrefillParity() throws Exception {
        assertNativeBatched(Fixture.LFM2_8B_A1B_Q4_K_XL, 512);
    }

    @Test
    public void lfm2_5_24b_a2b_nativeBatchedPrefillParityAt64() throws Exception {
        assertNativeBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, 64);
    }

    @Test
    public void lfm2_5_24b_a2b_nativeBatchedPrefillParity() throws Exception {
        assertNativeBatched(Fixture.LFM2_5_24B_A2B_Q4_K_XL, 512);
    }

    private void assertNativeBatched(Fixture fixture, int width) throws Exception {
        org.junit.Assume.assumeTrue(
                "the native path needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        org.junit.Assume.assumeTrue(
                "cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        String property = ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY;
        String previous = System.getProperty(property);
        System.setProperty(property, "true");
        try {
            assertParityBatched(fixture, FP16_KEY_VALUE, width);
        } finally {
            if (previous == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, previous);
            }
        }
    }

    @Test
    public void lfm2_5_24b_a2b_cpuGpuParity() throws Exception {
        assertParity(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16_KEY_VALUE);
    }

    @Test
    public void lfm2_5_24b_a2b_prefillDecodeParity() throws Exception {
        assertParityPrefillDecode(Fixture.LFM2_5_24B_A2B_Q4_K_XL, FP16_KEY_VALUE);
    }

    /** The batched prefill runs on Metal and CUDA; the provider refuses OpenCL by name. */
    private static void assumeBatchedPrefillHere() {
        org.junit.Assume.assumeTrue(
                "the lfm2moe batched prefill does not run on OpenCL",
                !org.beehive.jitllm.runtime.backend.BackendId.OPENCL.equals(
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend()));
    }
}
