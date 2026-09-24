package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * Gemma 4's Q4_0 logits against the CPU reference with the prompt ingested through the batched
 * prefill and its projections on cuBLAS ({@code --with-native-libraries}), then decoded.
 *
 * <p>The envelope is the JIT batched path's, {@link CpuGpuParity#GEMMA4_Q4_0_BATCHED}: the native
 * GEMMs multiply the same FP16 weights and activations with FP32 accumulation, in another order.
 *
 * <p>Its own class, and therefore its own JVM, like the JIT batched parity test.
 */
public class Gemma4Q4_0NativePrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ40NativePrefillParity() throws Exception {
        assumeTrue(
                "the native path needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        String previous = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        try {
            assertParityBatched(Fixture.GEMMA_4_E2B_Q4_0, GEMMA4_Q4_0_BATCHED, 7);
        } finally {
            if (previous == null) {
                System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
            } else {
                System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, previous);
            }
        }
    }
}
