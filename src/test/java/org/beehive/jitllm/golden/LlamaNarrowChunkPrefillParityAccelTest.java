package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Llama 3.2 1B F16's tensor-core batched prefill at a chunk narrower than one 128-row tile, against
 * the CPU reference with {@link #FP16}: the prompt runs as one chunk of 16 and a partial chunk.
 * With native libraries cuBLAS multiplies the chunk's tokens; without them the JIT GEMMs split the
 * padded rows ({@code LlamaF16NarrowPrefillDispatchAccelTest} has the dispatch).
 *
 * <p>Its own class, as {@link LlamaBatchedPrefillParityAccelTest} explains, so that the whole-tile
 * cases there do not load this fixture twice more in one JVM.
 */
public class LlamaNarrowChunkPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    private static final int CHUNK = 16;

    /** The native-library setting the run started with, put back after each case. */
    private String nativeLibraries;

    @Before
    public void rememberNativeLibraries() {
        nativeLibraries = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
    }

    @After
    public void restoreNativeLibraries() {
        if (nativeLibraries == null) {
            System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        } else {
            System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, nativeLibraries);
        }
    }

    @Test
    public void nativeLibraries() throws Exception {
        assumeTrue("tensor-core prefill", TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        assertParityBatched(Fixture.LLAMA_3_2_1B_F16, FP16, CHUNK);
    }

    @Test
    public void jitKernels() throws Exception {
        assumeTrue("tensor-core prefill", TensorCoreSupport.isTensorCoreCapableBackend());
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "false");
        assertParityBatched(Fixture.LLAMA_3_2_1B_F16, FP16, CHUNK);
    }
}
