package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.After;
import org.junit.Test;

/**
 * Qwen3.5-0.8B F16 on CUDA against the CPU reference, on cuBLAS and on the JIT kernels.
 *
 * <p>Example: at width 128 the state allocates the int8 scratch and no F16 projection reads it,
 * while the 63-row fixture prompt fills only part of the chunk. Width 7 allocates no int8 scratch.
 *
 * <p>Uses the file under the fixture directory, by name; skips when it is absent or the backend is
 * not CUDA.
 */
public class Qwen35F16BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
    }

    @After
    public void restoreNativeLibraries() {
        System.clearProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
    }

    @Test
    public void jitKernelsAtWidth128() throws Exception {
        assertF16Parity(128, false);
    }

    @Test
    public void nativeLibrariesAtWidth128() throws Exception {
        assertF16Parity(128, true);
    }

    @Test
    public void nativeLibrariesAtWidth7() throws Exception {
        assertF16Parity(7, true);
    }

    private void assertF16Parity(int width, boolean nativeLibraries) throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-F16.gguf");
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        BackendId backend = TornadoDevices.current().backend();
        assumeTrue("this runs on the " + backend + " backend", BackendId.CUDA.equals(backend));
        System.setProperty(
                ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, Boolean.toString(nativeLibraries));
        GoldenCapture.Result gpu =
                assertParityBatched(
                        file, "F16 width " + width + " native=" + nativeLibraries, FP16, width);
        // cuBLAS adds into the residual stream itself; the JIT GEMM stores, then adds in a pass.
        assertEquals(
                "which of cuBLAS and the JIT kernels ran ffn_down",
                !nativeLibraries,
                gpu.gridScheduler.keySet().stream()
                        .anyMatch(key -> key.endsWith(".ffn_down_residual")));
    }
}
