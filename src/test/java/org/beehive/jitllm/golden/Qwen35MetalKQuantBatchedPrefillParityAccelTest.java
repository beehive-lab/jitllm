package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;

/**
 * Qwen3.5-0.8B Q4_K_M on Metal against the CPU reference, with the prompt ingested in chunks of 8.
 *
 * <p>The file mixes Q4_K, Q5_K, Q6_K and Q8_0 projections and keeps its embeddings Q6_K. Each
 * projection runs by its own type: in the batched prefill the Q4_K and Q6_K weights take the
 * SIMD-group GEMMs the other families use, which stage the weights as FP16, so the rows are held to
 * {@link #FP16}.
 *
 * <p>Uses {@code Qwen3.5-0.8B-Q4_K_M.gguf} under the fixture directory, by name; skips when it is
 * absent or the backend is not Metal.
 */
public class Qwen35MetalKQuantBatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final String MODEL = "Qwen3.5-0.8B-Q4_K_M.gguf";

    @Test
    public void qwen35_0_8b_q4_k_m_batchedPrefillParityOnMetal() throws Exception {
        assertParityBatched(metalFile(), "Q4_K_M", FP16, 8);
    }

    private static Path metalFile() {
        Path file = GoldenFixture.modelsRoot().resolve(MODEL);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        BackendId backend = TornadoDevices.current().backend();
        assumeTrue(
                "this runs on the " + backend + " backend; the kernels are Metal's",
                BackendId.METAL.equals(backend));
        return file;
    }
}
