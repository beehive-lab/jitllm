package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;

// @formatter:off
/**
 * Qwen3.5-0.8B Q8_0 with the prompt ingested in chunks on Metal, against the CPU reference.
 *
 * <p>On Metal the batched Q8_0 projections run on the SIMD-group GEMMs: FP32 activations, the Q8_0
 * blocks dequantized to float as they are staged, FP32 accumulation. That is the host's arithmetic
 * up to summation order, so the rows are held to the same {@link #Q8_0} bounds as a single-token
 * Q8_0 path, with no packed-activation allowance. Width 8 makes the prompt several chunks, each
 * ending inside a 64-row tile; the whole-tile write-back is covered bit for bit by the kernel's own
 * test.
 *
 * <p>Uses {@code Qwen3.5-0.8B-Q8_0.gguf} under the fixture directory, by name; skips when it is
 * absent or the backend is not Metal.
 */
// @formatter:on
public class Qwen35MetalQ8BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final String MODEL = "Qwen3.5-0.8B-Q8_0.gguf";

    @Test
    public void qwen35_0_8b_q8_0_batchedPrefillParityOnMetal() throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve(MODEL);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        BackendId backend = TornadoDevices.current().backend();
        assumeTrue(
                "this runs on the " + backend + " backend; the kernels are Metal's",
                BackendId.METAL.equals(backend));
        assertParityBatched(file, "Q8_0", Q8_0, 8);
    }
}
