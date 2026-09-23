package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/** The FP16 key/value support matrix, as a pure function of the resolved configuration. */
public class Fp16KeyValueSupportTest {

    private static Optional<String> check(
            String arch,
            DataType weights,
            ExecutionMode mode,
            BackendId backend,
            boolean nvidia,
            boolean tensorCores) {
        return Fp16KeyValueSupport.unsupported(
                new Combination(arch, weights, mode, backend, nvidia, tensorCores));
    }

    @Test
    public void supportedConfigurations() {
        for (String arch : new String[] {"llama", "qwen3"}) {
            assertEquals(
                    Optional.empty(),
                    check(arch, DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, true, false));
            assertEquals(
                    Optional.empty(),
                    check(
                            arch,
                            DataType.F16,
                            ExecutionMode.BATCH_PREFILL_DECODE,
                            BackendId.CUDA,
                            true,
                            true));
        }
        for (ExecutionMode mode : ExecutionMode.values()) {
            assertEquals(
                    Optional.empty(),
                    check("qwen35", DataType.Q4_0, mode, BackendId.CUDA, true, true));
            // The host cache is FloatTensor for every family.
            assertEquals(
                    Optional.empty(),
                    check("phi3", DataType.Q8_0, mode, BackendId.CPU, false, false));
        }
    }

    @Test
    public void theReportedQ8BatchPrefillBugIsRefused() {
        Optional<String> reason =
                check(
                        "llama",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true,
                        true);
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("Q8_0"));
    }

    @Test
    public void pathsThatSilentlyKeptFp32AreRefused() {
        assertTrue(
                check(
                                "llama",
                                DataType.F16,
                                ExecutionMode.PREFILL_DECODE,
                                BackendId.CUDA,
                                true,
                                true)
                        .isPresent());
        assertTrue(
                check(
                                "llama",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true,
                                false)
                        .isPresent());
        assertTrue(
                check("qwen3", DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, false, false)
                        .isPresent());
        assertTrue(
                check("llama", DataType.Q4_0, ExecutionMode.STANDARD, BackendId.CUDA, true, true)
                        .isPresent());
        assertTrue(
                check("mistral", DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, false, true)
                        .isPresent());
        assertTrue(
                check("llama", DataType.F16, ExecutionMode.STANDARD, BackendId.OPENCL, true, false)
                        .isPresent());
        assertTrue(
                check(
                                "qwen35",
                                DataType.Q4_0,
                                ExecutionMode.STANDARD,
                                BackendId.METAL,
                                false,
                                false)
                        .isPresent());
    }
}
