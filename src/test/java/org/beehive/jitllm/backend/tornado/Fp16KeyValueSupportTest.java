package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
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
                new Combination(arch, weights, mode, backend, nvidia, tensorCores, true));
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

    /** The reported Q8_0 batched prefill, implemented: its writers and hand-off now carry FP16. */
    @Test
    public void llamaQ8IsSupportedInEveryModeAndQ4InSingleToken() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            assertEquals(
                    mode.toString(),
                    Optional.empty(),
                    check("qwen3", DataType.Q8_0, mode, BackendId.CUDA, true, true));
            assertEquals(
                    mode.toString(),
                    Optional.empty(),
                    check("llama", DataType.Q8_0, mode, BackendId.CUDA, true, true));
        }
        assertEquals(
                Optional.empty(),
                check("llama", DataType.Q4_0, ExecutionMode.STANDARD, BackendId.CUDA, true, false));
        assertTrue(
                check(
                                "llama",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true,
                                true)
                        .isPresent());
    }

    @Test
    public void pathsThatSilentlyKeptFp32AreRefused() {
        assertTrue(
                check(
                                "llama",
                                DataType.Q4_0,
                                ExecutionMode.PREFILL_DECODE,
                                BackendId.CUDA,
                                true,
                                true)
                        .isPresent());
        assertTrue(
                check("qwen3", DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, false, false)
                        .isPresent());
        assertTrue(
                check("qwen3", DataType.Q4_0, ExecutionMode.STANDARD, BackendId.CUDA, true, true)
                        .isPresent());
        assertTrue(
                check("gemma4", DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, true, true)
                        .isPresent());
        // The quantized gemma4 layers write and read FP16 in both of the family's modes, on CUDA.
        for (DataType weights : new DataType[] {DataType.Q8_0, DataType.Q4_0}) {
            for (ExecutionMode mode :
                    new ExecutionMode[] {
                        ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE
                    }) {
                assertTrue(
                        weights + " " + mode,
                        check("gemma4", weights, mode, BackendId.CUDA, true, true).isEmpty());
                assertTrue(
                        weights + " " + mode + " on OpenCL",
                        check("gemma4", weights, mode, BackendId.OPENCL, true, true).isPresent());
            }
        }
        // OpenCL is verified on NVIDIA-class devices only.
        assertTrue(
                Fp16KeyValueSupport.unsupported(
                                new Combination(
                                        "llama",
                                        DataType.F16,
                                        ExecutionMode.STANDARD,
                                        BackendId.OPENCL,
                                        false,
                                        false,
                                        false))
                        .isPresent());
        assertTrue(
                Fp16KeyValueSupport.unsupported(
                                new Combination(
                                        "mistral",
                                        DataType.Q8_0,
                                        ExecutionMode.STANDARD,
                                        BackendId.CUDA,
                                        false,
                                        true,
                                        false))
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

    @Test
    public void theRefusalNamesTheCombinationTheReasonAndTheWayOut() {
        var combination =
                new Combination(
                        "gemma4",
                        DataType.Q8_0,
                        ExecutionMode.STANDARD,
                        BackendId.CUDA,
                        true,
                        true,
                        true);
        String message = Fp16KeyValueSupport.refusal(combination, "the gemma4 layers keep FP32");
        assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
        assertTrue(message, message.contains("gemma4 / Q8_0 / STANDARD on cuda"));
        assertTrue(message, message.contains("the gemma4 layers keep FP32"));
        assertTrue(message, message.contains("--fp32-kv-cache"));
        assertTrue(message, message.contains("StorageOptions.fp32()"));
    }

    @Test
    public void mistralAndTheSingleTokenFamiliesAreSupported() {
        for (String arch :
                new String[] {"mistral", "qwen2", "deepseek-r1-distill-qwen", "phi3", "granite"}) {
            for (DataType weights : new DataType[] {DataType.F16, DataType.Q8_0}) {
                assertEquals(
                        arch,
                        Optional.empty(),
                        check(
                                arch,
                                weights,
                                ExecutionMode.STANDARD,
                                BackendId.CUDA,
                                !arch.equals("mistral"),
                                true));
            }
            assertTrue(
                    arch,
                    check(
                                    arch,
                                    DataType.F16,
                                    ExecutionMode.PREFILL_DECODE,
                                    BackendId.CUDA,
                                    true,
                                    true)
                            .isPresent());
        }
    }

    /** Without tensor cores batched prefill takes the scalar kernels, which have FP16 twins. */
    @Test
    public void scalarBatchedPrefillAndOpenClOnNvidiaAreSupported() {
        for (BackendId backend : new BackendId[] {BackendId.CUDA, BackendId.OPENCL}) {
            for (DataType weights : new DataType[] {DataType.F16, DataType.Q8_0}) {
                assertEquals(
                        Optional.empty(),
                        check(
                                "llama",
                                weights,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                backend,
                                true,
                                false));
                assertEquals(
                        Optional.empty(),
                        check("qwen3", weights, ExecutionMode.STANDARD, backend, true, false));
            }
        }
    }

    @Test
    public void qwen35StaysCudaOnly() {
        assertTrue(
                check(
                                "qwen35",
                                DataType.Q4_0,
                                ExecutionMode.STANDARD,
                                BackendId.OPENCL,
                                true,
                                false)
                        .isPresent());
    }
}
