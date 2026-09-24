package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Where --with-native-libraries has an implementation, as a pure function of the resolved facts.
 */
public class NativeLibrarySupportTest {

    private static Optional<String> check(
            String arch, DataType weights, ExecutionMode mode, BackendId backend, boolean tc) {
        return NativeLibrarySupport.unsupported(
                new Combination(arch, weights, mode, backend, true, tc, true));
    }

    @Test
    public void qwen3F16BatchedPrefillOnTensorCoresIsTheImplementedPath() {
        assertEquals(
                Optional.empty(),
                check(
                        "qwen3",
                        DataType.F16,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
    }

    @Test
    public void gemma4QuantizedBatchedPrefillOnTensorCoresIsImplemented() {
        for (DataType weights : new DataType[] {DataType.Q8_0, DataType.Q4_0}) {
            assertEquals(
                    Optional.empty(),
                    check(
                            "gemma4",
                            weights,
                            ExecutionMode.BATCH_PREFILL_DECODE,
                            BackendId.CUDA,
                            true));
        }
        assertTrue(
                check(
                                "gemma4",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true)
                        .isPresent());
        assertTrue(
                check("gemma4", DataType.Q8_0, ExecutionMode.STANDARD, BackendId.CUDA, true)
                        .isPresent());
    }

    @Test
    public void everythingElseIsRefused() {
        assertTrue(
                check(
                                "llama",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true)
                        .isPresent());
        assertTrue(
                check(
                                "qwen3",
                                DataType.Q8_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true)
                        .isPresent());
        assertTrue(
                check("qwen3", DataType.F16, ExecutionMode.STANDARD, BackendId.CUDA, true)
                        .isPresent());
        assertTrue(
                check("qwen3", DataType.F16, ExecutionMode.PREFILL_DECODE, BackendId.CUDA, true)
                        .isPresent());
        assertTrue(
                check(
                                "qwen3",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                false)
                        .isPresent());
        assertTrue(
                check(
                                "qwen3",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.OPENCL,
                                false)
                        .isPresent());
        assertTrue(
                check("phi3", DataType.F16, ExecutionMode.STANDARD, BackendId.CPU, false)
                        .isPresent());
    }

    @Test
    public void theRefusalNamesTheWayOut() {
        String message =
                NativeLibrarySupport.refusal(
                        new Combination(
                                "llama",
                                DataType.F16,
                                ExecutionMode.STANDARD,
                                BackendId.CUDA,
                                true,
                                true,
                                true),
                        "no native-library path");
        assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
        assertTrue(message, message.contains("llama / F16 / STANDARD on cuda"));
        assertTrue(message, message.contains("Drop --with-native-libraries"));
    }
}
