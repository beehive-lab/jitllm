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
    public void llamaQ4_0BatchedPrefillOnTensorCoresIsImplemented() {
        assertEquals(
                Optional.empty(),
                check(
                        "llama",
                        DataType.Q4_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertTrue(
                check(
                                "llama",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                false)
                        .isPresent());
        assertTrue(
                check("llama", DataType.Q4_0, ExecutionMode.STANDARD, BackendId.CUDA, true)
                        .isPresent());
    }

    @Test
    public void qwen35Q8_0BatchedPrefillOnTensorCoresIsImplemented() {
        assertEquals(
                Optional.empty(),
                check(
                        "qwen35",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertTrue(
                check(
                                "qwen35",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true)
                        .isPresent());
    }

    @Test
    public void decodedWeightProjectionsOnTensorCoresAreImplemented() {
        for (String family : new String[] {"llama", "qwen2", "phi3", "granite", "mistral"}) {
            for (DataType weights : new DataType[] {DataType.Q8_0, DataType.F16}) {
                assertEquals(
                        family + " " + weights,
                        Optional.empty(),
                        check(
                                family,
                                weights,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true));
            }
        }
        assertEquals(
                Optional.empty(),
                check(
                        "qwen3",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
    }

    @Test
    public void everythingElseIsRefused() {
        assertTrue(
                check(
                                "qwen3",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                true)
                        .isPresent());
        assertTrue(
                check(
                                "llama",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.CUDA,
                                false)
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

    // ── Metal: each provider declares what its MLX prefill implements ──────────

    @Test
    public void mlxFamiliesAreAdmittedOnMetalForTheWeightsTheirProvidersDeclare() {
        for (String family :
                new String[] {
                    "llama",
                    "mistral",
                    "qwen2",
                    "deepseek-r1-distill-qwen",
                    "qwen3",
                    "phi3",
                    "granite"
                }) {
            for (DataType weights : new DataType[] {DataType.F16, DataType.Q8_0}) {
                assertEquals(
                        family + " " + weights,
                        Optional.empty(),
                        check(
                                family,
                                weights,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.METAL,
                                false));
            }
        }
        for (DataType weights : new DataType[] {DataType.Q8_0, DataType.Q4_0}) {
            assertEquals(
                    "qwen35 " + weights + " has an MLX path for its projections",
                    Optional.empty(),
                    check(
                            "qwen35",
                            weights,
                            ExecutionMode.BATCH_PREFILL_DECODE,
                            BackendId.METAL,
                            false));
        }
        assertEquals(
                "gemma4 Q8_0 has an MLX path for its projections",
                Optional.empty(),
                check(
                        "gemma4",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.METAL,
                        false));
        assertEquals(
                "llama retains Q4_0 and has an MLX path for it",
                Optional.empty(),
                check(
                        "llama",
                        DataType.Q4_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.METAL,
                        false));
    }

    @Test
    public void onMetalWhatNoProviderDeclaresIsRefused() {
        // Q4_0 for a family whose provider lists no Q4_0 MLX path, and families with none at all.
        for (String family : new String[] {"qwen3", "qwen2", "phi3", "granite", "mistral"}) {
            Optional<String> reason =
                    check(
                            family,
                            DataType.Q4_0,
                            ExecutionMode.BATCH_PREFILL_DECODE,
                            BackendId.METAL,
                            false);
            assertTrue(family + " Q4_0", reason.isPresent());
            assertTrue(reason.get(), reason.get().contains("on metal"));
        }
        assertTrue(
                "qwen35 F16",
                check(
                                "qwen35",
                                DataType.F16,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.METAL,
                                false)
                        .isPresent());
        for (DataType weights : new DataType[] {DataType.F16, DataType.Q4_0}) {
            assertTrue(
                    "gemma4 " + weights,
                    check(
                                    "gemma4",
                                    weights,
                                    ExecutionMode.BATCH_PREFILL_DECODE,
                                    BackendId.METAL,
                                    false)
                            .isPresent());
        }
        for (String family : new String[] {"devstral", "qwen2-moe"}) {
            for (DataType weights : new DataType[] {DataType.F16, DataType.Q8_0, DataType.Q4_0}) {
                assertTrue(
                        family + " " + weights,
                        check(
                                        family,
                                        weights,
                                        ExecutionMode.BATCH_PREFILL_DECODE,
                                        BackendId.METAL,
                                        false)
                                .isPresent());
            }
        }
    }

    @Test
    public void onMetalOnlyTheBatchedPrefillRunsOnMlx() {
        for (ExecutionMode mode :
                new ExecutionMode[] {ExecutionMode.STANDARD, ExecutionMode.PREFILL_DECODE}) {
            Optional<String> reason = check("llama", DataType.F16, mode, BackendId.METAL, false);
            assertTrue(mode.toString(), reason.isPresent());
            assertTrue(reason.get(), reason.get().contains("--batch-prefill-size"));
        }
    }

    @Test
    public void theMetalCapabilityIsNotTheCudaOne() {
        // Gemma 4 has a cuBLAS path for Q4_0 and none on MLX; Llama has both.
        assertEquals(
                Optional.empty(),
                check(
                        "gemma4",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertEquals(
                Optional.empty(),
                check(
                        "gemma4",
                        DataType.Q4_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertTrue(
                check(
                                "gemma4",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.METAL,
                                false)
                        .isPresent());
        assertEquals(
                java.util.Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0),
                org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry.provider("llama")
                        .orElseThrow()
                        .nativeLibraryWeights(BackendId.METAL));
        assertEquals(
                java.util.Set.of(),
                org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry.provider("llama")
                        .orElseThrow()
                        .nativeLibraryWeights(BackendId.OPENCL));
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
