package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The admission every GPU plan passes first, as a pure function of the family, its weights, the
 * policy and the devices. The key/value cache precision is not an input: a family that cannot run
 * on a device is refused with an FP32 cache as with an FP16 one, and in single-token decode as in
 * either prefill.
 */
public class PlanSupportTest {

    private static final ExecutionPolicy SINGLE_TOKEN = ExecutionPolicy.builder().build();

    private static final ExecutionPolicy SEQUENTIAL =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(1)
                    .build();

    private static final ExecutionPolicy BATCHED =
            ExecutionPolicy.builder()
                    .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                    .prefillBatchSize(128)
                    .build();

    private static final Map<String, ExecutionPolicy> MODES =
            Map.of("single-token", SINGLE_TOKEN, "sequential", SEQUENTIAL, "batched", BATCHED);

    private static Optional<String> check(
            String architecture, ExecutionPolicy policy, boolean split, BackendId... devices) {
        return check(architecture, DataType.Q8_0, policy, split, devices);
    }

    private static Optional<String> check(
            String architecture,
            DataType weights,
            ExecutionPolicy policy,
            boolean split,
            BackendId... devices) {
        return PlanSupport.unsupported(architecture, weights, policy, List.of(devices), split);
    }

    @Test
    public void deepseek2RunsOnCudaInEveryMode() {
        MODES.forEach(
                (name, policy) ->
                        assertEquals(
                                name,
                                Optional.empty(),
                                check("deepseek2", policy, false, BackendId.CUDA)));
    }

    @Test
    public void deepseek2IsRefusedOffCudaInEveryMode() {
        for (BackendId backend : List.of(BackendId.OPENCL, BackendId.METAL)) {
            MODES.forEach(
                    (name, policy) -> {
                        Optional<String> reason = check("deepseek2", policy, false, backend);
                        assertTrue(name + " on " + backend, reason.isPresent());
                        assertTrue(reason.get(), reason.get().contains("CUDA"));
                    });
        }
    }

    @Test
    public void theCpuIsNeverRefused() {
        MODES.forEach(
                (name, policy) ->
                        assertEquals(
                                name,
                                Optional.empty(),
                                check("deepseek2", policy, false, BackendId.CPU)));
    }

    @Test
    public void aFamilyWithoutADeviceRestrictionRunsEverywhere() {
        for (BackendId backend : List.of(BackendId.CUDA, BackendId.OPENCL, BackendId.METAL)) {
            MODES.forEach(
                    (name, policy) ->
                            assertEquals(
                                    name + " on " + backend,
                                    Optional.empty(),
                                    check("llama", policy, false, backend)));
        }
    }

    @Test
    public void aSplitRefusesSequentialPrefill() {
        Optional<String> reason = check("llama", SEQUENTIAL, true, BackendId.CUDA);
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("--devices"));
        assertTrue(reason.get(), reason.get().contains("--batch-prefill-size"));
        // Without a split the same policy runs.
        assertEquals(Optional.empty(), check("llama", SEQUENTIAL, false, BackendId.CUDA));
    }

    @Test
    public void aSplitRunsSingleTokenAndBatchedPrefill() {
        for (String architecture : List.of("gemma4", "qwen35", "qwen35moe", "deepseek2")) {
            for (ExecutionPolicy policy : List.of(SINGLE_TOKEN, BATCHED)) {
                assertEquals(
                        architecture,
                        Optional.empty(),
                        check(architecture, policy, true, BackendId.CUDA, BackendId.CUDA));
            }
        }
    }

    @Test
    public void aFamilyWithoutStageLayersIsNotSplit() {
        for (String architecture : List.of("qwen3", "phi3", "mistral")) {
            Optional<String> reason =
                    check(architecture, SINGLE_TOKEN, true, BackendId.CUDA, BackendId.CUDA);
            assertTrue(architecture, reason.isPresent());
            assertTrue(reason.get(), reason.get().contains("drop --devices"));
            // Without a split the same family runs.
            assertEquals(
                    Optional.empty(), check(architecture, SINGLE_TOKEN, false, BackendId.CUDA));
        }
        // A family splits only the representations whose components build stages.
        for (String architecture : List.of("llama", "gemma4")) {
            assertTrue(
                    architecture,
                    check(architecture, DataType.F16, SINGLE_TOKEN, true, BackendId.CUDA)
                            .isPresent());
        }
    }

    @Test
    public void llamaQ8_0SplitsItsSingleTokenPlanOnly() {
        assertEquals(Optional.empty(), check("llama", SINGLE_TOKEN, true, BackendId.CUDA));
        Optional<String> reason = check("llama", BATCHED, true, BackendId.CUDA);
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("drop --batch-prefill-size"));
        assertEquals(Optional.empty(), check("llama", BATCHED, false, BackendId.CUDA));
    }

    @Test
    public void llamaQ4_0StagesNeedNativeLibrariesForABatchedSplit() {
        // One stage holds the whole model, whose layers decode the projections themselves.
        assertEquals(
                Optional.empty(), check("llama", DataType.Q4_0, BATCHED, true, BackendId.CUDA));
        Optional<String> reason =
                check("llama", DataType.Q4_0, BATCHED, true, BackendId.CUDA, BackendId.CUDA);
        assertTrue(reason.isPresent());
        assertTrue(reason.get(), reason.get().contains("--with-native-libraries"));
        assertEquals(
                Optional.empty(),
                check("llama", DataType.Q4_0, SINGLE_TOKEN, true, BackendId.CUDA, BackendId.CUDA));
    }

    @Test
    public void everyStageDeviceIsChecked() {
        assertTrue(check("deepseek2", BATCHED, true, BackendId.CUDA, BackendId.OPENCL).isPresent());
        assertEquals(
                Optional.empty(),
                check("deepseek2", BATCHED, true, BackendId.CUDA, BackendId.CUDA));
    }
}
