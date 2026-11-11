package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.junit.Test;

public class MemoryPreflightAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final String GPU_PROPERTY = "use.tornadovm";
    private static final String BUDGET = "tornado.device.memory";

    /** It runs from metadata alone: no weights are materialized, on host or device. */
    @Test
    public void theReportIsProducedWithoutLoadingTheModel() throws Exception {
        Path model = fixture();
        MemoryPlan plan =
                LocalModels.preflight(model, ModelOptions.builder().contextLength(512).build());

        assertTrue("the plan must have components", plan.components().size() >= 4);
        assertTrue("logical bytes must be reported", plan.logicalBytes() > 0);
        assertTrue(
                "the predicted budget is at least the logical bytes",
                plan.predictedBudgetBytes() >= plan.logicalBytes());
        // Metal parity task 13: confidence is backend-dependent, and asserting EXACT
        // unconditionally
        // asserted a CUDA-only fact. memory-validation.md requires Metal to stay CONSERVATIVE until
        // its own budget bisection exists ("must not report EXACT using CUDA-derived assumptions"),
        // so pin exactly that on Metal rather than relaxing the assertion — a regression that let
        // Metal claim EXACT again would fail here, which is the guarantee worth holding.
        if (memoryModelValidatedForThisBackend()) {
            assertEquals(
                    "a recognised topology is exact",
                    MemoryPlan.Confidence.EXACT,
                    plan.confidence());
        } else {
            assertEquals(
                    "Metal must stay CONSERVATIVE until its own thresholds are bisected",
                    MemoryPlan.Confidence.CONSERVATIVE,
                    plan.confidence());
        }
        assertTrue(
                "the report must name its assumptions", plan.assumptions().contains("context 512"));
        assertTrue("the largest component must be identified", plan.largestComponent().isPresent());
        // The contract excludes physical memory, and the report must not imply otherwise.
        assertTrue(
                "the report must say physical free memory is out of scope",
                plan.describe()
                        .contains("physical free device memory is not part of this contract"));
        System.out.println(plan.describe());
    }

    /**
     * A configuration predicted over the budget is reported as such, with its shortfall and
     * dominant component, and its load is not refused on that prediction: the allocation decides.
     * The budget set here is the prediction's; the test JVM's TornadoVM budget is larger, so the
     * load allocates and runs.
     */
    @Test
    public void anOverBudgetPredictionIsReportedAndRefusesNothing() throws Exception {
        Path model = fixture();
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousBudget = System.getProperty(BUDGET);
        System.setProperty(GPU_PROPERTY, "true");
        // Far below the ~2.4 GiB this configuration needs, and a value the backend itself accepts.
        System.setProperty(BUDGET, "512MB");
        try {
            ModelOptions options = ModelOptions.builder().contextLength(512).build();
            MemoryPlan plan = LocalModels.preflight(model, options);
            assertFalse("the plan must report that it does not fit", plan.fitsConfiguredBudget());
            assertTrue(
                    "even its buffers counted once exceed the budget",
                    plan.exceedsConfiguredBudgetAtLowerBound());
            assertTrue("it must name the dominant component", plan.largestComponent().isPresent());
            try (LocalModel loaded = LocalModels.load(model, options)) {
                assertTrue("the model must load", loaded != null);
            }
            System.out.println("over budget, reported and not refused:\n" + plan.describe());
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(BUDGET, previousBudget);
        }
    }

    /** A sufficient budget loads normally. */
    @Test
    public void aSufficientBudgetLoadsNormally() throws Exception {
        Path model = fixture();
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousBudget = System.getProperty(BUDGET);
        System.setProperty(GPU_PROPERTY, "true");
        System.setProperty(BUDGET, "8GB");
        try (LocalModel loaded =
                LocalModels.load(model, ModelOptions.builder().contextLength(512).build())) {
            assertTrue("the model must load", loaded != null);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(BUDGET, previousBudget);
        }
    }

    /**
     * Batched prefill must predict more than single-token, and say where the difference comes from:
     * its staging buffers, since the weights are shared between its two graph families.
     */
    @Test
    public void batchedPrefillPredictsItsExtraCost() throws Exception {
        Path model = fixture();
        MemoryPlan single =
                LocalModels.preflight(model, ModelOptions.builder().contextLength(512).build());
        MemoryPlan batched =
                LocalModels.preflight(
                        model,
                        ModelOptions.builder()
                                .contextLength(512)
                                .executionPolicy(
                                        org.beehive.jitllm.runtime.policy.ExecutionPolicy.builder()
                                                .phaseStrategy(
                                                        org.beehive.jitllm.runtime.policy
                                                                .ExecutionPolicy.PhaseStrategy
                                                                .PREFILL_DECODE)
                                                .prefillBatchSize(8)
                                                .build())
                                .build());
        assertTrue(
                "batched prefill must predict more than single-token",
                batched.predictedBudgetBytes() > single.predictedBudgetBytes());
        // The decode graphs consume the weights the batched-prefill graphs uploaded, so the plan
        // holds them once (measured: this model runs batched in a 2600 MB budget, where counting
        // them twice predicted 4.2 GB). What batched prefill adds is its staging, not a copy.
        assertEquals("the weights are not charged twice", 0L, batched.duplicationBytes());
        assertTrue(
                "the extra is batch staging",
                logicalBytesOf(batched, org.beehive.jitllm.runtime.memory.BufferClass.BATCH_STAGING)
                        > 0);
        // Key/value storage is compared on its own: single-token Llama F16 is the lowered tuple,
        // so it reserves the shared pool and its scratch block, while batched prefill keeps a
        // private cache. The two differ by exactly that one block, and by nothing else.
        long kvBlockBytes =
                2L
                        * org.beehive.jitllm.runtime.memory.KeyValueReservation.BLOCK_SIZE_TOKENS
                        * 16 // layers
                        * 512 // kvDim
                        * 4; // FP32
        assertEquals(
                "a single pooled session costs its private cache plus the scratch block",
                logicalBytesOf(batched, org.beehive.jitllm.runtime.memory.BufferClass.KV_CACHE)
                        + kvBlockBytes,
                logicalBytesOf(single, org.beehive.jitllm.runtime.memory.BufferClass.KV_CACHE));
        assertEquals(
                "the other logical bytes barely change; it is the multiplicity that does",
                (single.logicalBytes()
                                - logicalBytesOf(
                                        single,
                                        org.beehive.jitllm.runtime.memory.BufferClass.KV_CACHE))
                        / 1048576,
                (batched.logicalBytes()
                                - logicalBytesOf(
                                        batched,
                                        org.beehive.jitllm.runtime.memory.BufferClass.KV_CACHE)
                                - logicalBytesOf(
                                        batched,
                                        org.beehive.jitllm.runtime.memory.BufferClass
                                                .BATCH_STAGING))
                        / 1048576);
    }

    private static long logicalBytesOf(
            MemoryPlan plan, org.beehive.jitllm.runtime.memory.BufferClass bufferClass) {
        return plan.components().stream()
                .filter(c -> c.bufferClass() == bufferClass)
                .mapToLong(c -> c.logicalBytes())
                .sum();
    }

    /**
     * Whether this backend's memory model has been measured, which is what {@code EXACT} confidence
     * certifies. Mirrors {@code TornadoMemoryModel}'s own condition: CUDA's thresholds were
     * bisected, Metal's have not been (Metal parity task 13), so Metal is capped at {@code
     * CONSERVATIVE}. Read from the resolved device rather than a property, so it follows the device
     * the run actually got.
     */
    private static boolean memoryModelValidatedForThisBackend() {
        return TornadoDevices.current().backend() != BackendId.METAL;
    }

    private static Path fixture() {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }
        return model;
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
