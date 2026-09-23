package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.api.InsufficientDeviceMemoryException;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.memory.MemoryPlan;
import org.junit.Test;

public class MemoryPreflightAccelTest {

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
     * A known-over-capacity load fails before allocation, and says why — <b>where the backend's
     * memory model has been validated</b>.
     *
     * <p>Metal parity task 13: admission is enforced only at {@code EXACT} confidence, and Metal is
     * deliberately capped at {@code CONSERVATIVE} — whose documented meaning is "report-only; never
     * refuses on its own" (`memory-validation.md` §4). Asserting a refusal unconditionally asserted
     * a CUDA-only fact. On Metal this asserts the contract that actually applies there: the plan
     * still *reports* that the configuration does not fit, and still names the shortfall, while the
     * preflight itself does not refuse. That is a real assertion, not an exemption — a Metal build
     * that silently started refusing (or stopped reporting the shortfall) would fail it.
     */
    @Test
    public void anOverCapacityLoadIsRefusedBeforeAllocating() throws Exception {
        Path model = fixture();
        if (!memoryModelValidatedForThisBackend()) {
            assertOverCapacityIsReportedButNotRefused(model);
            return;
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        String previousBudget = System.getProperty(BUDGET);
        System.setProperty(GPU_PROPERTY, "true");
        // Far below the ~2.4 GiB this configuration needs, and a value the backend itself accepts.
        System.setProperty(BUDGET, "512MB");
        try {
            InsufficientDeviceMemoryException e =
                    assertThrows(
                            InsufficientDeviceMemoryException.class,
                            () ->
                                    LocalModels.load(
                                            model,
                                            ModelOptions.builder().contextLength(512).build()));

            assertFalse("the plan must not claim to fit", e.plan().fitsConfiguredBudget());
            String message = e.getMessage();
            assertTrue("it must state the requirement", message.contains("needs about"));
            assertTrue(
                    "it must state the configured budget", message.contains("configured budget"));
            assertTrue("it must state the shortfall", message.contains("short by"));
            assertTrue(
                    "it must name the dominant component", message.contains("Dominant component:"));
            assertTrue(
                    "it must not be mistaken for a free-memory reading",
                    message.contains("not a measurement of"));
            System.out.println(message);
        } finally {
            restore(GPU_PROPERTY, previousGpu);
            restore(BUDGET, previousBudget);
        }
    }

    /** A sufficient budget must not be refused — the check has to be a gate, not an obstacle. */
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
     * Batched prefill must predict more than single-token, and say where the difference comes from.
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
                                        org.beehive.jllm.runtime.policy.ExecutionPolicy.builder()
                                                .phaseStrategy(
                                                        org.beehive.jllm.runtime.policy
                                                                .ExecutionPolicy.PhaseStrategy
                                                                .PREFILL_DECODE)
                                                .prefillBatchSize(8)
                                                .build())
                                .build());
        assertTrue(
                "batched prefill must predict more than single-token",
                batched.predictedBudgetBytes() > single.predictedBudgetBytes());
        assertTrue(
                "and the difference must be reported as duplication rather than hidden",
                batched.duplicationBytes() > 0);
        // Key/value storage is compared on its own: single-token Llama F16 is the lowered tuple,
        // so it reserves the shared pool and its scratch block, while batched prefill keeps a
        // private cache. The two differ by exactly that one block, and by nothing else.
        long kvBlockBytes =
                2L
                        * org.beehive.jllm.runtime.memory.KeyValueReservation.BLOCK_SIZE_TOKENS
                        * 16 // layers
                        * 512 // kvDim
                        * 4; // FP32
        assertEquals(
                "a single pooled session costs its private cache plus the scratch block",
                logicalBytesOf(batched, org.beehive.jllm.runtime.memory.BufferClass.KV_CACHE)
                        + kvBlockBytes,
                logicalBytesOf(single, org.beehive.jllm.runtime.memory.BufferClass.KV_CACHE));
        assertEquals(
                "the other logical bytes barely change; it is the multiplicity that does",
                (single.logicalBytes()
                                - logicalBytesOf(
                                        single,
                                        org.beehive.jllm.runtime.memory.BufferClass.KV_CACHE))
                        / 1048576,
                (batched.logicalBytes()
                                - logicalBytesOf(
                                        batched,
                                        org.beehive.jllm.runtime.memory.BufferClass.KV_CACHE)
                                - logicalBytesOf(
                                        batched,
                                        org.beehive.jllm.runtime.memory.BufferClass.BATCH_STAGING))
                        / 1048576);
    }

    private static long logicalBytesOf(
            MemoryPlan plan, org.beehive.jllm.runtime.memory.BufferClass bufferClass) {
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

    /**
     * The {@code CONSERVATIVE} contract: the shortfall is reported, and the load is not refused.
     */
    private static void assertOverCapacityIsReportedButNotRefused(Path model) throws Exception {
        String previousBudget = System.getProperty(BUDGET);
        System.setProperty(BUDGET, "512MB");
        try {
            MemoryPlan plan =
                    LocalModels.preflight(model, ModelOptions.builder().contextLength(512).build());
            assertEquals(
                    "this backend's memory model is unvalidated, so the plan must say so",
                    MemoryPlan.Confidence.CONSERVATIVE,
                    plan.confidence());
            assertFalse(
                    "the plan must still report that the configuration does not fit,"
                            + " even though CONSERVATIVE means it will not refuse",
                    plan.fitsConfiguredBudget());
            assertTrue(
                    "and it must still name the dominant component",
                    plan.largestComponent().isPresent());
            System.out.println(
                    "[CONSERVATIVE] over-capacity reported, not refused:\n" + plan.describe());
        } finally {
            restore(BUDGET, previousBudget);
        }
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
