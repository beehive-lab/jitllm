package org.beehive.jitllm.api;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.memory.MemoryComponent;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * Llama Q8_0's tensor-core batched prefill decodes its projections to FP16, and keeps them resident
 * only where a copy per session fits the configured budget beside everything else. Under a budget
 * too small for the copies it decodes per chunk instead, and still generates; under one too small
 * for even that, the load is refused before anything is loaded.
 */
public class BatchPrefillResidencyAccelTest {

    private static final String BUDGET = "tornado.device.memory";
    private static final long MIB = 1L << 20;

    @Test
    public void aTightBudgetDecodesPerChunkAndStillGenerates() throws Exception {
        Path llama = locate();
        ModelOptions options = options();
        MemoryPlan unbounded = LocalModels.preflight(llama, options);
        MemoryComponent resident = workspace(unbounded);
        assertTrue(resident.name(), resident.name().contains("resident"));

        // One MiB short of the resident plan: the copies no longer fit, the scratch does.
        long budget = (unbounded.predictedBudgetBytes() - 1) / MIB;
        withProperty(
                BUDGET,
                budget + "MB",
                () -> {
                    MemoryComponent perChunk = workspace(LocalModels.preflight(llama, options));
                    assertTrue(perChunk.name(), perChunk.name().contains("per chunk"));
                    try (LocalModel loaded = LocalModels.load(llama, options)) {
                        assertFalse(generate((TextGenerationModel) loaded).isBlank());
                    }
                });
    }

    @Test
    public void aBudgetTooSmallForEitherIsRefusedBeforeLoading() throws Exception {
        Path llama = locate();
        ModelOptions options = options();
        MemoryPlan unbounded = LocalModels.preflight(llama, options);
        // Everything but the workspace fits, with a MiB to spare; not even one scratch does.
        long others = unbounded.predictedBudgetBytes() - workspace(unbounded).predictedBytes();
        long budget = others / MIB + 1;
        withProperty(
                BUDGET,
                budget + "MB",
                () -> {
                    InsufficientDeviceMemoryException refused =
                            assertThrows(
                                    InsufficientDeviceMemoryException.class,
                                    () -> LocalModels.load(llama, options));
                    assertFalse(refused.plan().fitsConfiguredBudget());
                    assertTrue(
                            refused.plan().describe(),
                            workspace(refused.plan()).name().contains("per chunk"));
                });
    }

    private static MemoryComponent workspace(MemoryPlan plan) {
        return plan.components().stream()
                .filter(c -> c.name().startsWith("batched prefill workspace"))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no workspace in " + plan.describe()));
    }

    private static ModelOptions options() {
        return ModelOptions.builder()
                .contextLength(512)
                .maxConcurrentSessions(2)
                .executionPolicy(
                        ExecutionPolicy.builder()
                                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                .prefillBatchSize(128)
                                .build())
                .build();
    }

    private static String generate(TextGenerationModel model) throws Exception {
        try (GenerationSession session = model.newSession()) {
            return session.generate(
                            GenerationRequest.builder()
                                    .prompt("The capital of France is")
                                    .maxNewTokens(8)
                                    .temperature(0.0f)
                                    .seed(42L)
                                    .build())
                    .text();
        }
    }

    private static Path locate() {
        Path file = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0),
                file != null);
        assumeTrue("needs tensor cores", TensorCoreSupport.isTensorCoreCapableBackend());
        return file;
    }

    private static void withProperty(String key, String value, ThrowingRunnable body)
            throws Exception {
        String gpu = System.getProperty("use.tornadovm");
        String previous = System.getProperty(key);
        System.setProperty("use.tornadovm", "true");
        System.setProperty(key, value);
        try {
            body.run();
        } finally {
            if (previous == null) System.clearProperty(key);
            else System.setProperty(key, previous);
            if (gpu == null) System.clearProperty("use.tornadovm");
            else System.setProperty("use.tornadovm", gpu);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
