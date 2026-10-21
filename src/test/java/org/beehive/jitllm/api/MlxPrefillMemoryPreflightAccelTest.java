package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * The configuration that used to die part-allocated — Mistral-7B-v0.3 Q8_0, batched prefill of 2048
 * on MLX under a 14 GB budget — is refused by the preflight before the first device buffer.
 *
 * <p>Before the MLX prefill was accounted, its plan charged nothing for the affine repack of the
 * projections, the run started allocating, and TornadoVM failed with {@code Unable to allocate
 * 62390288 bytes} part-way through ({@code GPUL-MEM-001} from the runtime, with no plan attached).
 * The same configuration on the JIT kernels runs in that budget.
 *
 * <p>Metal plans are {@code CONSERVATIVE}, which is report-only; the refusal is on the plan's lower
 * bound — its buffers each counted once — which already exceeds the budget. Nothing here builds a
 * plan or uploads a weight: the refusal is the facade's, and the JIT control asks the preflight
 * directly.
 */
// @formatter:on
public class MlxPrefillMemoryPreflightAccelTest {

    private static final int BATCH = 2048;
    private static final int CONTEXT = 2064;

    @Test
    public void mistral7bQ8OnMlxIsRefusedBeforeAllocation() throws Exception {
        Path model = GoldenFixture.locate(Fixture.MISTRAL_7B_Q8_0);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.MISTRAL_7B_Q8_0),
                model != null);
        assumeTrue(
                "the MLX prefill runs on Metal only",
                BackendId.METAL.equals(TornadoDevices.current().backend()));
        assumeTrue("MLX's mlx.metallib is not loadable here", MlxPrefill.available());

        String previousGpu = System.getProperty("use.tornadovm");
        String previousBudget = System.getProperty("tornado.device.memory");
        System.setProperty("use.tornadovm", "true");
        System.setProperty("tornado.device.memory", "14GB");
        try {
            InsufficientDeviceMemoryException refused =
                    assertThrows(
                            InsufficientDeviceMemoryException.class,
                            () -> LocalModels.load(model, options(true)).close());
            MemoryPlan plan = refused.plan();
            assertNotNull(
                    "refused by the runtime during the load rather than predicted before it: "
                            + refused.getMessage(),
                    plan);
            assertEquals(MemoryPlan.Confidence.CONSERVATIVE, plan.confidence());
            assertTrue(plan.describe(), plan.exceedsConfiguredBudgetAtLowerBound());
            assertTrue(
                    "the plan does not charge the MLX prefill: " + plan.describe(),
                    plan.components().stream()
                            .anyMatch(c -> c.name().startsWith("batched prefill workspace")));
            assertTrue(refused.getMessage(), refused.getMessage().startsWith("[GPUL-MEM-001]"));

            // The JIT kernels' plan for the same file, width and budget is not refused.
            MemoryPreflight.refuseIfOverCapacity(model, CONTEXT, options(false));
        } finally {
            restore("use.tornadovm", previousGpu);
            restore("tornado.device.memory", previousBudget);
        }
    }

    private static ModelOptions options(boolean nativeLibraries) {
        return ModelOptions.builder()
                .contextLength(CONTEXT)
                .executionPolicy(
                        ExecutionPolicy.builder()
                                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                                .prefillBatchSize(BATCH)
                                .nativeLibraries(nativeLibraries)
                                .build())
                .build();
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
