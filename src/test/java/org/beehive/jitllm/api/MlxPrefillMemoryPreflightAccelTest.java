package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxPrefill;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

// @formatter:off
/**
 * Mistral-7B-v0.3 Q8_0, batched prefill of 2048, under a 14 GB budget: on MLX its plan exceeds the
 * budget even at its lower bound, so the device default does not take MLX for it and runs the JIT
 * kernels, which fit. Asked for MLX explicitly, the load is not refused on that prediction: it
 * fails while allocating MLX's copies in host direct memory, with {@code GPUL-MEM-001} naming the
 * JVM's direct-memory limit.
 *
 * <p>Before the MLX prefill was accounted, its plan charged nothing for the affine repack of the
 * projections. Metal plans are {@code CONSERVATIVE}; the lower bound — the buffers each counted
 * once — is what the default's choice acts on.
 */
// @formatter:on
public class MlxPrefillMemoryPreflightAccelTest {

    private static final int BATCH = 2048;
    private static final int CONTEXT = 2064;

    @Test
    public void mistral7bQ8OnMlxIsSetAsideByTheDefaultAndFailsClearlyWhenAskedFor()
            throws Exception {
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
            MemoryPlan plan = LocalModels.preflight(model, options(true));
            assertEquals(MemoryPlan.Confidence.CONSERVATIVE, plan.confidence());
            assertTrue(plan.describe(), plan.exceedsConfiguredBudgetAtLowerBound());
            assertTrue(
                    "the plan does not charge the MLX prefill: " + plan.describe(),
                    plan.components().stream()
                            .anyMatch(c -> c.name().startsWith("batched prefill workspace")));
            // The device default's choice: MLX does not fit, the JIT kernels do.
            ModelOptions mlx = options(true);
            ModelOptions jit = options(false);
            assertFalse(LocalModels.fitsMemoryBudget(model, mlx, mlx.executionPolicy()));
            assertTrue(LocalModels.fitsMemoryBudget(model, jit, jit.executionPolicy()));

            // Asked for explicitly, MLX is loaded and fails while allocating, not before: its
            // copies are made in host direct memory, so the message names the JVM's limit.
            InsufficientDeviceMemoryException failure =
                    assertThrows(
                            InsufficientDeviceMemoryException.class,
                            () -> loadAndGenerate(model, mlx));
            assertTrue(failure.getMessage(), failure.getMessage().startsWith("[GPUL-MEM-001]"));
            assertTrue(failure.getMessage(), failure.getMessage().contains("direct memory"));
            assertTrue(failure.getMessage(), failure.getMessage().contains("MaxDirectMemorySize"));
            System.out.println(failure.getMessage());
        } finally {
            restore("use.tornadovm", previousGpu);
            restore("tornado.device.memory", previousBudget);
        }
    }

    private static void loadAndGenerate(Path model, ModelOptions options) throws Exception {
        try (LocalModel loaded = LocalModels.load(model, options);
                GenerationSession session = ((TextGenerationModel) loaded).newSession()) {
            session.generate(
                    GenerationRequest.builder()
                            .prompt("The capital of France is")
                            .maxNewTokens(4)
                            .temperature(0.0f)
                            .build());
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
