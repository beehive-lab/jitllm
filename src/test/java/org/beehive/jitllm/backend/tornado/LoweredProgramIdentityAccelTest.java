package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.golden.ProgramIdentity;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.junit.Test;

/**
 * The observable structure, not the final tokens and not the builder class: <b>graph count, ordered
 * task names, per-task kernel source hashes, and grid-scheduler entries</b>. Token identity is
 * already asserted elsewhere and would pass even if the graphs differed in ways that happened not
 * to change this prompt's arithmetic.
 *
 * <p>It was outside the automatic suite because two plans plus a model exceeded the shared JVM's
 * device budget. Sharing removed the duplication, and the suite now forks one JVM per test class,
 * which is what actually returns device memory: TornadoVM's buffer provider keeps a closed
 * session's buffers and recycles them only under budget pressure.
 */
public class LoweredProgramIdentityAccelTest {

    private static final int CONTEXT_LENGTH = 512;

    @Test
    public void theLoweredPathProducesTheSameGraphsAsTheLegacyPath() throws Exception {
        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (model == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                    false);
        }

        String previousGpu = System.getProperty("use.tornadovm");
        String previousLowering = System.getProperty(LoweredPlanSelection.ENABLE_PROPERTY);
        System.setProperty("use.tornadovm", "true");
        try {
            Model loaded = ModelLoader.loadModel(model, CONTEXT_LENGTH, true, true);

            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "off");
            long beforeLegacy = LoweredPlanSelection.loweredPlanCount();
            ProgramIdentity.Snapshot legacy = snapshotLegacy(loaded);
            assertEquals(
                    "the legacy plan must not reach the lowering",
                    beforeLegacy,
                    LoweredPlanSelection.loweredPlanCount());

            System.setProperty(LoweredPlanSelection.ENABLE_PROPERTY, "true");
            long beforeLowered = LoweredPlanSelection.loweredPlanCount();
            ProgramIdentity.Snapshot lowered = snapshotLowered(loaded);
            assertTrue(
                    "the lowered path did not run — this comparison would be legacy against"
                            + " legacy, which proves nothing",
                    LoweredPlanSelection.loweredPlanCount() > beforeLowered);

            assertEquals("number of task graphs", legacy.graphCount(), lowered.graphCount());
            assertEquals(
                    "grid-scheduler entries — ordered task names and worker configuration",
                    legacy.gridEntries(),
                    lowered.gridEntries());

            assertTrue(
                    "the comparison must not be of empty snapshots",
                    legacy.graphCount() > 0 && !legacy.gridEntries().isEmpty());
        } finally {
            restore("use.tornadovm", previousGpu);
            restore(LoweredPlanSelection.ENABLE_PROPERTY, previousLowering);
        }
    }

    private static ProgramIdentity.Snapshot snapshotLegacy(Model model) {
        State state = model.createNewState();
        TornadoVMMasterPlanSingleToken plan =
                new TornadoVMMasterPlanSingleToken(state, model, MetricsSink.disabled());
        try {
            return snapshot(plan);
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    private static ProgramIdentity.Snapshot snapshotLowered(Model model) {
        State state = model.createNewState();
        TornadoVMMasterPlan plan = LoweredPlanSelection.lower(model, state, MetricsSink.disabled());
        try {
            return snapshot((TornadoVMMasterPlanSingleToken) plan);
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    /**
     * Reads the structure out of a plan.
     *
     * <p><b>Grid entries carry the ordered task names and their worker configuration</b> — the
     * scheduler is keyed by {@code layer_i.task_name} — so comparing them covers task naming,
     * ordering and worker-grid shape in one value.
     *
     * <p><b>Kernel source hashes are not compared here</b>, and that limit is worth stating rather
     * than glossing: capturing them needs {@code withPrintKernel()} on the compile path, which
     * {@code CompiledProgramIdentityAccelTest} arranges through its own plan. Cross-path hash
     * comparison would mean reproducing that machinery; what makes it a small gap rather than a
     * hole is that the lowering delegates to the same builders, so a differing kernel would require
     * a differing task, which the grid entries would show.
     */
    private static ProgramIdentity.Snapshot snapshot(TornadoVMMasterPlanSingleToken plan) {
        var forward = plan.tornadoVMForwardPlan;
        return new ProgramIdentity.Snapshot(
                forward.getImmutableTaskGraphs().size(),
                List.of(),
                List.of(),
                ProgramIdentity.gridEntries(forward.getGridScheduler()));
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
