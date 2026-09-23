package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jllm.backend.tornado.plan.SingleTokenForwardPlan;
import org.beehive.jllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Asserts, in one process, that decoding does not change the program that was compiled at warm-up.
 * Per {@code verification-gates.md}:
 *
 * <ol>
 *   <li>compile once and record the task-graph count, the ordered task names, the grid-scheduler
 *       entry set and a SHA-256 over every task's generated kernel source;
 *   <li>decode at least 100 tokens;
 *   <li>assert all recorded values unchanged and that no additional compilation occurred.
 * </ol>
 *
 * <p><b>This gate is independent of numerical determinism</b> and therefore runs for FP16 as well
 * as Q8_0. None of the observables depend on the logits being reproducible; a failure here means
 * the <i>program</i> changed, which cannot be dismissed as arithmetic drift. The numerical half of
 * the identity claim belongs to {@link GoldenLogitsAccelTest}.
 *
 * <p>The gate owns its execution plan instead of reusing {@link TornadoVMMasterPlan}, because the
 * kernel-source dump happens on the compile path and {@code withPrintKernel()} must therefore be
 * set before warm-up — a master plan compiles inside its own constructor. The task graphs, the grid
 * scheduler and the decode sequence are the production ones; only the plan wrapper is local, and
 * {@link IdentityPlan} mirrors {@code TornadoVMMasterPlanSingleToken} exactly.
 */
public class CompiledProgramIdentityAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    /** "≥ 100 tokens" from the gate definition. */
    private static final int DECODE_TOKENS = 120;

    private static final int CONTEXT_LENGTH = GoldenCapture.CONTEXT_LENGTH;

    @Test
    public void llama3_2_1b_f16_programIsIdentical() throws Exception {
        assertProgramIdentity(Fixture.LLAMA_3_2_1B_F16);
    }

    @Test
    public void llama3_2_1b_q8_0_programIsIdentical() throws Exception {
        assertProgramIdentity(Fixture.LLAMA_3_2_1B_Q8_0);
    }

    private void assertProgramIdentity(Fixture fixture) throws Exception {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: fixture " + fixture.fileName, false);
        }
        if (ProgramIdentity.kernelDumpRedirectedToFile()) {
            System.out.println(
                    "[SKIP] environment absent — -Dtornado.print.kernel.dir sends the"
                            + " kernel dump to a file, where this gate cannot observe it; unset it to run.");
            assumeTrue("tornado.print.kernel.dir is set", false);
        }

        Model model = ModelLoader.loadModel(modelPath, CONTEXT_LENGTH, true, true);
        State state = model.createNewState();
        SingleTokenForwardPlan forwardPlan =
                ForwardPlanFactory.createSingleToken(model.weights().dataType(), state, model);

        ProgramIdentity.SourceRecorder recorder = ProgramIdentity.SourceRecorder.install();
        IdentityPlan plan = null;
        try {
            plan = new IdentityPlan(state, model, forwardPlan);
            plan.compile();

            List<String> compiled = recorder.sources();
            assertTrue(
                    "no kernel source was captured — withPrintKernel() produced nothing, so this"
                            + " gate would be asserting emptiness. Check that the backend still dumps via"
                            + " RuntimeUtilities.dumpKernel.",
                    !compiled.isEmpty());

            ProgramIdentity.Snapshot before =
                    ProgramIdentity.snapshot(
                            forwardPlan.getImmutableTaskGraphs().size(),
                            compiled,
                            forwardPlan.getGridScheduler());

            plan.forceCopyInReadOnlyData();
            int compilationsAfterWarmUp = recorder.count();

            // A compile-time counter of zero only means something if the profiler is running at
            // all; without this the JIT assertion below would hold on a plan that measures nothing.
            assertTrue(
                    "the profiler reported no execution time, so its compile-time counter cannot"
                            + " be trusted to be a measured zero",
                    plan.lastTotalTime() > 0);
            assertEquals("warm-up left JIT compilation pending", 0L, plan.lastCompileTime());

            int token = beginToken(model);
            for (int position = 0; position < DECODE_TOKENS; position++) {
                org.beehive.jllm.inference.Logits logits =
                        org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                                model, state, token, position, plan);
                token = argmax(logits);
                assertEquals(
                        "recompilation during decode at position " + position,
                        compilationsAfterWarmUp,
                        recorder.count());
                assertEquals(
                        "JIT compilation ran during decode at position " + position,
                        0L,
                        plan.lastCompileTime());
            }

            ProgramIdentity.Snapshot after =
                    ProgramIdentity.snapshot(
                            forwardPlan.getImmutableTaskGraphs().size(),
                            recorder.sources(),
                            forwardPlan.getGridScheduler());

            assertEquals("number of task graphs", before.graphCount(), after.graphCount());
            assertEquals("ordered task names", before.taskNames(), after.taskNames());
            assertEquals(
                    "per-task kernel source hashes", before.sourceHashes(), after.sourceHashes());
            assertEquals("grid-scheduler entries", before.gridEntries(), after.gridEntries());

            System.out.println(
                    "[IDENTITY] "
                            + fixture.quantization
                            + " "
                            + before.describe()
                            + " unchanged across "
                            + DECODE_TOKENS
                            + " decoded tokens");
        } finally {
            recorder.uninstall();
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }
    }

    private static int beginToken(Model model) {
        return model.shouldAddBeginOfText() ? model.chatFormat().getBeginOfText() : 0;
    }

    private static int argmax(org.beehive.jllm.inference.Logits logits) {
        int best = 0;
        float bestValue = logits.get(0);
        for (int i = 1; i < logits.size(); i++) {
            if (logits.get(i) > bestValue) {
                bestValue = logits.get(i);
                best = i;
            }
        }
        return best;
    }

    private static int argmax(FloatArray logits) {
        int best = 0;
        float bestValue = logits.get(0);
        for (int i = 1; i < logits.getSize(); i++) {
            if (logits.get(i) > bestValue) {
                bestValue = logits.get(i);
                best = i;
            }
        }
        return best;
    }

    /**
     * The production single-token plan, minus the compile-inside-the-constructor step, so that
     * {@code withPrintKernel()} can be set first. The graph ordering below is {@code
     * TornadoVMMasterPlanSingleToken}'s; keep the two in step.
     */
    private static final class IdentityPlan implements TornadoVMMasterPlan {

        private final State state;
        private final Model model;
        private final SingleTokenForwardPlan forwardPlan;
        private final SingleTokenForwardTaskGraphLayout layout;
        private final TornadoExecutionPlan executionPlan;
        private volatile long lastCompileTime;
        private volatile long lastTotalTime;

        IdentityPlan(State state, Model model, SingleTokenForwardPlan forwardPlan) {
            this.state = state;
            this.model = model;
            this.forwardPlan = forwardPlan;
            this.layout = forwardPlan.getTaskGraphLayout();
            this.executionPlan = createExecutionPlan();
        }

        @Override
        public TornadoExecutionPlan createExecutionPlan() {
            List<ImmutableTaskGraph> graphs = forwardPlan.getImmutableTaskGraphs();
            return new TornadoExecutionPlan(graphs.toArray(new ImmutableTaskGraph[0]));
        }

        /** Step 1 of the gate: compile, with the kernel dump enabled and the profiler recording. */
        void compile() {
            executionPlan.withProfiler(ProfilerMode.SILENT);
            executionPlan.withPrintKernel();
            executionPlan.withPreCompilation();
        }

        long lastTotalTime() {
            return lastTotalTime;
        }

        /**
         * Zero once warm-up is done: the profiler counts compilation per execution, not
         * cumulatively.
         */
        long lastCompileTime() {
            return lastCompileTime;
        }

        @Override
        public FloatArray tornadoVMForwardDecode(int position) {
            executionPlan
                    .withGraph(layout.activationIdx())
                    .withGridScheduler(forwardPlan.getGridScheduler())
                    .execute();

            state.setPosition(position);
            state.workspace.temp.clear();
            state.workspace.tempFFN.clear();

            for (int layer = 0; layer < model.configuration().numberOfLayers(); layer++) {
                executionPlan
                        .withGraph(layout.layerIdx(layer))
                        .withGridScheduler(forwardPlan.getGridScheduler())
                        .execute();
            }

            state.workspace.tempLogits.clear();
            state.workspace.wrapLogits.clear();
            TornadoExecutionResult result =
                    executionPlan
                            .withGraph(layout.logitsIdx())
                            .withGridScheduler(forwardPlan.getGridScheduler())
                            .execute();
            record(result);

            return state.workspace.wrapLogits;
        }

        @Override
        public void resetSequenceState() {}

        @Override
        public void forceCopyInReadOnlyData() {
            state.workspace.wrapX.clear();
            state.resetPositionHolder();
            TornadoExecutionResult result = null;
            for (int graph = 0; graph < layout.totalGraphs(); graph++) {
                result =
                        executionPlan
                                .withGraph(graph)
                                .withGridScheduler(forwardPlan.getGridScheduler())
                                .execute();
            }
            record(result);
        }

        private void record(TornadoExecutionResult result) {
            if (result == null) {
                return;
            }
            lastCompileTime = result.getProfilerResult().getCompileTime();
            lastTotalTime = result.getProfilerResult().getTotalTime();
        }

        @Override
        public void freeTornadoExecutionPlan() {
            executionPlan.freeDeviceMemory();
        }
    }
}
