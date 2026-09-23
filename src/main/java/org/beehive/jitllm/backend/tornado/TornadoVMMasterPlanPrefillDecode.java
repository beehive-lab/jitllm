package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jitllm.backend.tornado.plan.PrefillDecodeForwardPlan;
import org.beehive.jitllm.backend.tornado.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * GPU execution plan for sequential (single-token) prefill/decode separation.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all graphs so that the KV cache ({@code
 * wrapKeyCache}, {@code wrapValueCache}) is allocated once and remains on device across both
 * phases. Prefill and decode reuse the same N layer graphs; only the logits graph is skipped during
 * prefill.
 *
 * <p>Graph layout (N+2 graphs total):
 *
 * <pre>
 *   [0]      decodeActivation    single-token FP16 → FP32; KV-cache allocated on first execution
 *   [1.N]   layer_0.layer_N-1  transformer layers (attention + FFN)
 *   [N+1]    logits              final RMSNorm + wcls matmul
 * </pre>
 *
 * <p>Two forward passes:
 *
 * <ul>
 *   <li>{@link #tornadoVMForwardPrefill} — graphs 0.N (activation + layers), logits skipped. Called
 *       once per prompt token; populates the KV cache.
 *   <li>{@link #tornadoVMForwardDecode} — full pass including logits. Called once per generated
 *       token; returns logits for sampling.
 * </ul>
 */
// @formatter:on
public class TornadoVMMasterPlanPrefillDecode implements TornadoVMMasterPlan {

    /**
     * Rule 16: library code routes its output through the platform logger, so an embedder can
     * silence or redirect it. Reached only under {@code jitllm.EnableTimingForTornadoVMInit}.
     */
    private static final System.Logger LOGGER =
            System.getLogger(TornadoVMMasterPlanPrefillDecode.class.getName());

    @Override
    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        return PlanDiagnostics.describe(
                state, "prefill-decode", 1, "JIT kernels (no tensor-core MMA)", "JIT kernels");
    }

    private final State state;
    private final Model model;
    private final Configuration config;

    PrefillDecodeForwardPlan prefillDecodeForwardPlan;
    PrefillDecodeForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    /**
     * Rule 17 seam. Costs one boolean test per execution while the sink is the disabled default.
     */
    private final TornadoMetricsReporter metrics;

    // ── Construction ─────────────────────────────────────────────────────────
    TornadoVMMasterPlanPrefillDecode(State state, Model model, MetricsSink sink) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            LOGGER.log(System.Logger.Level.INFO, "Starting TornadoVM initialization...");
        }

        this.state = state;
        this.model = model;
        this.config = model.configuration();
        this.metrics = new TornadoMetricsReporter(sink);

        long startTime = System.nanoTime();
        this.executionPlan = createExecutionPlan();
        // Before compilation and the weight upload, so a plan that fails either still prints.
        TaskGraphChainPrinter.printIfRequested(taskGraphChain(), model);
        metrics.enableOn(executionPlan);
        long planCreationTime = System.nanoTime();

        if (CUDA_GRAPHS) {
            executionPlan.withAllGraphs().withCUDAGraph();
        }
        // Large one-shot uploads (the weights, below) chunked through pinned staging buffers
        // instead of pinning each whole segment: measured 2.6 s off a 14.3 s cold start of the
        // 27B model on CUDA, steady-state throughput unchanged; a no-op on the other backends.
        executionPlan.withStagedTransfers();
        executionPlan.withPreCompilation();
        long warmupTime = System.nanoTime();

        forceCopyInReadOnlyData();
        long copyTime = System.nanoTime();

        RunMetrics.setTornadoMetrics(
                planCreationTime - startTime, warmupTime - planCreationTime, copyTime - warmupTime);
        metrics.reportSetUp(
                planCreationTime - startTime, warmupTime - planCreationTime, copyTime - warmupTime);
    }

    // ── Plan construction ─────────────────────────────────────────────────────

    /** The graphs, when they run, and what each is, for {@code --print-taskgraph-chain}. */
    TaskGraphChainPrinter.Chain taskGraphChain() {
        var layout = taskGraphLayout;
        var roles = new java.util.HashMap<Integer, String>();
        roles.put(layout.activationIdx(), "activation");
        TaskGraphChainPrinter.label(roles, layout.layerIdx(0), layout.N(), "layers");
        roles.put(layout.logitsIdx(), "logits");
        var layers =
                TaskGraphChainPrinter.span(layout.layerIdx(0), layout.layerIdx(layout.N() - 1));
        var prefill =
                TaskGraphChainPrinter.concat(java.util.List.of(layout.activationIdx()), layers);
        var decode = TaskGraphChainPrinter.concat(prefill, java.util.List.of(layout.logitsIdx()));
        return new TaskGraphChainPrinter.Chain(
                "prefill-decode",
                prefillDecodeForwardPlan.getImmutableTaskGraphs(),
                prefillDecodeForwardPlan.getGridScheduler(),
                java.util.List.of(
                        new TaskGraphChainPrinter.Phase(
                                "warm-up",
                                "once, at plan build",
                                TaskGraphChainPrinter.span(0, layout.logitsIdx())),
                        new TaskGraphChainPrinter.Phase(
                                "prefill token", "per prompt token", prefill),
                        new TaskGraphChainPrinter.Phase(
                                "decode token", "per generated token", decode)),
                roles);
    }

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        DataType weightType = model.weights().dataType();
        this.prefillDecodeForwardPlan =
                ForwardPlanFactory.createPrefillDecode(weightType, state, model);
        this.taskGraphLayout = prefillDecodeForwardPlan.getTaskGraphLayout();
        var taskGraphs = prefillDecodeForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    @Override
    public void resetSequenceState() {
        TornadoVMMasterPlan.resetSequenceState(executionPlan, state, taskGraphLayout.layerIdx(0));
    }

    /** Runs all graphs once to trigger FIRST_EXECUTION uploads and warm up CUDA graphs. */
    // @formatter:off
    @Override
    public void forceCopyInReadOnlyData() {
        state.workspace.wrapX.clear();
        state.resetPositionHolder();

        for (int i = 0; i <= taskGraphLayout.logitsIdx(); i++) {
            var g =
                    executionPlan
                            .withGraph(i)
                            .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                g.withCUDAGraph();
            }
            metrics.report(g.execute());
        }
    }

    // @formatter:on

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * GPU prefill forward: activation + all transformer layers, logits skipped.
     *
     * @param position sequence position being processed
     */
    // @formatter:off
    public void tornadoVMForwardPrefill(int position) {
        var act =
                executionPlan
                        .withGraph(taskGraphLayout.activationIdx())
                        .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            act.withCUDAGraph();
        }
        metrics.report(act.execute());

        state.setPosition(position);
        state.workspace.temp.clear();
        state.workspace.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            var l =
                    executionPlan
                            .withGraph(taskGraphLayout.layerIdx(layer))
                            .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                l.withCUDAGraph();
            }
            metrics.report(l.execute());
        }
    }

    // @formatter:on

    /**
     * GPU decode forward: full execution including logits.
     *
     * @param position sequence position being processed
     * @return logits array for token sampling
     */
    // @formatter:off
    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        var act =
                executionPlan
                        .withGraph(taskGraphLayout.activationIdx())
                        .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            act.withCUDAGraph();
        }
        metrics.report(act.execute());

        state.setPosition(position);
        state.workspace.temp.clear();
        state.workspace.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            var l =
                    executionPlan
                            .withGraph(taskGraphLayout.layerIdx(layer))
                            .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                l.withCUDAGraph();
            }
            metrics.report(l.execute());
        }

        state.workspace.tempLogits.clear();
        state.workspace.wrapLogits.clear();
        var logits =
                executionPlan
                        .withGraph(taskGraphLayout.logitsIdx())
                        .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            logits.withCUDAGraph();
        }
        metrics.report(logits.execute());

        return state.workspace.wrapLogits;
    }

    // @formatter:on

    @Override
    public void freeTornadoExecutionPlan() {
        // Free the buffers, then close the plan. freeDeviceMemory() alone returns the device
        // allocations but leaves the plan — and its compiled code and task graphs — alive, so a
        // process that opens and closes several plans keeps accumulating them until the device
        // budget runs out. A session that has been closed must cost nothing.
        executionPlan.freeDeviceMemory();
        try {
            executionPlan.close();
        } catch (Exception e) {
            throw new IllegalStateException("failed to close the TornadoVM execution plan", e);
        }
    }
}
