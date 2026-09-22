package org.beehive.jllm.backend.tornado;

import org.beehive.jllm.auxiliary.RunMetrics;
import org.beehive.jllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jllm.backend.tornado.plan.SingleTokenForwardPlan;
import org.beehive.jllm.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Standard (single-token) GPU execution plan.
 *
 * <p>Processes one token at a time through preprocessing + N transformer layers + logits
 * projection.
 */
public class TornadoVMMasterPlanSingleToken implements TornadoVMMasterPlan {

    /**
     * Rule 16: library code routes its output through the platform logger, so an embedder can
     * silence or redirect it. Reached only under {@code jllm.EnableTimingForTornadoVMInit}.
     */
    private static final System.Logger LOGGER =
            System.getLogger(TornadoVMMasterPlanSingleToken.class.getName());

    private final State state;
    private final Model model;
    private final Configuration config;

    SingleTokenForwardPlan tornadoVMForwardPlan;
    SingleTokenForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    /**
     * Rule 17 seam. Costs one boolean test per execution while the sink is the disabled default.
     */
    private final TornadoMetricsReporter metrics;

    public TornadoVMMasterPlanSingleToken(State state, Model model, MetricsSink sink) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            LOGGER.log(System.Logger.Level.INFO, "Starting TornadoVM initialization...");
        }

        this.state = state;
        this.model = model;
        this.config = model.configuration();
        this.metrics = new TornadoMetricsReporter(sink);

        long startTime = System.nanoTime();
        this.executionPlan = createExecutionPlan();
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

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        DataType weightType = model.weights().dataType();
        this.tornadoVMForwardPlan = ForwardPlanFactory.createSingleToken(weightType, state, model);
        this.taskGraphLayout = tornadoVMForwardPlan.getTaskGraphLayout();
        var taskGraphs = tornadoVMForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    // @formatter:off
    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        var preGraph =
                executionPlan
                        .withGraph(taskGraphLayout.activationIdx())
                        .withGridScheduler(tornadoVMForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            preGraph.withCUDAGraph();
        }
        metrics.report(preGraph.execute());

        state.setPosition(position);
        state.workspace.temp.clear();
        state.workspace.tempFFN.clear();

        // By graph, not by layer: a family may hold several layers in one graph.
        for (int layer = 0; layer < taskGraphLayout.N(); layer++) {
            metrics.report(
                    executionPlan
                            .withGraph(taskGraphLayout.layerIdx(layer))
                            .withGridScheduler(tornadoVMForwardPlan.getGridScheduler())
                            .execute());
        }
        state.workspace.tempLogits.clear();
        state.workspace.wrapLogits.clear();
        var logitsGraph =
                executionPlan
                        .withGraph(taskGraphLayout.logitsIdx())
                        .withGridScheduler(tornadoVMForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            logitsGraph.withCUDAGraph();
        }
        metrics.report(logitsGraph.execute());

        return state.workspace.wrapLogits;
    }

    // @formatter:on

    @Override
    public void resetSequenceState() {
        TornadoVMMasterPlan.resetSequenceState(executionPlan, state, taskGraphLayout.layerIdx(0));
    }

    // @formatter:off
    @Override
    public void forceCopyInReadOnlyData() {
        state.workspace.wrapX.clear();
        state.resetPositionHolder();

        metrics.report(
                executionPlan
                        .withGraph(taskGraphLayout.activationIdx())
                        .withGridScheduler(tornadoVMForwardPlan.getGridScheduler())
                        .execute());

        // By graph, not by layer: a family may hold several layers in one graph.
        for (int layer = 0; layer < taskGraphLayout.N(); layer++) {
            metrics.report(
                    executionPlan
                            .withGraph(taskGraphLayout.layerIdx(layer))
                            .withGridScheduler(tornadoVMForwardPlan.getGridScheduler())
                            .execute());
        }

        metrics.report(
                executionPlan
                        .withGraph(taskGraphLayout.logitsIdx())
                        .withGridScheduler(tornadoVMForwardPlan.getGridScheduler())
                        .execute());
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
