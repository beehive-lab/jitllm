package org.beehive.jllm.backend.tornado;

import org.beehive.jllm.auxiliary.RunMetrics;
import org.beehive.jllm.backend.tornado.plan.BatchPrefillDecodeForwardPlan;
import org.beehive.jllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jllm.backend.tornado.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * GPU execution plan for batched prefill + single-token decode.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all TaskGraphs for batched prefill and
 * single-token decode phases:
 *
 * <p>TaskGraph layout (2N+3 TaskGraphs total):
 *
 * <pre>
 *   [0]         batchPrefillActivation  B×dim embeddings → FP32 wrapXBatch
 *   [1.N]      batch-prefill layers    B tokens, all transformer ops
 *   [N+1]       decodeActivation        single-token embedding → FP32 + KV-cache pass-through
 *   [N+2.2N+1] decode layers           single-token, standard kernels
 *   [2N+2]      logits
 * </pre>
 */
// @formatter:on
public class TornadoVMMasterPlanBatchPrefillDecode implements TornadoVMMasterPlan {

    /**
     * Rule 16: library code routes its output through the platform logger, so an embedder can
     * silence or redirect it. Reached only under {@code jllm.EnableTimingForTornadoVMInit}.
     */
    private static final System.Logger LOGGER =
            System.getLogger(TornadoVMMasterPlanBatchPrefillDecode.class.getName());

    @Override
    public org.beehive.jllm.runtime.backend.ExecutionInfo executionInfo() {
        var layers = batchPrefillDecodeForwardPlan.getBatchPrefillLayers();
        return PlanDiagnostics.describe(
                state,
                "batch-prefill-decode",
                state.executionPolicy().prefillBatchSize(),
                layers.describeProjections(),
                layers.describeAttention(),
                layers.describeNativeLibraries());
    }

    private final State state;
    private final Model model;
    private final Configuration config;

    BatchPrefillDecodeForwardPlan batchPrefillDecodeForwardPlan;
    BatchPrefillDecodeForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    /**
     * Rule 17 seam. Costs one boolean test per execution while the sink is the disabled default.
     */
    private final TornadoMetricsReporter metrics;

    // ── Construction ─────────────────────────────────────────────────────────
    TornadoVMMasterPlanBatchPrefillDecode(State initialState, Model model, MetricsSink sink) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            LOGGER.log(System.Logger.Level.INFO, "Starting TornadoVM initialization...");
        }

        this.state = initialState;
        this.model = model;
        this.config = model.configuration();
        this.metrics = new TornadoMetricsReporter(sink);

        long startTime = System.nanoTime();
        this.executionPlan = createExecutionPlan();
        metrics.enableOn(executionPlan);
        long planCreationTime = System.nanoTime();
        if (ENABLE_TORNADOVM_INIT_TIME) {
            // The device the plan was built for and the kernel families it was granted, then
            // the batched-prefill path the layer builder actually chose from them; the same
            // facts the tests read off the plan, reported where a user can see them.
            var device = org.beehive.jllm.backend.tornado.device.TornadoDevices.current();
            LOGGER.log(
                    System.Logger.Level.INFO,
                    "TornadoVM device: {0} ({1}) capabilities {2}",
                    device.displayName(),
                    device.id().backend(),
                    device.capabilities());
            var layers = batchPrefillDecodeForwardPlan.getBatchPrefillLayers();
            if (layers
                    instanceof org.beehive.jllm.backend.tornado.layers.Qwen35BatchPrefillLayers q) {
                LOGGER.log(
                        System.Logger.Level.INFO,
                        "qwen35 batched prefill: {0}",
                        q.describeDispatch());
            }
        }

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

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        DataType weightType = model.weights().dataType();
        this.batchPrefillDecodeForwardPlan =
                ForwardPlanFactory.createBatchPrefillDecode(weightType, state, model);
        this.taskGraphLayout = batchPrefillDecodeForwardPlan.getTaskGraphLayout();
        var taskGraphs = batchPrefillDecodeForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    @Override
    public void resetSequenceState() {
        TornadoVMMasterPlan.resetSequenceState(
                executionPlan, state, taskGraphLayout.batchLayerIdx(0));
    }

    // @formatter:off
    @Override
    public void forceCopyInReadOnlyData() {
        state.workspace.wrapX.clear();
        state.resetPositionHolder();
        state.workspace.wrapXBatch.clear();
        state.workspace.batchStartPosHolder.init(0);
        if (state.workspace.batchStartPosHolder.getSize() > 2) {
            state.workspace.batchStartPosHolder.set(2, state.kvSlot);
        }

        for (int i = 0; i <= taskGraphLayout.logitsIdx(); i++) {
            var g =
                    executionPlan
                            .withGraph(i)
                            .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                g.withCUDAGraph();
            }
            metrics.report(g.execute());
        }
    }

    // @formatter:on

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * Batch prefill: runs graphs 0.N (activation + N layers), skips logits. Caller is responsible
     * for copying batch embeddings into state before calling this.
     */
    // @formatter:off
    public void tornadoVMForwardBatchPrefill() {
        var batchAct =
                executionPlan
                        .withGraph(taskGraphLayout.batchActivationIdx())
                        .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            batchAct.withCUDAGraph();
        }
        metrics.report(batchAct.execute());

        // Over batch-prefill layer GRAPHS, not layers: a family may hold several layers in one
        // graph, and the layout is what knows how many that leaves. Identical to a loop over
        // layers whenever it builds one graph each.
        for (int g = 0; g < taskGraphLayout.batchLayerGraphs(); g++) {
            var batchLayer =
                    executionPlan
                            .withGraph(taskGraphLayout.batchLayerIdx(g))
                            .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                batchLayer.withCUDAGraph();
            }
            metrics.report(batchLayer.execute());
        }
    }

    /** Whether this plan carries a fallback batch-prefill family. */
    public boolean hasBatchPrefillFallback() {
        return taskGraphLayout.fallbackLayerGraphs() > 0;
    }

    /**
     * Batch prefill through the fallback family: the same layers, with the attention implementation
     * that handles a chunk whose queries do not start at position 0. The caller has already put
     * this chunk's embeddings and its start position into state, exactly as for the primary path.
     */
    // @formatter:off
    public void tornadoVMForwardBatchPrefillFallback() {
        var batchAct =
                executionPlan
                        .withGraph(taskGraphLayout.batchActivationIdx())
                        .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            batchAct.withCUDAGraph();
        }
        metrics.report(batchAct.execute());

        for (int g = 0; g < taskGraphLayout.fallbackLayerGraphs(); g++) {
            var layer =
                    executionPlan
                            .withGraph(taskGraphLayout.fallbackLayerIdx(g))
                            .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                layer.withCUDAGraph();
            }
            metrics.report(layer.execute());
        }
    }

    // @formatter:on

    /**
     * Single-token decode: runs graphs N+1.2N+2 (activation + N layers + logits). Caller is
     * responsible for copying the decode embedding into state before calling this.
     *
     * @param position sequence position
     * @return logits array for sampling
     */
    // @formatter:off
    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        state.setPosition(position);
        state.workspace.temp.clear();
        state.workspace.tempFFN.clear();

        var decodeAct =
                executionPlan
                        .withGraph(taskGraphLayout.decodeActivationIdx())
                        .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) {
            decodeAct.withCUDAGraph();
        }
        metrics.report(decodeAct.execute());

        // Over decode layer GRAPHS, not layers: a family may hold several layers in one graph,
        // and the layout is what knows how many graphs that leaves. Identical to a loop over
        // layers for every family that builds one graph each.
        for (int g = 0; g < taskGraphLayout.decodeLayerGraphs(); g++) {
            var decodeLayer =
                    executionPlan
                            .withGraph(taskGraphLayout.decodeLayerGraphIdx(g))
                            .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) {
                decodeLayer.withCUDAGraph();
            }
            metrics.report(decodeLayer.execute());
        }

        state.workspace.tempLogits.clear();
        state.workspace.wrapLogits.clear();

        var logits =
                executionPlan
                        .withGraph(taskGraphLayout.logitsIdx())
                        .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
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
