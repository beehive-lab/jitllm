package org.beehive.jitllm.backend.tornado;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.pipeline.PipelineTransport;
import org.beehive.jitllm.backend.tornado.pipeline.StageOwnership;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.ForwardPlanFactory;
import org.beehive.jitllm.backend.tornado.plan.PackedRepack;
import org.beehive.jitllm.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.StagedForwardPlanComponents;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The forward pass split by layers across several devices: pipeline parallelism, as llama.cpp's
 * {@code --split-mode layer} does it.
 *
 * <p>Stage {@code s} runs on its own device with its own plan and holds a contiguous range of
 * layers, their weights and their key/value cache. The first stage also runs the embedding, the
 * last also the final norm and the vocabulary projection. Only the hidden state crosses between
 * devices, once per stage boundary per token (a chunk's hidden states, for a batched prefill):
 *
 * <pre>
 *   stage 0 (device 0): activation, layers [0, k1), send x ──┐
 *   stage 1 (device 1): receive x, layers [k1, k2), ...  ◀───┘ ... logits
 * </pre>
 *
 * <p>Every stage is built from the family's plan components ({@link StagedForwardPlanComponents})
 * on a state of its own, so this class names no family. A later stage's receive graph takes the
 * name of the activation graph it stands in for, which is what the family's first layer consumes.
 *
 * <p>For one sequence the stages run one after the other, so a split adds memory, not speed: a
 * model too large for one device runs on several. A batched prefill overlaps a prompt's chunks: the
 * first device takes the next chunk while the next device still works on the previous one.
 */
// @formatter:on
public final class TornadoVMMasterPlanPipeline implements BatchPrefillDecodePlan {

    private record Stage(
            int firstLayer,
            int endLayer,
            State state,
            TornadoDevice device,
            TornadoExecutionPlan plan) {}

    private final State state;
    private final Stage[] stages;
    private final TornadoExecutionPlan[] plans;
    private final PipelineTransport transport;
    private final StageOwnership ownership;
    private final DeviceSplit split;

    /**
     * Whether the prompt is prefilled in chunks: then each stage's plan holds a prefill program and
     * a decode program, and a step runs one of them; otherwise a step runs the whole plan.
     */
    private final boolean batched;

    private final int batchSize;

    /** How the first stage's batched prefill runs its projections, for the run configuration. */
    private String prefillProjections = "JIT kernels";

    /** Per stage, the graph indices of its prefill program, in order. Null unless batched. */
    private final int[][] prefillGraphs;

    /**
     * Per stage, the graph indices one generated token runs, in order: the decode program of a
     * batched plan, every graph but the repack ones of a single-token plan.
     */
    private final int[][] tokenGraphs;

    /**
     * Whether a token runs every stage's whole plan: a single-token plan whose stages carry no
     * graphs that run only once, at load.
     */
    private final boolean wholePlanTokens;

    public TornadoVMMasterPlanPipeline(
            State state, Model model, DeviceSplit split, MetricsSink sink) {
        this.state = state;
        this.split = split;
        var policy = state.executionPolicy();
        TornadoDevice[] devices = stageDevices(split);
        // The mode, every stage's device and the family's stages, before any transport or stage
        // exists.
        String architecture = model.architectureId().toString();
        PlanSupport.unsupported(
                        architecture,
                        model.weights().dataType(),
                        policy,
                        java.util.Arrays.stream(devices)
                                .map(device -> TornadoDevices.of(device).backend())
                                .toList(),
                        true)
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(
                                    PlanSupport.refusal(architecture, model, reason));
                        });
        this.batched = ExecutionMode.of(policy) == ExecutionMode.BATCH_PREFILL_DECODE;
        this.batchSize = batched ? policy.prefillBatchSize() : 1;
        ExecutionMode mode = batched ? ExecutionMode.BATCH_PREFILL_DECODE : ExecutionMode.STANDARD;

        int layers = model.configuration().numberOfLayers();
        int stageCount = devices.length;
        this.stages = new Stage[stageCount];
        this.plans = new TornadoExecutionPlan[stageCount];
        this.prefillGraphs = batched ? new int[stageCount][] : null;
        this.tokenGraphs = new int[stageCount][];
        var repack = new PackedRepack.Graphs[stageCount];

        int[] bounds = split.layerBounds(layers, ForwardPlanFactory.stageLayerAlignment(model));
        this.transport = PipelineTransport.create(split.transport(), devices);
        this.ownership = new StageOwnership(transport);
        try {
            for (int s = 0; s < stageCount; s++) {
                buildStage(
                        s,
                        bounds[s],
                        bounds[s + 1],
                        model,
                        state,
                        mode,
                        transport,
                        devices,
                        repack);
            }
            this.wholePlanTokens =
                    !batched && java.util.Arrays.stream(repack).allMatch(java.util.Objects::isNull);
            // Before the warm-up, which is the first execution of the graphs that read the
            // weights.
            for (int s = 0; s < stageCount; s++) {
                PackedRepack.run(plans[s], repack[s]);
            }
            forceCopyInReadOnlyData();
        } catch (RuntimeException e) {
            // Until the constructor returns the caller has nothing to close: whatever exists by
            // now — stages built or half-configured, the transport — is released here.
            throw ownership.releaseAfter(e);
        } catch (Error e) {
            throw ownership.releaseAfter(e);
        }
    }

    /**
     * One stage of the single-token plan: [embedding | receive], its layers, [send | logits]. A
     * step runs the whole plan.
     */
    private void addSingleTokenStage(
            PipelineTransport transport,
            int s,
            int first,
            int end,
            State stageState,
            StagedForwardPlanComponents components,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        if (s == 0) {
            add(components.singleTokenActivation(), graphs, scheduler);
        } else {
            TaskGraph receive = new TaskGraph("activationUpdate");
            transport.addTokenReceive(receive, s, stageState.workspace.wrapX, s - 1);
            graphs.add(receive.snapshot());
        }
        TransformerLayerTaskGraphs layers = components.singleTokenTransformerLayers(first, end);
        graphs.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);
        handOff(
                transport,
                s,
                layers.getLastFFNLayerTaskGraphID(),
                stageState,
                () -> components.singleTokenLogits(layers.getLastFFNLayerTaskGraphID()),
                "handoff_out",
                graphs,
                scheduler);
    }

    // @formatter:off
    /**
     * One stage of the batched plan: two programs in one plan, so they share the stage's weights
     * and key/value cache on its device.
     *
     * <pre>
     *   prefill: [prefillActivation | receive chunk], prefill layers [first, end), [send chunk]
     *   decode:  [decodeActivation  | receive x   ], decode layers  [first, end), [send x | logits]
     * </pre>
     *
     * <p>A later stage's {@code decodeActivation} only receives the hidden state; its first decode
     * layer takes the cache from the stage's last prefill layer.
     */
    // @formatter:on
    private void addBatchedStage(
            PipelineTransport transport,
            int s,
            int first,
            int end,
            State stageState,
            BatchPrefillDecodeForwardPlanComponents whole,
            StagedForwardPlanComponents staged,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        if (s == 0) {
            // The host stages a chunk the device does not decode in the session's carrier, which
            // the first stage then reads.
            if (state.workspace.wrapXBatch != null) {
                stageState.workspace.wrapXBatch = state.workspace.wrapXBatch;
            }
            add(whole.batchPrefillActivation(batchSize), graphs, scheduler);
            // The host stages each chunk in the session's state: its raw embedding rows where the
            // activation decodes them on the device, otherwise its decoded rows.
            if (stageState.workspace.embeddingQ8Batch != null) {
                state.workspace.embeddingQ8Batch = stageState.workspace.embeddingQ8Batch;
            }
        } else {
            TaskGraph receive = new TaskGraph("prefillActivation");
            transport.addReceive(receive, s, stageState.workspace.wrapXBatch, s - 1);
            graphs.add(receive.snapshot());
        }
        BatchPrefillTransformerLayerTaskGraphs prefill =
                staged.batchPrefillTransformerLayers(batchSize, first, end);
        graphs.addAll(prefill.getLayerImmutableTaskGraphs());
        prefill.updateGridScheduler(scheduler);
        if (s == 0) {
            prefillProjections = prefill.describeProjections();
        }
        String lastPrefill = prefill.getLastLayerTaskGraphID();
        if (s < stages.length - 1) {
            TaskGraph send = new TaskGraph("prefillHandoff");
            transport.addSend(send, s, lastPrefill, stageState.workspace.wrapXBatch, s + 1);
            graphs.add(send.snapshot());
        }
        prefillGraphs[s] = IntStream.range(0, graphs.size()).toArray();

        int decodeStart = graphs.size();
        if (s == 0) {
            add(whole.batchDecodeActivation(lastPrefill), graphs, scheduler);
        } else {
            // Only the hidden state: the first decode layer takes this stage's cache straight
            // from its last prefill layer.
            TaskGraph receive = new TaskGraph("decodeActivation");
            transport.addTokenReceive(receive, s, stageState.workspace.wrapX, s - 1);
            graphs.add(receive.snapshot());
        }
        TransformerLayerTaskGraphs decode = staged.batchDecodeTransformerLayers(first, end);
        graphs.addAll(decode.getFFNLayerImmutableTaskGraphs());
        decode.updateGridScheduler(scheduler);
        handOff(
                transport,
                s,
                decode.getLastFFNLayerTaskGraphID(),
                stageState,
                () -> whole.decodeLogits(decode.getLastFFNLayerTaskGraphID()),
                "decodeHandoff",
                graphs,
                scheduler);
        tokenGraphs[s] = IntStream.range(decodeStart, graphs.size()).toArray();
    }

    /** The end of a stage's token program: a send to the next stage, or on the last the logits. */
    private void handOff(
            PipelineTransport transport,
            int s,
            String lastLayer,
            State stageState,
            java.util.function.Supplier<AbstractLogitsTaskGraph> logits,
            String sendName,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        if (s < stages.length - 1) {
            TaskGraph send = new TaskGraph(sendName);
            transport.addTokenSend(send, s, lastLayer, stageState.workspace.wrapX, s + 1);
            graphs.add(send.snapshot());
        } else {
            AbstractLogitsTaskGraph last = logits.get();
            graphs.add(last.getImmutableTaskGraph());
            last.updateGridScheduler(scheduler);
        }
    }

    private static void add(
            ActivationTaskGraph activation,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        graphs.add(activation.getImmutableTaskGraph());
        activation.updateGridScheduler(scheduler);
    }

    /**
     * Builds stage {@code s}: its state, its layers and its plan, with the layers choosing their
     * kernels for the device the stage runs on rather than for the first one.
     */
    private void buildStage(
            int s,
            int first,
            int end,
            Model model,
            State state,
            ExecutionMode mode,
            PipelineTransport created,
            TornadoDevice[] devices,
            PackedRepack.Graphs[] repack) {
        TornadoDevices.building(
                TornadoDevices.of(devices[s]),
                () -> {
                    buildStageOnItsDevice(
                            s, first, end, model, state, mode, created, devices, repack);
                    return null;
                });
    }

    private void buildStageOnItsDevice(
            int s,
            int first,
            int end,
            Model model,
            State state,
            ExecutionMode mode,
            PipelineTransport created,
            TornadoDevice[] devices,
            PackedRepack.Graphs[] repack) {
        State stageState = ForwardPlanFactory.stageState(model, state, first, end, batchSize);
        StagedForwardPlanComponents components =
                ForwardPlanFactory.stageComponents(mode, stageState, model);
        if (s == 0) {
            // The token loop writes the embedding row into the session's state.
            stageState.workspace.embeddingX = state.workspace.embeddingX;
        }

        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        GridScheduler scheduler = new GridScheduler();
        if (batched) {
            addBatchedStage(
                    created,
                    s,
                    first,
                    end,
                    stageState,
                    (BatchPrefillDecodeForwardPlanComponents) components,
                    components,
                    graphs,
                    scheduler);
        } else {
            addSingleTokenStage(created, s, first, end, stageState, components, graphs, scheduler);
        }
        created.updateGridScheduler(s, scheduler);
        if (!batched) {
            tokenGraphs[s] = IntStream.range(0, graphs.size()).toArray();
        }
        // Graphs that repack this stage's packed weights, run once at load; no program
        // includes them.
        repack[s] = stageState.workspace.packedRepack.appendGraphs(graphs, scheduler);

        TornadoExecutionPlan plan =
                new TornadoExecutionPlan(graphs.toArray(new ImmutableTaskGraph[0]));
        // Owned before it is configured: configuring can fail, and the wrappers the calls below
        // return share this plan's executor and device buffers.
        plans[s] = plan;
        ownership.own(plan::freeDeviceMemory, plan::close);
        plan.withDevice(devices[s]).withGridScheduler(scheduler);
        if (TornadoVMMasterPlan.CUDA_GRAPHS && !batched && repack[s] == null) {
            plan.withCUDAGraph();
        } else if (TornadoVMMasterPlan.CUDA_GRAPHS) {
            // The token program only: a graph's capture is decided when its bytecode is
            // compiled, so it is enabled here, graph by graph, before precompilation. The
            // prefill program and the repack keep launching their kernels directly.
            for (int graph : tokenGraphs[s]) {
                plan.withGraph(graph).withCUDAGraph();
            }
            plan.withAllGraphs();
        }
        plan.withStagedTransfers();
        plan.withPreCompilation();
        stages[s] = new Stage(first, end, stageState, devices[s], plan);
    }

    /** The split's {@code backend:device} indices as TornadoVM devices, in stage order. */
    static TornadoDevice[] stageDevices(DeviceSplit split) {
        TornadoDevice[] devices = new TornadoDevice[split.stages()];
        for (int i = 0; i < devices.length; i++) {
            String[] bd = split.devices().get(i).split(":");
            devices[i] =
                    TornadoExecutionPlan.getDevice(
                            Integer.parseInt(bd[0]), Integer.parseInt(bd[1]));
        }
        return devices;
    }

    @Override
    public ExecutionInfo executionInfo() {
        StringBuilder description =
                new StringBuilder(
                        "pipeline ("
                                + split.transport().name().toLowerCase(java.util.Locale.ROOT)
                                + "):");
        for (Stage stage : stages) {
            description
                    .append(' ')
                    .append(stage.device().getDeviceName())
                    .append(" [")
                    .append(stage.firstLayer())
                    .append(',')
                    .append(stage.endLayer())
                    .append(')');
        }
        if (batched) {
            return PlanDiagnostics.describe(
                    state,
                    "batch-prefill-decode " + description,
                    batchSize,
                    prefillProjections,
                    "JIT kernels");
        }
        return PlanDiagnostics.describe(
                state,
                description.toString(),
                1,
                "JIT kernels (no tensor-core MMA)",
                "JIT kernels");
    }

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        return plans[0];
    }

    @Override
    public void forceCopyInReadOnlyData() {
        // One full step uploads every FIRST_EXECUTION buffer, the weights among them.
        for (Stage stage : stages) {
            stage.state().workspace.wrapX.clear();
            stage.state().resetPositionHolder();
            if (batched) {
                // No active rows: the warm-up prefill writes nothing to the cache.
                var workspace = stage.state().workspace;
                workspace.wrapXBatch.clear();
                workspace.batchStartPosHolder.init(0);
                workspace.batchStartPosHolder.set(CHUNK_SLOT, stage.state().kvSlot);
            }
        }
        if (batched) {
            transport.execute(plans, prefillGraphs, TornadoVMMasterPlan.CUDA_GRAPHS);
        }
        executeToken();
    }

    /** One generated token's step through every stage. */
    private void executeToken() {
        transport.executeToken(
                plans, wholePlanTokens ? null : tokenGraphs, TornadoVMMasterPlan.CUDA_GRAPHS);
    }

    @Override
    public void tornadoVMForwardBatchPrefill() {
        if (!batched) {
            throw new IllegalStateException("this pipeline plan was built without batched prefill");
        }
        // The host staged the chunk's start position, active rows and slot in the session state.
        int[] chunk = state.workspace.batchStartPosHolder.toHeapArray();
        for (Stage stage : stages) {
            stageControls(chunk, stage.state().workspace.batchStartPosHolder, stage.state().kvSlot);
        }
        transport.execute(plans, prefillGraphs, TornadoVMMasterPlan.CUDA_GRAPHS);
    }

    /** Index of the key/value slot in a chunk's controls, after its start position and rows. */
    static final int CHUNK_SLOT = 2;

    /**
     * A chunk's controls in a stage's own key/value address space: the session's start position and
     * active rows, and the stage's slot. A stage state owns its cache and block table, so the
     * session's slot in a shared pool is no address there.
     */
    static void stageControls(int[] chunk, IntArray holder, int stageSlot) {
        for (int i = 0; i < holder.getSize() && i < chunk.length; i++) {
            holder.set(i, i == CHUNK_SLOT ? stageSlot : chunk[i]);
        }
    }

    @Override
    public boolean overlapsPrefillChunks() {
        return batched && stages.length > 1;
    }

    // @formatter:off
    /**
     * The chunks run overlapped: every stage's thread walks them in order, the first staging each
     * chunk itself before running it, so the first device takes chunk {@code c + 1} while the next
     * still runs chunk {@code c}. Each stage keeps its own key/value state, and the first stage's
     * carrier is free again once its run of a chunk has returned — the send has completed by then —
     * so nothing two chunks share is in use by both.
     *
     * <p>A later stage reads a chunk's start position, active rows and slot from what the first
     * stage published when it staged that chunk, never from the session state, which by then may
     * hold a later chunk. A failure while staging or running on the first stage is passed to every
     * chunk the others still wait for.
     */
    // @formatter:on
    @Override
    public void tornadoVMForwardBatchPrefillChunks(
            int chunks, java.util.function.IntConsumer stageChunk) {
        if (!batched) {
            throw new IllegalStateException("this pipeline plan was built without batched prefill");
        }
        @SuppressWarnings("unchecked")
        java.util.concurrent.CompletableFuture<int[]>[] staged =
                new java.util.concurrent.CompletableFuture[chunks];
        for (int c = 0; c < chunks; c++) {
            staged[c] = new java.util.concurrent.CompletableFuture<>();
        }
        IntArray session = state.workspace.batchStartPosHolder;
        transport.executeChunks(
                plans,
                chunks,
                (stage, c, plan) -> {
                    int[] chunk;
                    if (stage == 0) {
                        try {
                            stageChunk.accept(c);
                        } catch (RuntimeException | Error e) {
                            for (var future : staged) {
                                future.completeExceptionally(e);
                            }
                            throw e;
                        }
                        chunk = new int[session.getSize()];
                        for (int i = 0; i < chunk.length; i++) {
                            chunk[i] = session.get(i);
                        }
                        staged[c].complete(chunk);
                    } else {
                        chunk = staged[c].join();
                    }
                    State stageState = stages[stage].state();
                    stageControls(
                            chunk, stageState.workspace.batchStartPosHolder, stageState.kvSlot);
                    try {
                        PipelineTransport.executeGraphs(
                                plan, prefillGraphs[stage], TornadoVMMasterPlan.CUDA_GRAPHS);
                    } catch (RuntimeException | Error e) {
                        if (stage == 0) {
                            for (var future : staged) {
                                future.completeExceptionally(e);
                            }
                        }
                        throw e;
                    }
                });
    }

    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        for (Stage stage : stages) {
            var workspace = stage.state().workspace;
            stage.state().setPosition(position);
            workspace.temp.clear();
            workspace.tempFFN.clear();
            workspace.tempLogits.clear();
            workspace.wrapLogits.clear();
        }
        executeToken();
        State last = stages[stages.length - 1].state();
        // A token sampled on the device is in the last stage's state; the loop reads the
        // session's.
        state.workspace.sampledToken.set(0, last.workspace.sampledToken.get(0));
        return last.workspace.wrapLogits;
    }

    @Override
    public void resetSequenceState() {
        // Zero each stage's recurrent state on the host and on its device. Graph 1 is every
        // stage's first layer graph (graph 0 is its activation or receive graph), which binds the
        // buffers. A family with only a key/value cache keeps none: the cache is overwritten
        // position by position.
        for (int s = 0; s < stages.length; s++) {
            State stageState = stages[s].state();
            stageState.resetSequenceState();
            Object[] recurrent = stageState.recurrentDeviceBuffers();
            if (recurrent.length > 0) {
                plans[s].withGraph(1).transferToDevice(recurrent);
            }
        }
    }

    @Override
    public void freeTornadoExecutionPlan() {
        ownership.release();
    }
}
