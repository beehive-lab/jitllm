package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek2TornadoWeights;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The single-token {@code deepseek2} layers, several to a task graph (a whole stage by default).
 *
 * <p>Per layer: the input norm and its quantization; {@code attn_q_a} and {@code attn_kv_a_mqa} as
 * packed integer dot products, their norms, {@code attn_q_b}; the rotation and the cache row; each
 * head's query absorbed through {@code attn_k_b}; attention over the latent rows in slices, merged;
 * the attended latents pushed out through {@code attn_v_b}; {@code attn_output} into the residual.
 * Then the feed-forward's norm and either the dense SwiGLU of a leading block or the routed experts
 * plus the shared expert, both added into the residual.
 *
 * <p>Q8_0 weights only: every projection here is a packed-integer kernel over Q8_0 blocks.
 */
// @formatter:on
public class DeepSeek2Layers
        extends AbstractTransformerLayerTaskGraphs<
                DeepSeek2TornadoWeights, DeepSeek2Configuration> {

    /** Layers per task graph; the whole stage at the default. */
    static final int LAYERS_PER_GRAPH = 64;

    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    private final DeepSeek2State ds;
    private final DeepSeek2LayerWeights<TornadoTensor> w;
    private final String activationGraphName;

    /** In the batched plan, the prefill layers whose arrays these share; otherwise null. */
    private final DeepSeek2BatchPrefillLayers prefill;

    /** Whether the state keeps its latent cache in half precision. */
    private final boolean fp16Cache;

    /** The latent cache in the representation the state allocated. */
    private Object cache() {
        return fp16Cache ? ds.workspace.wrapKeyCacheFP16 : ds.workspace.wrapKeyCache;
    }

    /** A router bias of zeros, for a file that carries none. */
    private final FloatArray zeroBias;

    /** Every task's worker grid, by qualified name, recorded as the graphs are built. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

    public DeepSeek2Layers(
            String taskGraphName,
            DeepSeek2State state,
            DeepSeek2TornadoWeights weights,
            DeepSeek2Configuration config,
            SchedulerType schedulerType) {
        this(
                taskGraphName,
                state,
                weights,
                config,
                schedulerType,
                "activationUpdate",
                0,
                config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * The first of them takes its activation from the graph named {@code activationGraphName}.
     */
    public DeepSeek2Layers(
            String taskGraphName,
            DeepSeek2State state,
            DeepSeek2TornadoWeights weights,
            DeepSeek2Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            int firstLayer,
            int endLayer) {
        this(
                taskGraphName,
                state,
                weights,
                config,
                schedulerType,
                activationGraphName,
                firstLayer,
                endLayer,
                null);
    }

    /**
     * The decode layers of a plan that also prefills in batches: they take the cache and every
     * weight the prefill graphs bind from those graphs, so the plan holds one copy of each.
     */
    public DeepSeek2Layers(
            String taskGraphName,
            DeepSeek2State state,
            DeepSeek2TornadoWeights weights,
            DeepSeek2Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            int firstLayer,
            int endLayer,
            DeepSeek2BatchPrefillLayers prefill) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.prefill = prefill;
        this.ds = state;
        this.fp16Cache = state.usesFp16KeyValueCache();
        this.w = weights.layers;
        this.activationGraphName = activationGraphName;
        this.zeroBias = new FloatArray(Math.max(1, config.expertCount()));
        restrictToLayers(firstLayer, endLayer);
        requireQ8(firstLayer, endLayer);
        setupFFNLayers();
    }

    private void requireQ8(int first, int end) {
        for (int l = first; l < end; l++) {
            TornadoTensor[] projections =
                    config.isDenseLayer(l)
                            ? new TornadoTensor[] {
                                w.qA()[l],
                                w.qB()[l],
                                w.kvAMqa()[l],
                                w.kB()[l],
                                w.vB()[l],
                                w.wo()[l],
                                w.ffnGate()[l],
                                w.ffnUp()[l],
                                w.ffnDown()[l]
                            }
                            : new TornadoTensor[] {
                                w.qA()[l],
                                w.qB()[l],
                                w.kvAMqa()[l],
                                w.kB()[l],
                                w.vB()[l],
                                w.wo()[l],
                                w.gateExperts()[l],
                                w.upExperts()[l],
                                w.downExperts()[l],
                                w.sharedGate()[l],
                                w.sharedUp()[l],
                                w.sharedDown()[l]
                            };
            for (TornadoTensor t : projections) {
                if (t.dataType() != DataType.Q8_0) {
                    throw new UnsupportedOperationException(
                            "deepseek2 on the accelerator reads Q8_0 projections only; block "
                                    + l
                                    + " holds "
                                    + t.dataType());
                }
            }
        }
    }

    // ── graphs ────────────────────────────────────────────────────────────────

    @Override
    protected void setupFFNLayers() {
        int layers = endLayer(config.numberOfLayers());
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        for (int first = firstLayer; first < layers; first += LAYERS_PER_GRAPH) {
            TaskGraph graph = new TaskGraph(layerGraphName(first));
            int last = Math.min(first + LAYERS_PER_GRAPH, layers) - 1;
            for (int layer = first; layer <= last; layer++) {
                appendLayer(graph, layer);
            }
            lastFFNLayerTaskGraphID = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
        ffnLayerITGs = List.copyOf(graphs);
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph graph = new TaskGraph(layerGraphName(layerIndex));
        appendLayer(graph, layerIndex);
        return graph;
    }

    private boolean firstLayerOfGraph(int layerIndex) {
        return (layerIndex - firstLayer) % LAYERS_PER_GRAPH == 0;
    }

    private boolean lastLayerOfGraph(int layerIndex) {
        return (layerIndex - firstLayer) % LAYERS_PER_GRAPH == LAYERS_PER_GRAPH - 1
                || layerIndex == endLayer(config.numberOfLayers()) - 1;
    }

    /** Adds a task and records its grid under its qualified name. */
    private void task(TaskGraph graph, int layer, String name, WorkerGrid grid, TaskAdder adder) {
        String task = "l" + layer + "_" + name;
        adder.add(task);
        grids.put(graph.getTaskGraphName() + "." + task, grid);
    }

    @FunctionalInterface
    private interface TaskAdder {
        void add(String taskName);
    }

    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    private void bindScratch(TaskGraph graph, int layerIndex) {
        var ws = ds.workspace;
        Object[] scratch = {
            context,
            ws.wrapXb,
            ws.wrapXbQuants,
            ws.wrapXbScales,
            ws.wrapXbSums,
            ws.wrapHb,
            ws.wrapQueryLatent,
            ws.wrapQ,
            ws.wrapCompressedKv,
            ws.wrapAbsorbedQuery,
            ws.wrapLatentOut,
            ws.wrapAtt,
            ws.wrapAttSplit,
            ws.wrapRouterLogits,
            ws.wrapSelectedExperts,
            ws.wrapRoutingWeights,
            ws.wrapSharedGate,
            ws.wrapMoeHidden,
            ws.wrapMoeHiddenQuants,
            ws.wrapMoeHiddenQScales,
            ws.wrapMoeHiddenQSums,
            zeroBias
        };
        Object[] shared = {
            cache(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
        if (layerIndex == firstLayer) {
            graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder);
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch);
            if (prefill != null) {
                graph.consumeFromDevice(prefill.getLastLayerTaskGraphID(), shared);
            } else {
                graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, shared);
            }
        } else {
            String predecessor = layerGraphName(layerIndex - 1);
            graph.consumeFromDevice(predecessor, scratch);
            graph.consumeFromDevice(predecessor, shared);
            graph.consumeFromDevice(predecessor, ws.positionHolder);
        }
    }

    private void bindWeights(TaskGraph graph, int l) {
        if (prefill != null) {
            graph.consumeFromDevice(
                    prefill.graphName(l),
                    DeepSeek2BatchPrefillLayers.sharedWeights(w, config, l).toArray());
            // The prefill reads these in other forms: padded, and in FP16.
            graph.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    deviceArray(w.kvAMqa()[l]),
                    deviceArray(w.kB()[l]),
                    deviceArray(w.vB()[l]));
            return;
        }
        List<Object> arrays = new ArrayList<>();
        for (TornadoTensor t :
                new TornadoTensor[] {
                    w.attnNorm()[l], w.qA()[l], w.qANorm()[l], w.qB()[l], w.kvAMqa()[l],
                    w.kvANorm()[l], w.kB()[l], w.vB()[l], w.wo()[l], w.ffnNorm()[l]
                }) {
            arrays.add(deviceArray(t));
        }
        if (config.isDenseLayer(l)) {
            arrays.add(deviceArray(w.ffnGate()[l]));
            arrays.add(deviceArray(w.ffnUp()[l]));
            arrays.add(deviceArray(w.ffnDown()[l]));
        } else {
            arrays.add(deviceArray(w.router()[l]));
            if (w.routerBias()[l] != null) {
                arrays.add(deviceArray(w.routerBias()[l]));
            }
            arrays.add(deviceArray(w.gateExperts()[l]));
            arrays.add(deviceArray(w.upExperts()[l]));
            arrays.add(deviceArray(w.downExperts()[l]));
            arrays.add(deviceArray(w.sharedGate()[l]));
            arrays.add(deviceArray(w.sharedUp()[l]));
            arrays.add(deviceArray(w.sharedDown()[l]));
        }
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, arrays.toArray());
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var ws = ds.workspace;
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int rank = config.kvLoraRank();
        final int rope = config.ropeDim();
        final int nope = config.nopeDim();
        final int qHead = config.queryHeadDim();
        final int qLora = config.qLoraRank();
        final int valueHead = config.valueHeadDim();
        final int attnOut = config.attentionOutputInputDim();
        final float eps = config.rmsNormEps();
        final int splits = DeepSeek2Configuration.DECODE_ATTENTION_SPLITS;
        final int cacheOffset = ds.cacheLayerOffset(l);

        if (firstLayerOfGraph(l)) {
            String producer = l == firstLayer ? activationGraphName : layerGraphName(l - 1);
            g.consumeFromDevice(producer, ws.wrapX);
            bindScratch(g, l);
        }
        bindWeights(g, l);

        // ── attention ──
        normQuantize(g, l, "attn_norm", ws.wrapX, ws.wrapXb, w.attnNorm()[l], dim, eps);
        task(
                g,
                l,
                "q_kv_a",
                lanes((qLora + rank + rope) * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::twoProjectionsQ8_0,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                w.qA()[l].asByteArray(),
                                ws.wrapQueryLatent,
                                w.kvAMqa()[l].asByteArray(),
                                ws.wrapCompressedKv,
                                dim,
                                qLora,
                                rank + rope));
        task(
                g,
                l,
                "latent_norms",
                lanes(2 * DeepSeek2Kernels.GROUP, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::latentNorms,
                                context,
                                ws.wrapQueryLatent,
                                w.qANorm()[l].asFloatArray(),
                                ws.wrapCompressedKv,
                                w.kvANorm()[l].asFloatArray(),
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums,
                                qLora,
                                rank,
                                eps));
        project(g, l, "q_b", w.qB()[l], ws.wrapQ, qLora, config.queryDim());

        int ropeLanes = heads * rope / 2 + rope / 2 + rank;
        task(
                g,
                l,
                "rope_cache",
                lanes(ropeLanes, 128),
                t -> {
                    if (fp16Cache) {
                        g.task(
                                t,
                                DeepSeek2Kernels::ropeAndCacheFP16,
                                context,
                                ws.positionHolder,
                                ws.wrapQ,
                                ws.wrapCompressedKv,
                                weights.freq_cis_realFlat.asFloatArray(),
                                weights.freq_cis_imagFlat.asFloatArray(),
                                ws.wrapKeyCacheFP16,
                                heads,
                                qHead,
                                nope,
                                rope,
                                rank,
                                cacheOffset);
                    } else {
                        g.task(
                                t,
                                DeepSeek2Kernels::ropeAndCache,
                                context,
                                ws.positionHolder,
                                ws.wrapQ,
                                ws.wrapCompressedKv,
                                weights.freq_cis_realFlat.asFloatArray(),
                                weights.freq_cis_imagFlat.asFloatArray(),
                                ws.wrapKeyCache,
                                heads,
                                qHead,
                                nope,
                                rope,
                                rank,
                                cacheOffset);
                    }
                });
        int absorbWarps = heads * rank + (heads * rope + 31) / 32;
        task(
                g,
                l,
                "absorb",
                lanes(absorbWarps * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::absorbQuery,
                                context,
                                w.kB()[l].asByteArray(),
                                ws.wrapQ,
                                ws.wrapAbsorbedQuery,
                                heads,
                                rank,
                                nope,
                                rope,
                                qHead));
        task(
                g,
                l,
                "attention",
                lanes(heads * splits * DeepSeek2Kernels.GROUP, DeepSeek2Kernels.GROUP),
                t -> {
                    if (fp16Cache) {
                        g.task(
                                t,
                                DeepSeek2Kernels::attentionSplitFP16,
                                context,
                                ws.positionHolder,
                                ws.wrapAbsorbedQuery,
                                ws.wrapKeyCacheFP16,
                                ws.wrapAtt,
                                ws.wrapAttSplit,
                                heads,
                                rank,
                                rope,
                                cacheOffset,
                                config.contextLength(),
                                splits,
                                config.attentionScale());
                    } else {
                        g.task(
                                t,
                                DeepSeek2Kernels::attentionSplit,
                                context,
                                ws.positionHolder,
                                ws.wrapAbsorbedQuery,
                                ws.wrapKeyCache,
                                ws.wrapAtt,
                                ws.wrapAttSplit,
                                heads,
                                rank,
                                rope,
                                cacheOffset,
                                config.contextLength(),
                                splits,
                                config.attentionScale());
                    }
                });
        task(
                g,
                l,
                "attention_combine",
                lanes(heads * rank, 256),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::combineSplits,
                                context,
                                ws.wrapAttSplit,
                                ws.wrapLatentOut,
                                heads,
                                rank,
                                splits));
        task(
                g,
                l,
                "decompress",
                lanes(heads * valueHead * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::decompressLatent,
                                context,
                                w.vB()[l].asByteArray(),
                                ws.wrapLatentOut,
                                ws.wrapXb,
                                heads,
                                valueHead,
                                rank));
        quantize(g, l, "attn_out_q8", ws.wrapXb, attnOut);
        projectResidual(g, l, "wo", w.wo()[l], attnOut, dim);

        // ── feed-forward ──
        normQuantize(g, l, "ffn_norm", ws.wrapX, ws.wrapXb, w.ffnNorm()[l], dim, eps);
        if (config.isDenseLayer(l)) {
            int hidden = config.hiddenDim();
            task(
                    g,
                    l,
                    "ffn_gate_up",
                    lanes(hidden * projectionLocal(dim), projectionLocal(dim)),
                    t ->
                            g.task(
                                    t,
                                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                                    context,
                                    ws.wrapXbQuants,
                                    ws.wrapXbScales,
                                    ws.wrapHb,
                                    w.ffnGate()[l].asByteArray(),
                                    w.ffnUp()[l].asByteArray(),
                                    dim,
                                    hidden,
                                    projectionLocal(dim)));
            quantize(g, l, "ffn_hidden_q8", ws.wrapHb, hidden);
            projectResidual(g, l, "ffn_down", w.ffnDown()[l], hidden, dim);
        } else {
            mixtureOfExperts(g, l);
        }

        if (lastLayerOfGraph(l)) {
            g.persistOnDevice(ws.wrapX, cache());
        }
    }

    private void mixtureOfExperts(TaskGraph g, int l) {
        var ws = ds.workspace;
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        FloatArray bias = w.routerBias()[l] != null ? w.routerBias()[l].asFloatArray() : zeroBias;
        task(
                g,
                l,
                "router",
                lanes(experts * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::routerLogits,
                                context,
                                ws.wrapXb,
                                w.router()[l].asFloatArray(),
                                ws.wrapRouterLogits,
                                dim,
                                experts));
        task(
                g,
                l,
                "router_topk",
                lanes(32, 32),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::routerTopKSigmoid,
                                context,
                                ws.wrapRouterLogits,
                                bias,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                ws.wrapSharedGate,
                                experts,
                                used,
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale()));
        task(
                g,
                l,
                "experts_gate_up",
                lanes((used * hidden + shared) * 32, Qwen35MoeKernels.LOCAL),
                t ->
                        g.task(
                                t,
                                Qwen35MoeKernels::expertsGateUpQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapSelectedExperts,
                                w.gateExperts()[l].asByteArray(),
                                w.upExperts()[l].asByteArray(),
                                w.sharedGate()[l].asByteArray(),
                                w.sharedUp()[l].asByteArray(),
                                ws.wrapMoeHidden,
                                dim,
                                hidden,
                                shared,
                                used));
        int moeHidden = used * hidden + shared;
        task(
                g,
                l,
                "experts_hidden_q8",
                lanes(moeHidden, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                ws.wrapMoeHidden,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapMoeHiddenQSums));
        task(
                g,
                l,
                "experts_down",
                lanes(dim * 32, Qwen35MoeKernels.LOCAL),
                t ->
                        g.task(
                                t,
                                Qwen35MoeKernels::expertsDownResidualQ8_0DP4A,
                                context,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                ws.wrapSharedGate,
                                w.downExperts()[l].asByteArray(),
                                w.sharedDown()[l].asByteArray(),
                                ws.wrapX,
                                dim,
                                hidden,
                                shared,
                                used));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /** {@link #rmsNorm} and the quantization of its output for the projections after it. */
    private void normQuantize(
            TaskGraph g,
            int l,
            String name,
            FloatArray in,
            FloatArray out,
            TornadoTensor weight,
            int n,
            float eps) {
        var ws = ds.workspace;
        task(
                g,
                l,
                name,
                lanes(n, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::rmsNormQuantize,
                                context,
                                in,
                                out,
                                weight.asFloatArray(),
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums,
                                n,
                                eps));
    }

    /** {@code x}'s first {@code n} values in Q8 blocks, into the shared quantized activation. */
    private void quantize(TaskGraph g, int l, String name, FloatArray x, int n) {
        var ws = ds.workspace;
        task(
                g,
                l,
                name,
                lanes(n, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                x,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums));
    }

    /** {@code out = w . x}, the activation quantized; {@code n} inputs, {@code d} outputs. */
    private void project(
            TaskGraph g, int l, String name, TornadoTensor weight, FloatArray out, int n, int d) {
        var ws = ds.workspace;
        int local = projectionLocal(n);
        task(
                g,
                l,
                name,
                lanes(d * local, local),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                out,
                                weight.asByteArray(),
                                n,
                                d,
                                local));
    }

    /** {@code x += w . activation}, the activation quantized. */
    private void projectResidual(
            TaskGraph g, int l, String name, TornadoTensor weight, int n, int d) {
        var ws = ds.workspace;
        int local = projectionLocal(n);
        task(
                g,
                l,
                name,
                lanes(d * local, local),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ8_0DP4A
                                        ::matrixVectorGenericWithResidualQ8_0DP4A,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapX,
                                weight.asByteArray(),
                                n,
                                d,
                                local));
    }

    /** Lanes per output row: as many as the row has blocks, between one and four warps. */
    private static int projectionLocal(int n) {
        int blocks = n / 32;
        return blocks >= 128 ? 128 : blocks >= 64 ? 64 : 32;
    }

    /** {@code count} lanes rounded up to whole workgroups of {@code local}. */
    private static WorkerGrid lanes(int count, int local) {
        int global = (count + local - 1) / local * local;
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
        return scheduler;
    }
}
