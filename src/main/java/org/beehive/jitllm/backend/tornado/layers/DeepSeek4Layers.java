package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2Kernels;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek4Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.DeepSeek4DeviceArrays;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek4TornadoWeights;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The single-token {@code deepseek4} layers (DeepSeek-V4), the whole trunk in one task graph.
 *
 * <p>The first layer copies the embedding into every residual stream. Each layer then runs two
 * sublayers, each inside a hyper-connection — the coefficients from the streams, the collapse, the
 * sublayer's norm and quantization, the sublayer into {@code branch}, the expansion back:
 *
 * <ul>
 *   <li>attention: {@code q_a} and {@code kv}; with a compression ratio, the compressor's
 *       projections and the pooling of a completed block; with an indexer, its compressor, head
 *       weights, query and block selection; the latent norms, {@code q_b}, the heads' norm and
 *       rotation, the key/value into the window; attention with sinks; the grouped {@code
 *       output_a}; {@code output_b};
 *   <li>the mixture of experts: square-root-softplus routing, from the token's table in the
 *       hash-routed layers; the routed and shared experts' clamped SwiGLU and their down
 *       projection.
 * </ul>
 *
 * <p>The last layer collapses the streams through the learned mHC head into {@code wrapX} for the
 * logits. Q8_0 projections; the routed experts all Q8_0 or all Q4_0, in at most two stacks; the
 * compressors, the indexer's head weights and the router in F32.
 */
// @formatter:on
public class DeepSeek4Layers
        extends AbstractTransformerLayerTaskGraphs<
                DeepSeek4TornadoWeights, DeepSeek4Configuration> {

    private final DeepSeek4State ds;
    private final DeepSeek4LayerWeights<TornadoTensor> w;
    private final String activationGraphName;

    /** Every task's worker grid, by qualified name, recorded as the graph is built. */
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();

    /** Each layer's attention and compressor shapes ({@link DeepSeek4Kernels}' index constants). */
    private final IntArray[] attentionShapes;

    private final IntArray[] compressorShapes;
    private final IntArray[] indexerShapes;

    /** In the batched plan, the prefill layers whose arrays these share; otherwise null. */
    private final DeepSeek4BatchPrefillLayers prefill;

    /** The layers this graph runs: {@code [firstLayer, endLayer)}. */
    private final int firstLayer;

    private final int endLayer;

    public DeepSeek4Layers(
            DeepSeek4State state,
            DeepSeek4TornadoWeights weights,
            DeepSeek4Configuration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        this(state, weights, config, schedulerType, activationGraphName, null);
    }

    /**
     * The decode layers of a plan that also prefills in batches: they take the window, the
     * compressors' entries, the compressed keys and every weight from the prefill graph, so the
     * plan holds one copy of each.
     */
    public DeepSeek4Layers(
            DeepSeek4State state,
            DeepSeek4TornadoWeights weights,
            DeepSeek4Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            DeepSeek4BatchPrefillLayers prefill) {
        this(
                state,
                weights,
                config,
                schedulerType,
                activationGraphName,
                prefill,
                0,
                config.numberOfLayers());
    }

    // @formatter:off
    /**
     * The layers {@code [firstLayer, endLayer)}, one stage of a model split across devices. The
     * first stage copies the embedding into the residual streams; a later one takes the streams
     * from {@code activationGraphName}, the graph that received them. The last stage collapses the
     * streams through the mHC head into the hidden state the logits read; an earlier one leaves the
     * streams on its device for the send to the next stage. {@code prefill}, when not null, holds
     * the same layers.
     */
    // @formatter:on
    public DeepSeek4Layers(
            DeepSeek4State state,
            DeepSeek4TornadoWeights weights,
            DeepSeek4Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            DeepSeek4BatchPrefillLayers prefill,
            int firstLayer,
            int endLayer) {
        super(state, weights, config, schedulerType);
        this.prefill = prefill;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.ds = state;
        this.w = weights.layers;
        this.activationGraphName = activationGraphName;
        int layers = config.numberOfLayers();
        this.attentionShapes = new IntArray[layers];
        this.compressorShapes = new IntArray[layers];
        this.indexerShapes = new IntArray[layers];
        requireSupported();
        buildShapes();
        setupFFNLayers();
    }

    private void requireSupported() {
        if (config.headDim() > DeepSeek4Kernels.MAX_WIDTH
                || config.indexerHeadDim() > DeepSeek4Kernels.MAX_WIDTH) {
            throw new UnsupportedOperationException(
                    "deepseek4 on the accelerator pools compressed keys of at most "
                            + DeepSeek4Kernels.MAX_WIDTH
                            + " values");
        }
        if (config.expertCount() > DeepSeek4Kernels.MAX_EXPERTS) {
            throw new UnsupportedOperationException(
                    "deepseek4 on the accelerator routes over at most "
                            + DeepSeek4Kernels.MAX_EXPERTS
                            + " experts");
        }
        for (int l = firstLayer; l < endLayer; l++) {
            var a = w.attention();
            var f = w.ffn();
            List<TornadoTensor> q8 =
                    new ArrayList<>(
                            List.of(
                                    a.qA()[l],
                                    a.qB()[l],
                                    a.kv()[l],
                                    a.outputA()[l],
                                    a.outputB()[l],
                                    f.sharedGate()[l],
                                    f.sharedUp()[l],
                                    f.sharedDown()[l]));
            List<TornadoTensor> f32 = new ArrayList<>(List.of(f.router()[l]));
            if (config.ratio(l) != 0) {
                f32.add(w.compressor().kv()[l]);
                f32.add(w.compressor().gate()[l]);
            }
            if (config.hasIndexer(l)) {
                var x = w.indexer();
                q8.add(x.qB()[l]);
                f32.addAll(List.of(x.headWeights()[l], x.compressorKv()[l], x.compressorGate()[l]));
            }
            for (TornadoTensor t : q8) {
                require(t.dataType() == DataType.Q8_0, l, "Q8_0 projections", t);
            }
            for (TornadoTensor t : f32) {
                require(t.dataType() == DataType.F32, l, "F32 compressors and router", t);
            }
            List<TornadoTensor> routed = new ArrayList<>(f.gateExperts().get(l));
            routed.addAll(f.upExperts().get(l));
            routed.addAll(f.downExperts().get(l));
            if (f.gateExperts().get(l).size() > 2) {
                throw new UnsupportedOperationException(
                        "deepseek4 on the accelerator reads the routed experts from at most two"
                                + " stacks; layer "
                                + l
                                + " holds "
                                + f.gateExperts().get(l).size());
            }
            DataType routedType = routed.get(0).dataType();
            for (TornadoTensor t : routed) {
                require(
                        t.dataType() == routedType
                                && (routedType == DataType.Q8_0 || routedType == DataType.Q4_0),
                        l,
                        "routed experts all in Q8_0 or all in Q4_0",
                        t);
            }
        }
    }

    private static void require(boolean ok, int l, String what, TornadoTensor t) {
        if (!ok) {
            throw new UnsupportedOperationException(
                    "deepseek4 on the accelerator reads "
                            + what
                            + "; layer "
                            + l
                            + " holds "
                            + t.dataType());
        }
    }

    private void buildShapes() {
        for (int l = firstLayer; l < endLayer; l++) {
            int r = config.ratio(l);
            IntArray s = new IntArray(DeepSeek4Kernels.SHAPE_SIZE);
            s.set(DeepSeek4Kernels.SHAPE_HEADS, config.numberOfHeads());
            s.set(DeepSeek4Kernels.SHAPE_HEAD_DIM, config.headDim());
            s.set(DeepSeek4Kernels.SHAPE_ROPE_DIM, config.ropeDim());
            s.set(DeepSeek4Kernels.SHAPE_WINDOW, config.slidingWindow());
            s.set(DeepSeek4Kernels.SHAPE_WINDOW_OFFSET, ds.windowOffset(config, l));
            s.set(DeepSeek4Kernels.SHAPE_COMPRESSED_OFFSET, r == 0 ? 0 : ds.compressedOffset(l));
            s.set(DeepSeek4Kernels.SHAPE_STRIDE, config.maxAttended());
            s.set(DeepSeek4Kernels.SHAPE_RATIO, r);
            s.set(DeepSeek4Kernels.SHAPE_USE_SELECTION, config.hasIndexer(l) ? 1 : 0);
            s.set(
                    DeepSeek4Kernels.SHAPE_SELECTION_CAPACITY,
                    gs().workspace.deepseek4.selected.getSize());
            attentionShapes[l] = s;
            if (r != 0) {
                compressorShapes[l] =
                        compressorShape(
                                r,
                                DeepSeek4Configuration.overlap(r),
                                config.headDim(),
                                ds.pendingOffset(l),
                                ds.compressedOffset(l));
            }
            if (config.hasIndexer(l)) {
                indexerShapes[l] =
                        compressorShape(
                                r,
                                2,
                                config.indexerHeadDim(),
                                ds.indexerPendingOffset(l),
                                ds.indexerKeyOffset(l));
            }
        }
    }

    private IntArray compressorShape(int r, int o, int width, int pending, int cache) {
        IntArray s = new IntArray(DeepSeek4Kernels.C_SIZE);
        s.set(DeepSeek4Kernels.C_RATIO, r);
        s.set(DeepSeek4Kernels.C_OVERLAP, o);
        s.set(DeepSeek4Kernels.C_WIDTH, width);
        s.set(DeepSeek4Kernels.C_ROPE_DIM, config.ropeDim());
        s.set(DeepSeek4Kernels.C_PENDING_OFFSET, pending);
        s.set(DeepSeek4Kernels.C_CACHE_OFFSET, cache);
        return s;
    }

    private DeepSeek4State gs() {
        return ds;
    }

    // ── graph ─────────────────────────────────────────────────────────────────

    @Override
    protected void setupFFNLayers() {
        TaskGraph graph = new TaskGraph(layerGraphName(firstLayer));
        for (int l = firstLayer; l < endLayer; l++) {
            appendLayer(graph, l);
        }
        lastFFNLayerTaskGraphID = graph.getTaskGraphName();
        ffnLayerITGs = List.of(graph.snapshot());
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        throw new UnsupportedOperationException("the deepseek4 trunk is one task graph");
    }

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

    /**
     * Binds the scratch arrays and the caches. On the first stage {@code firstStage} the hidden
     * state arrives from the activation and the streams start here; on a later stage the streams
     * arrive and the hidden state is scratch.
     */
    private void bindScratch(TaskGraph graph, boolean firstStage) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder, d.hashExperts);
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, firstStage ? d.streams : ws.wrapX);
        graph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                ws.wrapXb,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapXbSums,
                ws.wrapQueryLatent,
                ws.wrapQ,
                ws.wrapAtt,
                ws.wrapRouterLogits,
                ws.wrapSelectedExperts,
                ws.wrapRoutingWeights,
                ws.wrapMoeHidden,
                ws.wrapMoeHiddenQuants,
                ws.wrapMoeHiddenQScales,
                ws.wrapMoeHiddenQSums,
                d.residual,
                d.mixes,
                d.mixPartials,
                d.coeffs,
                d.branch,
                d.ropeFrequencies,
                d.compressRopeFrequencies,
                d.keyValue,
                d.attnOut,
                d.grouped,
                d.compressorKv,
                d.compressorGate,
                d.indexQuery,
                d.indexHeadWeights,
                d.indexBlockScores,
                d.selected,
                d.selectedCount,
                d.empty);
        Object[] caches = {d.window, d.pendingKv, d.pendingScore, d.compressedKeys};
        if (prefill != null) {
            graph.consumeFromDevice(DeepSeek4BatchPrefillLayers.GRAPH, caches);
        } else {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, caches);
        }
        List<Object> shapes = new ArrayList<>();
        for (int l = firstLayer; l < endLayer; l++) {
            shapes.add(attentionShapes[l]);
            if (compressorShapes[l] != null) {
                shapes.add(compressorShapes[l]);
            }
            if (indexerShapes[l] != null) {
                shapes.add(indexerShapes[l]);
            }
        }
        graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, shapes.toArray());
        if (endLayer == config.numberOfLayers()) {
            graph.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.hcHeadFn.asFloatArray(),
                    weights.hcHeadBase.asFloatArray(),
                    weights.hcHeadScale.asFloatArray());
        }
    }

    /** Every device array layer {@code l} reads. */
    static List<Object> layerArrays(
            DeepSeek4LayerWeights<TornadoTensor> w, DeepSeek4Configuration config, int l) {
        List<TornadoTensor> tensors = new ArrayList<>();
        var m = w.mhc();
        var a = w.attention();
        tensors.addAll(
                List.of(
                        m.attnFn()[l],
                        m.attnBase()[l],
                        m.attnScale()[l],
                        m.ffnFn()[l],
                        m.ffnBase()[l],
                        m.ffnScale()[l],
                        w.attnNorm()[l],
                        w.ffnNorm()[l],
                        a.qA()[l],
                        a.qANorm()[l],
                        a.qB()[l],
                        a.kv()[l],
                        a.kvNorm()[l],
                        a.outputA()[l],
                        a.outputB()[l],
                        a.sinks()[l]));
        if (config.ratio(l) != 0) {
            var c = w.compressor();
            tensors.addAll(List.of(c.kv()[l], c.gate()[l], c.ape()[l], c.norm()[l]));
        }
        if (config.hasIndexer(l)) {
            var x = w.indexer();
            tensors.addAll(
                    List.of(
                            x.qB()[l],
                            x.headWeights()[l],
                            x.compressorKv()[l],
                            x.compressorGate()[l],
                            x.compressorApe()[l],
                            x.compressorNorm()[l]));
        }
        var f = w.ffn();
        tensors.add(f.router()[l]);
        if (!config.isHashRouted(l)) {
            tensors.add(f.routerBias()[l]);
        }
        tensors.addAll(f.gateExperts().get(l));
        tensors.addAll(f.upExperts().get(l));
        tensors.addAll(f.downExperts().get(l));
        tensors.addAll(List.of(f.sharedGate()[l], f.sharedUp()[l], f.sharedDown()[l]));
        List<Object> arrays = new ArrayList<>();
        for (TornadoTensor t : tensors) {
            arrays.add(deviceArray(t));
        }
        return arrays;
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        final int dim = config.dim();
        final int n = config.hcCount();
        final float eps = config.rmsNormEps();
        var m = w.mhc();

        if (l == firstLayer && l > 0) {
            // A later stage: the streams arrive from the previous one.
            g.consumeFromDevice(activationGraphName, d.streams);
            bindScratch(g, false);
        }
        if (l == 0) {
            g.consumeFromDevice(activationGraphName, ws.wrapX);
            bindScratch(g, true);
            task(
                    g,
                    l,
                    "broadcast",
                    lanes(n * dim, 128),
                    t ->
                            g.task(
                                    t,
                                    Glm5NextKernels::hcBroadcast,
                                    context,
                                    ws.wrapX,
                                    d.streams,
                                    dim,
                                    n));
        }
        if (prefill != null) {
            g.consumeFromDevice(
                    DeepSeek4BatchPrefillLayers.GRAPH, layerArrays(w, config, l).toArray());
        } else {
            g.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, layerArrays(w, config, l).toArray());
        }

        hyperConnectionPre(g, l, "hc_attn", m.attnFn()[l], m.attnBase()[l], m.attnScale()[l]);
        normQuantize(g, l, "attn_norm", w.attnNorm()[l], dim, eps);
        attention(g, l);
        hyperConnectionPost(g, l, "hc_attn");

        hyperConnectionPre(g, l, "hc_ffn", m.ffnFn()[l], m.ffnBase()[l], m.ffnScale()[l]);
        normQuantize(g, l, "ffn_norm", w.ffnNorm()[l], dim, eps);
        mixtureOfExperts(g, l);
        hyperConnectionPost(g, l, "hc_ffn");

        if (l == endLayer - 1 && endLayer < config.numberOfLayers()) {
            g.persistOnDevice(d.streams);
        }
        if (l == config.numberOfLayers() - 1) {
            head(g, l);
            g.persistOnDevice(ws.wrapX);
        }
    }

    private void hyperConnectionPre(
            TaskGraph g,
            int l,
            String name,
            TornadoTensor fn,
            TornadoTensor base,
            TornadoTensor scale) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        final int dim = config.dim();
        final int n = config.hcCount();
        final int mixCount = config.hcMixCount();
        task(
                g,
                l,
                name + "_mix",
                groups(mixCount * Glm5NextKernels.HC_CHUNKS, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcMixes,
                                context,
                                d.streams,
                                fn.asFloatArray(),
                                d.mixPartials,
                                n * dim,
                                mixCount));
        task(
                g,
                l,
                name + "_coef",
                groups(1, hcLanes(mixCount)),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcCoefficients,
                                context,
                                d.mixPartials,
                                base.asFloatArray(),
                                scale.asFloatArray(),
                                d.mixes,
                                d.coeffs,
                                n,
                                n * dim,
                                config.hcSinkhornIterations(),
                                config.hcEps(),
                                (float) DeepSeek4Configuration.HC_POST_SCALE,
                                config.rmsNormEps()));
        task(
                g,
                l,
                name + "_collapse",
                lanes(dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcCollapse,
                                context,
                                d.streams,
                                d.coeffs,
                                d.residual,
                                ws.wrapX,
                                dim,
                                n));
    }

    private void hyperConnectionPost(TaskGraph g, int l, String name) {
        DeepSeek4DeviceArrays d = ds.workspace.deepseek4;
        final int dim = config.dim();
        final int n = config.hcCount();
        task(
                g,
                l,
                name + "_expand",
                lanes(n * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcExpand,
                                context,
                                d.residual,
                                d.branch,
                                d.coeffs,
                                d.streams,
                                dim,
                                n));
    }

    /** The learned mHC head: its mixes over the streams, then the weighted collapse. */
    private void head(TaskGraph g, int l) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        final int dim = config.dim();
        final int n = config.hcCount();
        task(
                g,
                l,
                "head_mix",
                groups(n * Glm5NextKernels.HC_CHUNKS, Glm5NextKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::hcMixes,
                                context,
                                d.streams,
                                weights.hcHeadFn.asFloatArray(),
                                d.mixPartials,
                                n * dim,
                                n));
        task(
                g,
                l,
                "head_collapse",
                lanes(dim, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::hcHead,
                                context,
                                d.mixPartials,
                                weights.hcHeadBase.asFloatArray(),
                                weights.hcHeadScale.asFloatArray(),
                                d.streams,
                                ws.wrapX,
                                n,
                                dim,
                                config.hcEps(),
                                config.rmsNormEps()));
    }

    // ── attention ─────────────────────────────────────────────────────────────

    private void attention(TaskGraph g, int l) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var a = w.attention();
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int hd = config.headDim();
        final int qLora = config.qLoraRank();
        final int r = config.ratio(l);
        final FloatArray freqs = r == 0 ? d.ropeFrequencies : d.compressRopeFrequencies;

        task(
                g,
                l,
                "q_kv_a",
                lanes((qLora + hd) * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::twoProjectionsQ8_0,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                a.qA()[l].asByteArray(),
                                ws.wrapQueryLatent,
                                a.kv()[l].asByteArray(),
                                d.keyValue,
                                dim,
                                qLora,
                                hd));
        if (r != 0) {
            var c = w.compressor();
            int o = DeepSeek4Configuration.overlap(r);
            compress(
                    g,
                    l,
                    "comp",
                    c.kv()[l],
                    c.gate()[l],
                    c.ape()[l],
                    c.norm()[l],
                    o * hd,
                    compressorShapes[l]);
        }
        if (config.hasIndexer(l)) {
            var x = w.indexer();
            compress(
                    g,
                    l,
                    "idx_comp",
                    x.compressorKv()[l],
                    x.compressorGate()[l],
                    x.compressorApe()[l],
                    x.compressorNorm()[l],
                    2 * config.indexerHeadDim(),
                    indexerShapes[l]);
            projectF32(
                    g,
                    l,
                    "idx_weights",
                    x.headWeights()[l],
                    d.indexHeadWeights,
                    dim,
                    config.indexerHeads());
        }
        task(
                g,
                l,
                "latent_norms",
                groups(2, DeepSeek2Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek2Kernels::latentNorms,
                                context,
                                ws.wrapQueryLatent,
                                a.qANorm()[l].asFloatArray(),
                                d.keyValue,
                                a.kvNorm()[l].asFloatArray(),
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapXbSums,
                                qLora,
                                hd,
                                config.rmsNormEps()));
        project(g, l, "q_b", a.qB()[l], ws.wrapQ, qLora, config.queryDim());
        if (config.hasIndexer(l)) {
            indexer(g, l);
        }
        task(
                g,
                l,
                "attn_prep",
                groups(heads + 1, localFor(hd)),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::attentionPrep,
                                context,
                                ws.wrapQ,
                                d.keyValue,
                                d.window,
                                freqs,
                                ws.positionHolder,
                                heads,
                                hd,
                                config.ropeDim(),
                                config.slidingWindow(),
                                ds.windowOffset(config, l),
                                config.rmsNormEps()));
        task(
                g,
                l,
                "attention",
                groups(heads, DeepSeek4Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::attention,
                                context,
                                ws.wrapQ,
                                d.window,
                                d.compressedKeys,
                                d.selected,
                                d.selectedCount,
                                a.sinks()[l].asFloatArray(),
                                ws.wrapAtt,
                                d.attnOut,
                                freqs,
                                ws.positionHolder,
                                attentionShapes[l],
                                config.attentionScale()));
        quantize(g, l, "attn_out_q8", d.attnOut, config.queryDim());
        final int rows = config.groupedOutputDim();
        task(
                g,
                l,
                "output_a",
                lanes(rows * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::groupedProjection,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                a.outputA()[l].asByteArray(),
                                d.grouped,
                                rows,
                                config.outputLoraRank(),
                                config.groupInputDim()));
        quantize(g, l, "grouped_q8", d.grouped, rows);
        project(g, l, "output_b", a.outputB()[l], d.branch, rows, dim);
    }

    /** A compressor's two F32 projections of {@code wrapXb}, then its store and pooling. */
    private void compress(
            TaskGraph g,
            int l,
            String name,
            TornadoTensor kv,
            TornadoTensor gate,
            TornadoTensor ape,
            TornadoTensor norm,
            int row,
            IntArray shape) {
        DeepSeek4DeviceArrays d = ds.workspace.deepseek4;
        final int dim = config.dim();
        projectF32(g, l, name + "_kv", kv, d.compressorKv, dim, row);
        projectF32(g, l, name + "_gate", gate, d.compressorGate, dim, row);
        task(
                g,
                l,
                name + "_pool",
                groups(1, DeepSeek4Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::compressor,
                                context,
                                d.compressorKv,
                                d.compressorGate,
                                ape.asFloatArray(),
                                norm.asFloatArray(),
                                d.pendingKv,
                                d.pendingScore,
                                d.compressedKeys,
                                d.compressRopeFrequencies,
                                ds.workspace.positionHolder,
                                shape,
                                config.rmsNormEps()));
    }

    /** The indexer's query from the quantized query latent, its block scores and selection. */
    private void indexer(TaskGraph g, int l) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var x = w.indexer();
        final int ih = config.indexerHeads();
        final int id = config.indexerHeadDim();
        final int r = config.ratio(l);
        final int maxBlocks = d.indexBlockScores.getSize();
        project(g, l, "idx_q_b", x.qB()[l], d.indexQuery, config.qLoraRank(), ih * id);
        task(
                g,
                l,
                "idx_rotate",
                lanes(ih * config.ropeDim() / 2, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::rotateHeads,
                                context,
                                d.indexQuery,
                                d.compressRopeFrequencies,
                                ws.positionHolder,
                                ih,
                                id,
                                config.ropeDim()));
        task(
                g,
                l,
                "idx_scores",
                lanes(maxBlocks * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::indexerScores,
                                context,
                                d.indexQuery,
                                d.indexHeadWeights,
                                d.compressedKeys,
                                d.indexBlockScores,
                                ws.positionHolder,
                                ih,
                                id,
                                r,
                                ds.indexerKeyOffset(l),
                                (float) (1.0 / Math.sqrt((double) id * ih))));
        task(
                g,
                l,
                "idx_select",
                groups(1, DeepSeek4Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::indexerSelect,
                                context,
                                d.indexBlockScores,
                                d.selected,
                                d.selectedCount,
                                ws.positionHolder,
                                r,
                                config.indexerTopK()));
    }

    // ── mixture of experts ────────────────────────────────────────────────────

    private void mixtureOfExperts(TaskGraph g, int l) {
        var ws = ds.workspace;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int moeHidden = used * hidden + shared;
        final boolean hashed = config.isHashRouted(l);
        List<TornadoTensor> gates = f.gateExperts().get(l);
        List<TornadoTensor> ups = f.upExperts().get(l);
        List<TornadoTensor> downs = f.downExperts().get(l);
        int last = gates.size() - 1;
        int split = gates.size() == 1 ? experts : config.expertChunkSize();
        ByteArray gate0 = gates.get(0).asByteArray(), gate1 = gates.get(last).asByteArray();
        ByteArray up0 = ups.get(0).asByteArray(), up1 = ups.get(last).asByteArray();
        ByteArray down0 = downs.get(0).asByteArray(), down1 = downs.get(last).asByteArray();
        boolean q4 = gates.get(0).dataType() == DataType.Q4_0;
        FloatArray bias = hashed ? d.empty : f.routerBias()[l].asFloatArray();

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
                                f.router()[l].asFloatArray(),
                                ws.wrapRouterLogits,
                                dim,
                                experts));
        task(
                g,
                l,
                "router_topk",
                groups(1, DeepSeek4Kernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4Kernels::routerSqrtSoftplus,
                                context,
                                ws.wrapRouterLogits,
                                bias,
                                d.hashExperts,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                experts,
                                used,
                                hashed ? 1 : 0,
                                ds.hashOffset(l),
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale()));
        task(
                g,
                l,
                "experts_gate_up",
                lanes(moeHidden * 32, 128),
                t ->
                        g.task(
                                t,
                                q4
                                        ? Glm5NextKernels::expertsGateUpClampedQ4_0
                                        : Glm5NextKernels::expertsGateUpClamped,
                                context,
                                ws.wrapXbQuants,
                                ws.wrapXbScales,
                                ws.wrapSelectedExperts,
                                gate0,
                                up0,
                                gate1,
                                up1,
                                f.sharedGate()[l].asByteArray(),
                                f.sharedUp()[l].asByteArray(),
                                ws.wrapMoeHidden,
                                dim,
                                hidden,
                                shared,
                                used,
                                split,
                                config.swigluLimit()));
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
                lanes(dim * 32, 128),
                t ->
                        g.task(
                                t,
                                q4
                                        ? Glm5NextKernels::expertsDownQ4_0
                                        : Glm5NextKernels::expertsDown,
                                context,
                                ws.wrapMoeHiddenQuants,
                                ws.wrapMoeHiddenQScales,
                                ws.wrapSelectedExperts,
                                ws.wrapRoutingWeights,
                                down0,
                                down1,
                                f.sharedDown()[l].asByteArray(),
                                d.branch,
                                dim,
                                hidden,
                                shared,
                                used,
                                split));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /** The sublayer's RMS norm of {@code wrapX} into {@code wrapXb}, quantized for projections. */
    private void normQuantize(
            TaskGraph g, int l, String name, TornadoTensor weight, int n, float eps) {
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
                                ws.wrapX,
                                ws.wrapXb,
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

    /** {@code out = w . wrapXb} over an F32 weight; {@code n} inputs, {@code d} outputs. */
    private void projectF32(
            TaskGraph g, int l, String name, TornadoTensor weight, FloatArray out, int n, int d) {
        var ws = ds.workspace;
        task(
                g,
                l,
                name,
                lanes(d * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::projectF32,
                                context,
                                ws.wrapXb,
                                weight.asFloatArray(),
                                out,
                                n,
                                d));
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

    /** Lanes per output row: as many as the row has blocks, between one and four warps. */
    private static int projectionLocal(int n) {
        int blocks = n / 32;
        return blocks >= 128 ? 128 : blocks >= 64 ? 64 : 32;
    }

    /** A power-of-two workgroup no wider than {@code n} or {@link DeepSeek4Kernels#GROUP}. */
    private static int localFor(int n) {
        int p = 32;
        while (p * 2 <= Math.min(n, DeepSeek4Kernels.GROUP)) {
            p *= 2;
        }
        return p;
    }

    private static WorkerGrid lanes(int count, int local) {
        int global = (count + local - 1) / local * local;
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    /** The power-of-two workgroup {@link Glm5NextKernels#hcCoefficients} needs for the mixes. */
    private static int hcLanes(int mixCount) {
        if (mixCount > Glm5NextKernels.HC_MAX_MIXES) {
            throw new IllegalArgumentException("too many mHC mixing coefficients: " + mixCount);
        }
        return Math.max(32, Integer.highestOneBit(mixCount - 1) << 1);
    }

    private static WorkerGrid groups(int count, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(count * local);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
        return scheduler;
    }
}
