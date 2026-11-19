package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek4BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek4Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Glm5NextKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.DeepSeek4BatchArrays;
import org.beehive.jitllm.backend.tornado.workspace.DeepSeek4DeviceArrays;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.weights.DeepSeek4LayerWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek4TornadoWeights;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The batched-prefill {@code deepseek4} layers: the whole trunk over one chunk of tokens, in one
 * task graph, the same steps as {@link DeepSeek4Layers} with a row per token.
 *
 * <p>The hyper-connections, the norms, the projections, the routing and the experts run over every
 * row at once. The compressors walk the chunk's rows in order in one launch, as each row's pending
 * entries follow from the rows before it; the indexer and the attention then read the blocks
 * complete at each row's own position. A row's window is the ring's positions before the chunk and
 * the chunk's own rotated keys; the chunk's last keys go into the ring after the attention.
 *
 * <p>The window, the compressors' pending entries and the compressed and indexer keys are the
 * decode plan's own arrays, kept on the device after the chunk; the decode layers take them, and
 * every weight, from this graph by name. The mHC head and the logits are the decode's: a chunk
 * needs no logits.
 */
// @formatter:on
public class DeepSeek4BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    public static final String GRAPH = "deepseek4Prefill";

    private final DeepSeek4State ds;
    private final DeepSeek4Configuration config;
    private final DeepSeek4LayerWeights<TornadoTensor> w;
    private final int batch;
    private final KernelContext context = new KernelContext();
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();
    private final ImmutableTaskGraph graph;

    private final IntArray[] attentionShapes;
    private final IntArray[] compressorShapes;
    private final IntArray[] indexerShapes;

    /** The grouped output GEMM's fixed tiles ({@link #outputTiles()}). */
    private final IntArray outputTiles;

    /** The layers this graph runs: {@code [firstLayer, endLayer)}. */
    private final int firstLayer;

    private final int endLayer;

    public DeepSeek4BatchPrefillLayers(
            DeepSeek4State state,
            DeepSeek4TornadoWeights weights,
            DeepSeek4Configuration config,
            int batch) {
        this(state, weights, config, batch, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)}, one stage of a model split across devices. The
     * first stage copies the chunk's embeddings into the residual streams; a later one receives the
     * streams. A stage before the last keeps the streams on its device for the send to the next.
     */
    public DeepSeek4BatchPrefillLayers(
            DeepSeek4State state,
            DeepSeek4TornadoWeights weights,
            DeepSeek4Configuration config,
            int batch,
            int firstLayer,
            int endLayer) {
        if (state.workspace.deepseek4Batch == null) {
            throw new IllegalStateException(
                    "the deepseek4 batched prefill needs a state built for batch " + batch);
        }
        PrefillFamily family = PrefillFamily.of(state);
        if (family.isNarrow()) {
            // Every chunk-wide array is sized to the plan's padded width, and the decode graphs
            // take the caches from this graph by name, so short chunks run at that width.
            throw PrefillFamily.decline(
                    "the deepseek4 batched prefill runs every chunk at the plan's width");
        }
        this.ds = state;
        this.config = config;
        this.w = weights.layers;
        this.batch = batch;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        int layers = config.numberOfLayers();
        this.attentionShapes = new IntArray[layers];
        this.compressorShapes = new IntArray[layers];
        this.indexerShapes = new IntArray[layers];
        buildShapes();
        this.outputTiles = outputTiles();
        DeepSeek4BatchArrays b = state.workspace.deepseek4Batch;
        int[] expertShape = {
            config.dim(),
            config.expertHiddenDim(),
            config.sharedHiddenDim(),
            config.expertsUsed(),
            Math.min(config.expertChunkSize(), config.expertCount()),
            batch
        };
        for (int i = 0; i < expertShape.length; i++) {
            b.expertShape.set(i, expertShape[i]);
        }
        TaskGraph g = family.taskGraph(GRAPH);
        bind(g);
        var ws = state.workspace;
        if (firstLayer == 0) {
            task(
                    "broadcast",
                    lanes(batch * config.hcCount() * config.dim(), 128),
                    t ->
                            g.task(
                                    t,
                                    Glm5NextBatchKernels::hcBroadcast,
                                    context,
                                    ws.wrapXBatch,
                                    b.streams,
                                    config.dim(),
                                    config.hcCount(),
                                    batch));
        }
        for (int l = firstLayer; l < endLayer; l++) {
            appendLayer(g, l);
        }
        List<Object> kept = new ArrayList<>(List.of(persistent()));
        if (endLayer < layers) {
            kept.add(b.streams);
        }
        g.persistOnDevice(kept.toArray());
        this.graph = g.snapshot();
    }

    private void buildShapes() {
        DeepSeek4BatchArrays b = ds.workspace.deepseek4Batch;
        int capacity = b.selected.getSize() / paddedBatch();
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
            s.set(DeepSeek4Kernels.SHAPE_SELECTION_CAPACITY, capacity);
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

    // ── binding ───────────────────────────────────────────────────────────────

    /** What the decode layers continue from: the window, the pending entries and the keys. */
    public Object[] persistent() {
        DeepSeek4DeviceArrays d = ds.workspace.deepseek4;
        return new Object[] {d.window, d.pendingKv, d.pendingScore, d.compressedKeys};
    }

    /** Every weight this graph binds, which the decode layers take from it by name. */
    public List<Object> weightArrays() {
        List<Object> arrays = new ArrayList<>();
        for (int l = firstLayer; l < endLayer; l++) {
            arrays.addAll(DeepSeek4Layers.layerArrays(w, config, l));
        }
        return arrays;
    }

    private void bind(TaskGraph g) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        // The first stage receives the chunk's embeddings, a later one the streams.
        if (firstLayer == 0) {
            g.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, ws.wrapXBatch);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, b.streams);
        } else {
            g.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, b.streams);
        }
        g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder, b.hashExperts);
        g.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                d.ropeFrequencies,
                d.compressRopeFrequencies,
                d.empty,
                b.residual,
                b.mixes,
                b.mixPartials,
                b.coeffs,
                b.normed,
                b.quants,
                b.scales,
                b.sums,
                b.a8,
                b.a8Scales,
                b.branch,
                b.queryLatent,
                b.keyValue,
                b.query,
                b.attnOut,
                b.grouped,
                b.scores,
                b.compressorKv,
                b.compressorGate,
                b.indexQuery,
                b.indexHeadWeights,
                b.indexBlockScores,
                b.selected,
                b.selectedCount,
                b.routerLogits,
                b.expertIds,
                b.expertWeights,
                b.moeHidden,
                b.moeQuants,
                b.moeScales,
                b.moeSums,
                b.expertShape,
                b.sortedToken,
                b.position,
                b.tiles,
                b.routedHidden,
                b.routedHidden8,
                b.routedHiddenScales,
                b.routedOut,
                b.sharedGate,
                b.sharedUp,
                b.sharedOut,
                b.groupMajor,
                b.groupMajor8,
                b.groupMajorScales,
                b.groupMajorOut,
                outputTiles);
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
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, shapes.toArray());
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, persistent());
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, weightArrays().toArray());
    }

    private void task(String name, WorkerGrid grid, TaskAdder adder) {
        adder.add(name);
        grids.put(GRAPH + "." + name, grid);
    }

    @FunctionalInterface
    private interface TaskAdder {
        void add(String taskName);
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private void appendLayer(TaskGraph g, int l) {
        var m = w.mhc();
        String p = "l" + l + "_";
        hyperConnectionPre(g, p + "hc_attn", m.attnFn()[l], m.attnBase()[l], m.attnScale()[l]);
        norm(g, p + "attn_norm", w.attnNorm()[l]);
        attention(g, p, l);
        hyperConnectionPost(g, p + "hc_attn");
        hyperConnectionPre(g, p + "hc_ffn", m.ffnFn()[l], m.ffnBase()[l], m.ffnScale()[l]);
        norm(g, p + "ffn_norm", w.ffnNorm()[l]);
        mixtureOfExperts(g, p, l);
        hyperConnectionPost(g, p + "hc_ffn");
    }

    private void hyperConnectionPre(
            TaskGraph g, String name, TornadoTensor fn, TornadoTensor base, TornadoTensor scale) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        final int dim = config.dim();
        final int n = config.hcCount();
        final int mixCount = config.hcMixCount();
        if (mixCount > Glm5NextBatchKernels.HC_TILE) {
            throw new UnsupportedOperationException(
                    "the deepseek4 batched mHC takes at most "
                            + Glm5NextBatchKernels.HC_TILE
                            + " mixing coefficients");
        }
        final int squares = paddedBatch() * mixCount * Glm5NextKernels.HC_CHUNKS;
        task(
                name + "_mix",
                groups(
                        (batch + Glm5NextBatchKernels.HC_TILE - 1)
                                / Glm5NextBatchKernels.HC_TILE
                                * Glm5NextKernels.HC_CHUNKS,
                        Glm5NextBatchKernels.HC_TILE * Glm5NextBatchKernels.HC_TILE / 4),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcMixes,
                                context,
                                b.streams,
                                fn.asFloatArray(),
                                b.mixPartials,
                                n * dim,
                                mixCount,
                                squares,
                                batch));
        task(
                name + "_finish",
                lanes(batch * mixCount, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcFinishMixes,
                                context,
                                b.mixPartials,
                                b.mixes,
                                n * dim,
                                mixCount,
                                squares,
                                config.rmsNormEps(),
                                batch));
        task(
                name + "_coef",
                lanes(batch, 32),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcCoefficients,
                                context,
                                b.mixes,
                                base.asFloatArray(),
                                scale.asFloatArray(),
                                b.coeffs,
                                n,
                                config.hcSinkhornIterations(),
                                config.hcEps(),
                                DeepSeek4Configuration.HC_POST_SCALE,
                                batch));
        task(
                name + "_collapse",
                lanes(batch * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcCollapse,
                                context,
                                b.streams,
                                b.coeffs,
                                b.residual,
                                ws.wrapXBatch,
                                dim,
                                n,
                                batch));
    }

    private void hyperConnectionPost(TaskGraph g, String name) {
        DeepSeek4BatchArrays b = ds.workspace.deepseek4Batch;
        final int dim = config.dim();
        final int n = config.hcCount();
        task(
                name + "_expand",
                lanes(batch * n * dim, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::hcExpand,
                                context,
                                b.residual,
                                b.branch,
                                b.coeffs,
                                b.streams,
                                dim,
                                n,
                                batch));
    }

    /** The sublayer's norm of {@code wrapXBatch} into {@code normed}, quantized. */
    private void norm(TaskGraph g, String name, TornadoTensor weight) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        rmsNorm(g, name, ws.wrapXBatch, b.normed, weight, config.dim());
        quantize(g, name + "_q8", b.normed, config.dim());
    }

    private void rmsNorm(
            TaskGraph g, String name, FloatArray in, FloatArray out, TornadoTensor weight, int n) {
        task(
                name,
                groups(batch, Glm5NextBatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::rmsNorm,
                                context,
                                in,
                                out,
                                weight.asFloatArray(),
                                n,
                                config.rmsNormEps(),
                                batch));
    }

    // ── attention ─────────────────────────────────────────────────────────────

    private void attention(TaskGraph g, String p, int l) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var a = w.attention();
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int hd = config.headDim();
        final int qLora = config.qLoraRank();
        final int r = config.ratio(l);
        final FloatArray freqs = r == 0 ? d.ropeFrequencies : d.compressRopeFrequencies;

        project(g, p + "q_a", a.qA()[l], b.queryLatent, dim, qLora);
        project(g, p + "kv", a.kv()[l], b.keyValue, dim, hd);
        if (r != 0) {
            var c = w.compressor();
            int o = DeepSeek4Configuration.overlap(r);
            compress(
                    g,
                    p + "comp",
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
                    p + "idx_comp",
                    x.compressorKv()[l],
                    x.compressorGate()[l],
                    x.compressorApe()[l],
                    x.compressorNorm()[l],
                    2 * config.indexerHeadDim(),
                    indexerShapes[l]);
            project(
                    g,
                    p + "idx_weights",
                    x.headWeights()[l],
                    b.indexHeadWeights,
                    dim,
                    config.indexerHeads());
        }
        rmsNorm(g, p + "q_a_norm", b.queryLatent, b.queryLatent, a.qANorm()[l], qLora);
        rmsNorm(g, p + "kv_norm", b.keyValue, b.keyValue, a.kvNorm()[l], hd);
        quantize(g, p + "q_latent_q8", b.queryLatent, qLora);
        project(g, p + "q_b", a.qB()[l], b.query, qLora, config.queryDim());
        if (config.hasIndexer(l)) {
            indexer(g, p, l);
        }
        task(
                p + "q_prep",
                groups(batch * heads, localFor(hd)),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::queryPrep,
                                context,
                                b.query,
                                freqs,
                                ws.batchStartPosHolder,
                                heads,
                                hd,
                                config.ropeDim(),
                                config.rmsNormEps()));
        task(
                p + "kv_rotate",
                lanes(batch * config.ropeDim() / 2, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::rotateHeads,
                                context,
                                b.keyValue,
                                freqs,
                                ws.batchStartPosHolder,
                                1,
                                hd,
                                config.ropeDim(),
                                batch));
        task(
                p + "attention",
                groups(batch * heads, DeepSeek4BatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::attention,
                                context,
                                b.query,
                                b.keyValue,
                                d.window,
                                d.compressedKeys,
                                b.selected,
                                b.selectedCount,
                                a.sinks()[l].asFloatArray(),
                                b.scores,
                                b.attnOut,
                                freqs,
                                ws.batchStartPosHolder,
                                attentionShapes[l],
                                config.attentionScale()));
        task(
                p + "window_store",
                lanes(batch * hd, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::storeWindow,
                                context,
                                b.keyValue,
                                d.window,
                                ws.batchStartPosHolder,
                                hd,
                                config.slidingWindow(),
                                ds.windowOffset(config, l),
                                batch));
        final int rows = config.groupedOutputDim();
        if (groupedOutputGemm(l)) {
            groupedOutput(g, p, l);
        } else {
            quantize(g, p + "attn_out_q8", b.attnOut, config.queryDim());
            task(
                    p + "output_a",
                    lanes(batch * rows * 32, 128),
                    t ->
                            g.task(
                                    t,
                                    DeepSeek4BatchKernels::groupedProjection,
                                    context,
                                    b.quants,
                                    b.scales,
                                    a.outputA()[l].asByteArray(),
                                    b.grouped,
                                    rows,
                                    config.outputLoraRank(),
                                    config.groupInputDim(),
                                    batch));
        }
        quantize(g, p + "grouped_q8", b.grouped, rows);
        project(g, p + "output_b", a.outputB()[l], b.branch, rows, dim);
    }

    /**
     * Whether layer {@code l}'s grouped output projection runs on the grouped int8 GEMM: Q8_0
     * weights, whole 128-column tiles of output per group and whole 64-value rounds of input.
     */
    private boolean groupedOutputGemm(int l) {
        return w.attention().outputA()[l].dataType() == DataType.Q8_0
                && config.outputLoraRank() % Qwen35MoeBatchKernels.TILE_COLS == 0
                && config.groupInputDim() % Int8GemmKernels.I8_BK == 0;
    }

    // @formatter:off
    /**
     * The grouped output projection {@code wo_a} as the grouped expert GEMM: each group is an
     * "expert" of {@code outputLoraRank} rows, and its rows are the chunk's rows of that group's
     * input. The attention output is laid out group after group and quantized, the GEMM runs over
     * fixed tiles ({@link #outputTiles}), and its output is laid back out row after row.
     */
    // @formatter:on
    private void groupedOutput(TaskGraph g, String p, int l) {
        DeepSeek4BatchArrays b = ds.workspace.deepseek4Batch;
        final int groups = config.outputGroups();
        final int in = config.groupInputDim();
        final int rank = config.outputLoraRank();
        final int padded = paddedBatch();
        task(
                p + "attn_out_groups",
                lanes(padded * groups * in, 256),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::toGroupMajor,
                                context,
                                b.attnOut,
                                b.groupMajor,
                                groups,
                                in,
                                padded));
        task(
                p + "attn_out_groups_q8",
                lanes(groups * padded * in, 128),
                t ->
                        g.task(
                                t,
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                context,
                                b.groupMajor,
                                b.groupMajor8,
                                b.groupMajorScales,
                                in));
        task(
                p + "output_a",
                tileGrid(rank, outputTiles.get(0)),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::groupedDown,
                                context,
                                b.groupMajor8,
                                b.groupMajorScales,
                                outputTiles,
                                w.attention().outputA()[l].asByteArray(),
                                b.groupMajorOut,
                                rank,
                                in,
                                0,
                                groups));
        task(
                p + "output_a_rows",
                lanes(padded * groups * rank, 256),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::fromGroupMajor,
                                context,
                                b.groupMajorOut,
                                b.grouped,
                                groups,
                                rank,
                                padded));
    }

    /**
     * The grouped output GEMM's tiles: every group's padded rows in 64-row tiles, as {@code (group,
     * first row, rows)} after the count, the layout {@link Qwen35MoeBatchKernels#groupByExpert}
     * writes.
     */
    private IntArray outputTiles() {
        int groups = config.outputGroups();
        int padded = paddedBatch();
        int perGroup = padded / Qwen35MoeBatchKernels.TILE_ROWS;
        IntArray tiles = new IntArray(1 + 3 * groups * perGroup);
        int n = 0;
        for (int grp = 0; grp < groups; grp++) {
            for (int j = 0; j < perGroup; j++) {
                tiles.set(1 + 3 * n, grp);
                tiles.set(2 + 3 * n, grp * padded + j * Qwen35MoeBatchKernels.TILE_ROWS);
                tiles.set(3 + 3 * n, Qwen35MoeBatchKernels.TILE_ROWS);
                n++;
            }
        }
        tiles.set(0, n);
        return tiles;
    }

    /** A compressor's two F32 projections of {@code normed}, then its rows in order. */
    private void compress(
            TaskGraph g,
            String name,
            TornadoTensor kv,
            TornadoTensor gate,
            TornadoTensor ape,
            TornadoTensor norm,
            int row,
            IntArray shape) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        final int dim = config.dim();
        project(g, name + "_kv", kv, b.compressorKv, dim, row);
        project(g, name + "_gate", gate, b.compressorGate, dim, row);
        task(
                name + "_pool",
                groups(1, DeepSeek4BatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::compressor,
                                context,
                                b.compressorKv,
                                b.compressorGate,
                                ape.asFloatArray(),
                                norm.asFloatArray(),
                                d.pendingKv,
                                d.pendingScore,
                                d.compressedKeys,
                                d.compressRopeFrequencies,
                                ws.batchStartPosHolder,
                                shape,
                                config.rmsNormEps()));
    }

    /** The indexer's query from the quantized query latent, its block scores and selection. */
    private void indexer(TaskGraph g, String p, int l) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var x = w.indexer();
        final int ih = config.indexerHeads();
        final int id = config.indexerHeadDim();
        final int r = config.ratio(l);
        final int maxBlocks = b.indexBlockScores.getSize() / paddedBatch();
        final int capacity = b.selected.getSize() / paddedBatch();
        project(g, p + "idx_q_b", x.qB()[l], b.indexQuery, config.qLoraRank(), ih * id);
        task(
                p + "idx_rotate",
                lanes(batch * ih * config.ropeDim() / 2, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::rotateHeads,
                                context,
                                b.indexQuery,
                                d.compressRopeFrequencies,
                                ws.batchStartPosHolder,
                                ih,
                                id,
                                config.ropeDim(),
                                batch));
        task(
                p + "idx_scores",
                lanes(batch * maxBlocks * 32, 128),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::indexerScores,
                                context,
                                b.indexQuery,
                                b.indexHeadWeights,
                                d.compressedKeys,
                                b.indexBlockScores,
                                ws.batchStartPosHolder,
                                ih,
                                id,
                                r,
                                ds.indexerKeyOffset(l),
                                maxBlocks,
                                (float) (1.0 / Math.sqrt((double) id * ih)),
                                batch));
        task(
                p + "idx_select",
                groups(batch, DeepSeek4BatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::indexerSelect,
                                context,
                                b.indexBlockScores,
                                b.selected,
                                b.selectedCount,
                                ws.batchStartPosHolder,
                                r,
                                config.indexerTopK(),
                                maxBlocks,
                                capacity));
    }

    // ── mixture of experts ────────────────────────────────────────────────────

    private void mixtureOfExperts(TaskGraph g, String p, int l) {
        if (groupedExperts(l)) {
            groupedMixtureOfExperts(g, p, l);
            return;
        }
        DeepSeek4BatchArrays b = ds.workspace.deepseek4Batch;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int moeHidden = used * hidden + shared;
        List<TornadoTensor> gates = f.gateExperts().get(l);
        List<TornadoTensor> ups = f.upExperts().get(l);
        List<TornadoTensor> downs = f.downExperts().get(l);
        int last = gates.size() - 1;
        int split = gates.size() == 1 ? experts : config.expertChunkSize();
        ByteArray gate0 = gates.get(0).asByteArray(), gate1 = gates.get(last).asByteArray();
        ByteArray up0 = ups.get(0).asByteArray(), up1 = ups.get(last).asByteArray();
        ByteArray down0 = downs.get(0).asByteArray(), down1 = downs.get(last).asByteArray();
        boolean q4 = gates.get(0).dataType() == DataType.Q4_0;

        route(g, p, l);
        task(
                p + "experts_gate_up",
                lanes(batch * moeHidden * 32, 128),
                t ->
                        g.task(
                                t,
                                q4
                                        ? Glm5NextBatchKernels::expertsGateUpClampedQ4_0
                                        : Glm5NextBatchKernels::expertsGateUpClamped,
                                context,
                                b.quants,
                                b.scales,
                                b.expertIds,
                                gate0,
                                up0,
                                gate1,
                                up1,
                                f.sharedGate()[l].asByteArray(),
                                f.sharedUp()[l].asByteArray(),
                                b.moeHidden,
                                b.expertShape,
                                config.swigluLimit()));
        task(
                p + "experts_hidden_q8",
                lanes(batch * moeHidden, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                b.moeHidden,
                                b.moeQuants,
                                b.moeScales,
                                b.moeSums));
        task(
                p + "experts_down",
                lanes(batch * dim * 32, 128),
                t ->
                        g.task(
                                t,
                                q4
                                        ? Glm5NextBatchKernels::expertsDownQ4_0
                                        : Glm5NextBatchKernels::expertsDown,
                                context,
                                b.moeQuants,
                                b.moeScales,
                                b.expertIds,
                                b.expertWeights,
                                down0,
                                down1,
                                f.sharedDown()[l].asByteArray(),
                                b.branch,
                                dim,
                                hidden,
                                shared,
                                used,
                                split,
                                batch));
    }

    /**
     * The router logits and the {@code sqrtsoftplus} top-k (or, on a hash-routed layer, the staged
     * experts with their scores), into {@code expertIds} and {@code expertWeights}.
     */
    private void route(TaskGraph g, String p, int l) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        DeepSeek4DeviceArrays d = ws.deepseek4;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final boolean hashed = config.isHashRouted(l);
        FloatArray bias = hashed ? d.empty : f.routerBias()[l].asFloatArray();
        task(
                p + "router",
                lanes(batch * experts * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::routerLogits,
                                context,
                                b.normed,
                                f.router()[l].asFloatArray(),
                                b.routerLogits,
                                dim,
                                experts,
                                batch));
        task(
                p + "router_topk",
                groups(batch, DeepSeek4BatchKernels.GROUP),
                t ->
                        g.task(
                                t,
                                DeepSeek4BatchKernels::routerSqrtSoftplus,
                                context,
                                b.routerLogits,
                                bias,
                                b.hashExperts,
                                b.expertIds,
                                b.expertWeights,
                                experts,
                                used,
                                hashed ? 1 : 0,
                                ds.hashOffset(l),
                                b.hashStride,
                                config.expertWeightsNorm() ? 1 : 0,
                                config.expertWeightsScale()));
    }

    /**
     * Whether layer {@code l}'s experts run on the grouped int8 GEMMs: Q8_0 or Q4_0 routed experts,
     * a Q8_0 shared expert, and widths of whole 128-column tiles.
     */
    private boolean groupedExperts(int l) {
        var f = w.ffn();
        DataType experts = f.gateExperts().get(l).get(0).dataType();
        return (experts == DataType.Q8_0 || experts == DataType.Q4_0)
                && f.sharedGate()[l].dataType() == DataType.Q8_0
                && config.dim() % Qwen35MoeBatchKernels.TILE_COLS == 0
                && config.expertHiddenDim() % Qwen35MoeBatchKernels.TILE_COLS == 0
                && Glm5NextBatchPrefillLayers.gemmShape(config.dim(), config.sharedHiddenDim())
                && Glm5NextBatchPrefillLayers.gemmShape(config.sharedHiddenDim(), config.dim());
    }

    // @formatter:off
    /**
     * The mixture of experts on int8 tensor cores, as {@code glm5next} runs it:
     *
     * <ol>
     *   <li>the routing; the assignments sorted by expert and cut into 64-row tiles;
     *   <li>per stack of experts, the grouped clamped gate/up over the tiles of its experts, then
     *       the grouped down projection, the hidden activations quantized between;
     *   <li>the shared expert over every row as GEMMs, the clamp between;
     *   <li>each row's routed outputs weighted and added to the shared one, into {@code branch}.
     * </ol>
     */
    // @formatter:on
    private void groupedMixtureOfExperts(TaskGraph g, String p, int l) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        var f = w.ffn();
        final int dim = config.dim();
        final int experts = config.expertCount();
        final int used = config.expertsUsed();
        final int hidden = config.expertHiddenDim();
        final int shared = config.sharedHiddenDim();
        final int assignments = batch * used;
        List<TornadoTensor> gates = f.gateExperts().get(l);
        boolean q4 = gates.get(0).dataType() == DataType.Q4_0;
        List<TornadoTensor> ups = f.upExperts().get(l);
        List<TornadoTensor> downs = f.downExperts().get(l);
        int split = gates.size() == 1 ? experts : config.expertChunkSize();
        int[][] ranges =
                gates.size() == 1
                        ? new int[][] {{0, experts}}
                        : new int[][] {{0, split}, {split, experts}};

        route(g, p, l);
        task(
                p + "group",
                groups(1, Qwen35MoeBatchKernels.GROUP_THREADS),
                t ->
                        g.task(
                                t,
                                Qwen35MoeBatchKernels::groupByExpert,
                                context,
                                b.expertIds,
                                ws.batchStartPosHolder,
                                b.sortedToken,
                                b.position,
                                b.tiles,
                                experts,
                                used,
                                batch));
        for (int s = 0; s < ranges.length; s++) {
            int lo = ranges[s][0];
            int hi = ranges[s][1];
            ByteArray gate = gates.get(s).asByteArray();
            ByteArray up = ups.get(s).asByteArray();
            task(
                    p + "experts_gate_up" + s,
                    tileGrid(hidden, b.maxTiles),
                    t ->
                            g.task(
                                    t,
                                    q4
                                            ? Glm5NextBatchKernels::groupedGateUpClampedQ4_0
                                            : Glm5NextBatchKernels::groupedGateUpClamped,
                                    context,
                                    b.a8,
                                    b.a8Scales,
                                    b.sortedToken,
                                    b.tiles,
                                    gate,
                                    up,
                                    b.routedHidden,
                                    dim,
                                    hidden,
                                    lo,
                                    hi,
                                    config.swigluLimit()));
        }
        task(
                p + "experts_hidden_q8",
                lanes(assignments * hidden, 128),
                t ->
                        g.task(
                                t,
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                context,
                                b.routedHidden,
                                b.routedHidden8,
                                b.routedHiddenScales,
                                hidden));
        for (int s = 0; s < ranges.length; s++) {
            int lo = ranges[s][0];
            int hi = ranges[s][1];
            ByteArray down = downs.get(s).asByteArray();
            task(
                    p + "experts_down" + s,
                    tileGrid(dim, b.maxTiles),
                    t ->
                            g.task(
                                    t,
                                    q4
                                            ? Glm5NextBatchKernels::groupedDownQ4_0
                                            : Glm5NextBatchKernels::groupedDown,
                                    context,
                                    b.routedHidden8,
                                    b.routedHiddenScales,
                                    b.tiles,
                                    down,
                                    b.routedOut,
                                    dim,
                                    hidden,
                                    lo,
                                    hi));
        }
        // The shared expert reads the same quantized input as the routed ones, so it runs after
        // their gate/up and before its own hidden activation replaces that input.
        project(g, p + "shared_gate", f.sharedGate()[l], b.sharedGate, dim, shared);
        project(g, p + "shared_up", f.sharedUp()[l], b.sharedUp, dim, shared);
        task(
                p + "shared_swiglu",
                lanes(batch * shared, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextKernels::clampedSwiGLU,
                                context,
                                b.sharedGate,
                                b.sharedUp,
                                batch * shared,
                                config.swigluLimit()));
        quantize(g, p + "shared_hidden_q8", b.sharedGate, shared);
        project(g, p + "shared_down", f.sharedDown()[l], b.sharedOut, shared, dim);
        task(
                p + "experts_combine",
                lanes(batch * dim, 256),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::combineExperts,
                                context,
                                b.routedOut,
                                b.position,
                                b.expertWeights,
                                b.sharedOut,
                                b.branch,
                                ws.batchStartPosHolder,
                                dim,
                                used));
    }

    /** The grouped GEMMs' grid: a block per 128 output columns and tile of sorted rows. */
    private static WorkerGrid tileGrid(int columns, int maxTiles) {
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        columns
                                / Qwen35MoeBatchKernels.TILE_COLS
                                * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        grid.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        return grid;
    }

    // ── building blocks ───────────────────────────────────────────────────────

    /**
     * Every row's first {@code n} values of {@code x}, quantized per block in both forms: packed
     * for the DP4A kernels and bytewise for the int8 GEMMs.
     */
    private void quantize(TaskGraph g, String name, FloatArray x, int n) {
        DeepSeek4BatchArrays b = ds.workspace.deepseek4Batch;
        task(
                name,
                lanes(batch * n, 32),
                t ->
                        g.task(
                                t,
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                context,
                                x,
                                b.quants,
                                b.scales,
                                b.sums));
        task(
                name + "_i8",
                lanes(batch * n, 128),
                t ->
                        g.task(
                                t,
                                Int8GemmKernels::quantizeActivationsQ8Warp,
                                context,
                                x,
                                b.a8,
                                b.a8Scales,
                                n));
    }

    // @formatter:off
    /**
     * {@code out_b = w . x_b} for every row; {@code n} inputs, {@code d} outputs. An F32 weight
     * reads {@code normed}; a Q8_0 one the quantized activation, on the int8 tensor-core GEMM when
     * the shape allows it ({@link Glm5NextBatchPrefillLayers#gemmShape}), a DP4A warp per row and
     * output otherwise.
     */
    // @formatter:on
    private void project(
            TaskGraph g, String name, TornadoTensor weight, FloatArray out, int n, int d) {
        var ws = ds.workspace;
        DeepSeek4BatchArrays b = ws.deepseek4Batch;
        if (weight.dataType() == DataType.F32) {
            task(
                    name,
                    lanes(batch * d * 32, 128),
                    t ->
                            g.task(
                                    t,
                                    Glm5NextBatchKernels::projectF32,
                                    context,
                                    b.normed,
                                    weight.asFloatArray(),
                                    out,
                                    n,
                                    d,
                                    batch));
            return;
        }
        if (Glm5NextBatchPrefillLayers.gemmShape(n, d)) {
            int padded = paddedBatch();
            WorkerGrid2D grid =
                    new WorkerGrid2D(
                            padded / Int8GemmKernels.I8_BM * Int8GemmKernels.Q8_GEMM_THREADS,
                            d / Int8GemmKernels.I8_BN);
            grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
            task(
                    name,
                    grid,
                    t ->
                            g.task(
                                    t,
                                    Int8GemmKernels::gemmInt8Q8_0,
                                    context,
                                    b.a8,
                                    b.a8Scales,
                                    weight.asByteArray(),
                                    out,
                                    out,
                                    padded,
                                    d,
                                    n,
                                    Int8GemmKernels.EPILOGUE_STORE,
                                    out,
                                    1,
                                    ws.batchStartPosHolder,
                                    d,
                                    0));
            return;
        }
        task(
                name,
                lanes(batch * d * 32, 128),
                t ->
                        g.task(
                                t,
                                Glm5NextBatchKernels::projectQ8_0,
                                context,
                                b.quants,
                                b.scales,
                                out,
                                weight.asByteArray(),
                                n,
                                d,
                                batch));
    }

    private int paddedBatch() {
        return (batch + 127) / 128 * 128;
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

    private static WorkerGrid groups(int count, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(count * local);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    // ── BatchPrefillTransformerLayerTaskGraphs ────────────────────────────────

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return List.of(graph);
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return GRAPH;
    }

    @Override
    public String describeProjections() {
        return "int8 tensor-core GEMMs on Q8_0 weights where the shape allows, DP4A otherwise";
    }

    @Override
    public String describeAttention() {
        return "sliding window plus compressed blocks with sinks, a workgroup per row and head";
    }
}
