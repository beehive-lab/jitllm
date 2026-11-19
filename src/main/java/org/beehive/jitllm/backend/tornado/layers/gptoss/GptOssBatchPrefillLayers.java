package org.beehive.jitllm.backend.tornado.layers.gptoss;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.kernels.GptOssBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.GptOssKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TensorCoreAttentionKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PrefillFamily;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.GptOssTornadoWeights;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * gpt-oss's batched prefill, every layer in one graph. A chunk is normed and quantized to int8
 * blocks once per projection input; Q, K, V and the attention output run as int8 tensor-core GEMMs
 * on the Q8_0 weights ({@link Int8GemmKernels#gemmInt8Q8_0}), with the output rows past the last
 * whole 128-column tile (64 of gpt-oss's 2880) and the output bias in a warp-per-row tail. The
 * experts are sorted by expert and run as grouped int8 GEMMs over MXFP4 ({@link
 * GptOssBatchKernels}); the attention is the decode kernel's online softmax with sinks, one
 * work-group per (row, head).
 */
public class GptOssBatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    private final GptOssState state;
    private final GptOssTornadoWeights weights;
    private final GptOssLayerWeights<TornadoTensor> w;
    private final GptOssConfiguration config;
    private final KernelContext context;
    private final PrefillFamily family;
    private final int batch;
    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();
    private final List<ImmutableTaskGraph> graphs = new ArrayList<>();
    private final String graphName = GraphNames.batchPrefillLayer(0);
    private final String lastGraph;

    /** The tensor-core attention's FP16 staging: queries, gathered keys and values. */
    private final boolean tensorCoreAttention;

    private final int paddedPositions;
    private final HalfFloatArray queries;
    private final HalfFloatArray keys;
    private final HalfFloatArray values;

    public GptOssBatchPrefillLayers(
            GptOssState state,
            GptOssTornadoWeights weights,
            GptOssConfiguration config,
            int batchSize) {
        if (batchSize % Int8GemmKernels.I8_BM != 0) {
            throw new UnsupportedOperationException(
                    "the gpt-oss batched prefill needs a chunk width that is a multiple of "
                            + Int8GemmKernels.I8_BM
                            + ", not "
                            + batchSize);
        }
        this.state = state;
        this.family = PrefillFamily.of(state);
        GptOssBatchPrefillLayers primary = family.primary(GptOssBatchPrefillLayers.class);
        this.context = primary != null ? primary.context : new KernelContext();
        this.weights = weights;
        this.w = weights.layers();
        this.config = config;
        this.batch = batchSize;
        this.tensorCoreAttention =
                state.usesFp16KeyValueCache() && TensorCoreSupport.isTensorCoreCapableBackend();
        this.paddedPositions = (config.contextLength() + 127) / 128 * 128;
        if (primary != null) {
            this.queries = primary.queries;
            this.keys = primary.keys;
            this.values = primary.values;
        } else if (tensorCoreAttention) {
            this.queries = new HalfFloatArray(batchSize * config.queryDim());
            this.keys = new HalfFloatArray(config.kvDim() * paddedPositions);
            this.values = new HalfFloatArray(config.kvDim() * paddedPositions);
        } else {
            this.queries = null;
            this.keys = null;
            this.values = null;
        }
        TaskGraph g = family.taskGraph(graphName);
        for (int l = 0; l < config.numberOfLayers(); l++) {
            appendLayer(g, l);
        }
        this.lastGraph = g.getTaskGraphName();
        graphs.add(g.snapshot());
    }

    /** The graph every layer's weights are uploaded in: what the decode graphs consume from. */
    public String graphName(int layer) {
        return graphName;
    }

    @Override
    public String describeProjections() {
        return "int8 tensor-core GEMMs on Q8_0 attention and MXFP4 experts";
    }

    /** The arrays of layer {@code l} the decode graphs share with this one. */
    public static Object[] layerWeights(GptOssTornadoWeights weights, int l) {
        GptOssLayerWeights<TornadoTensor> w = weights.layers();
        return new Object[] {
            weights.rms_att_weightLayered[l].asFloatArray(),
            weights.wqLayered[l].deviceArray(),
            weights.wkLayered[l].deviceArray(),
            weights.wvLayered[l].deviceArray(),
            weights.woLayered[l].deviceArray(),
            w.queryBias()[l].asFloatArray(),
            w.keyBias()[l].asFloatArray(),
            w.valueBias()[l].asFloatArray(),
            w.outputBias()[l].asFloatArray(),
            w.sinks()[l].asFloatArray(),
            weights.rms_ffn_weightLayered[l].asFloatArray(),
            w.router()[l].asFloatArray(),
            w.routerBias()[l].asFloatArray(),
            GptOssLayers.experts(w.gateExperts()[l]),
            w.gateExpertsBias()[l].asFloatArray(),
            GptOssLayers.experts(w.upExperts()[l]),
            w.upExpertsBias()[l].asFloatArray(),
            GptOssLayers.experts(w.downExperts()[l]),
            w.downExpertsBias()[l].asFloatArray()
        };
    }

    private Object[] scratch() {
        var ws = state.workspace;
        return new Object[] {
            context,
            ws.wrapNormedBatch,
            ws.wrapQBatch,
            ws.wrapKBatch,
            ws.wrapVBatch,
            ws.wrapXbBatch,
            ws.wrapQ8ActBatch,
            ws.wrapQ8ActScales,
            ws.wrapMoeLogitsBatch,
            ws.wrapMoeIdsBatch,
            ws.wrapMoeWeightsBatch,
            ws.wrapMoeSortedToken,
            ws.wrapMoePosition,
            ws.wrapMoeTiles,
            ws.wrapMoeHiddenBatch,
            ws.wrapMoeHiddenQ8,
            ws.wrapMoeHiddenScales,
            ws.wrapMoeOutBatch,
            keyCache(),
            valueCache(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    private Object[] attentionScratch() {
        return tensorCoreAttention ? new Object[] {queries, keys, values} : new Object[0];
    }

    private void appendLayer(TaskGraph g, int l) {
        var ws = state.workspace;
        int dim = config.dim();
        int qDim = config.queryDim();
        int kvDim = config.kvDim();
        int heads = config.numberOfHeads();
        int experts = config.numberOfExperts();
        int used = config.numberOfExpertsUsed();
        int hidden = config.expertHiddenDim();
        int assignments = batch * used;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, experts);
        float eps = config.rmsNormEps();
        String p = "l" + l + "_";
        if (l == 0) {
            g.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, ws.wrapXBatch);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch());
            if (tensorCoreAttention) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, attentionScratch());
            }
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.wrapBlockTable);
        }
        UnifiedMemoryWeights.upload(g, ws, layerWeights(weights, l));

        // ── attention ──
        task(g, p + "attn_norm", rowsGrid(batch, 256));
        g.task(
                p + "attn_norm",
                GptOssBatchKernels::rmsNormRows,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                weights.rms_att_weightLayered[l].asFloatArray(),
                ws.batchStartPosHolder,
                dim,
                eps);
        quantize(g, p + "attn_q8", ws.wrapNormedBatch, dim);
        gemm(
                g,
                p + "q",
                weights.wqLayered[l].asByteArray(),
                ws.wrapQBatch,
                qDim,
                qDim,
                dim,
                Int8GemmKernels.EPILOGUE_STORE);
        gemm(
                g,
                p + "k",
                weights.wkLayered[l].asByteArray(),
                ws.wrapKBatch,
                kvDim,
                kvDim,
                dim,
                Int8GemmKernels.EPILOGUE_STORE);
        gemm(
                g,
                p + "v",
                weights.wvLayered[l].asByteArray(),
                ws.wrapVBatch,
                kvDim,
                kvDim,
                dim,
                Int8GemmKernels.EPILOGUE_STORE);
        if (tensorCoreAttention) {
            task(g, p + "rope", elements(batch * heads * config.headSize() / 2, 128));
            g.task(
                    p + "rope",
                    GptOssBatchKernels::biasRopeCacheFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    w.queryBias()[l].asFloatArray(),
                    w.keyBias()[l].asFloatArray(),
                    w.valueBias()[l].asFloatArray(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    heads,
                    config.numberOfKeyValueHeads(),
                    kvDim,
                    l,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            task(g, p + "attention_gather", elements(paddedPositions * kvDim, 256));
            g.task(
                    p + "attention_gather",
                    TensorCoreAttentionKernels::gatherPagedKeysTransposed,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    keys,
                    values,
                    l,
                    kvDim,
                    config.headSize(),
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    paddedPositions);
            task(g, p + "attention_q16", elements(batch * qDim, 256));
            g.task(
                    p + "attention_q16",
                    TensorCoreAttentionKernels::queriesToHalf,
                    context,
                    ws.wrapQBatch,
                    queries,
                    qDim,
                    qDim,
                    batch * qDim);
            task(
                    g,
                    p + "attention",
                    rowsGrid(
                            batch / TensorCoreAttentionKernels.FLASH_ROWS * heads,
                            TensorCoreAttentionKernels.FLASH_THREADS));
            g.task(
                    p + "attention",
                    GptOssBatchKernels::flashAttentionSinks64,
                    context,
                    ws.batchStartPosHolder,
                    queries,
                    keys,
                    values,
                    ws.wrapXbBatch,
                    heads,
                    config.kvMul(),
                    qDim,
                    batch / TensorCoreAttentionKernels.FLASH_ROWS,
                    paddedPositions,
                    (float) (1.0 / Math.sqrt(config.headSize())),
                    w.sinks()[l].asFloatArray(),
                    config.isSlidingWindowLayer(l) ? config.slidingWindow() : 0);
        } else if (state.usesFp16KeyValueCache()) {
            task(g, p + "rope", elements(batch * heads * config.headSize() / 2, 128));
            g.task(
                    p + "rope",
                    GptOssBatchKernels::biasRopeCacheFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    w.queryBias()[l].asFloatArray(),
                    w.keyBias()[l].asFloatArray(),
                    w.valueBias()[l].asFloatArray(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    heads,
                    config.numberOfKeyValueHeads(),
                    kvDim,
                    l,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            task(g, p + "attention", rowsGrid(batch * heads, GptOssKernels.HEAD));
            g.task(
                    p + "attention",
                    GptOssBatchKernels::attentionFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    w.sinks()[l].asFloatArray(),
                    ws.wrapXbBatch,
                    heads,
                    kvDim,
                    config.kvMul(),
                    config.isSlidingWindowLayer(l) ? config.slidingWindow() : 0,
                    l,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            task(g, p + "rope", elements(batch * heads * config.headSize() / 2, 128));
            g.task(
                    p + "rope",
                    GptOssBatchKernels::biasRopeCache,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    w.queryBias()[l].asFloatArray(),
                    w.keyBias()[l].asFloatArray(),
                    w.valueBias()[l].asFloatArray(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    heads,
                    config.numberOfKeyValueHeads(),
                    kvDim,
                    l,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            task(g, p + "attention", rowsGrid(batch * heads, GptOssKernels.HEAD));
            g.task(
                    p + "attention",
                    GptOssBatchKernels::attention,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    w.sinks()[l].asFloatArray(),
                    ws.wrapXbBatch,
                    heads,
                    kvDim,
                    config.kvMul(),
                    config.isSlidingWindowLayer(l) ? config.slidingWindow() : 0,
                    l,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
        quantize(g, p + "attn_out_q8", ws.wrapXbBatch, qDim);
        int covered = dim / Int8GemmKernels.I8_BN * Int8GemmKernels.I8_BN;
        gemm(
                g,
                p + "wo",
                weights.woLayered[l].asByteArray(),
                ws.wrapXBatch,
                covered,
                dim,
                qDim,
                Int8GemmKernels.EPILOGUE_RESIDUAL);
        if (covered < dim) {
            task(g, p + "wo_tail", elements(batch * (dim - covered) * 32, 256));
            g.task(
                    p + "wo_tail",
                    GptOssBatchKernels::tailResidualQ8_0,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapXbBatch,
                    weights.woLayered[l].asByteArray(),
                    ws.wrapXBatch,
                    dim,
                    qDim,
                    covered);
        }
        task(g, p + "wo_bias", elements(batch * dim, 256));
        g.task(
                p + "wo_bias",
                GptOssBatchKernels::biasRows,
                context,
                ws.batchStartPosHolder,
                w.outputBias()[l].asFloatArray(),
                ws.wrapXBatch,
                dim);

        // ── experts ──
        task(g, p + "ffn_norm", rowsGrid(batch, 256));
        g.task(
                p + "ffn_norm",
                GptOssBatchKernels::rmsNormRows,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                weights.rms_ffn_weightLayered[l].asFloatArray(),
                ws.batchStartPosHolder,
                dim,
                eps);
        task(g, p + "router", elements(batch * experts * 32, 256));
        g.task(
                p + "router",
                GptOssBatchKernels::router,
                context,
                ws.batchStartPosHolder,
                ws.wrapNormedBatch,
                w.router()[l].asFloatArray(),
                w.routerBias()[l].asFloatArray(),
                ws.wrapMoeLogitsBatch,
                experts,
                dim);
        task(g, p + "topk", elements(batch, 128));
        g.task(
                p + "topk",
                GptOssBatchKernels::topKSoftmax,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeLogitsBatch,
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                experts,
                used);
        quantize(g, p + "ffn_q8", ws.wrapNormedBatch, dim);
        task(g, p + "group", rowsGrid(1, Qwen35MoeBatchKernels.GROUP_THREADS));
        g.task(
                p + "group",
                Qwen35MoeBatchKernels::groupByExpert,
                context,
                ws.wrapMoeIdsBatch,
                ws.batchStartPosHolder,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                experts,
                used,
                batch);
        task(g, p + "gate_up", groupedGrid(hidden, maxTiles));
        g.task(
                p + "gate_up",
                GptOssBatchKernels::groupedGateUpMXFP4,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                ws.wrapMoeSortedToken,
                ws.wrapMoeTiles,
                GptOssLayers.experts(w.gateExperts()[l]),
                GptOssLayers.experts(w.upExperts()[l]),
                w.gateExpertsBias()[l].asFloatArray(),
                w.upExpertsBias()[l].asFloatArray(),
                ws.wrapMoeHiddenBatch,
                dim,
                hidden);
        task(g, p + "hidden_q8", elements(assignments * hidden, 256));
        g.task(
                p + "hidden_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                hidden);
        task(g, p + "down", groupedGrid(dim, maxTiles));
        g.task(
                p + "down",
                GptOssBatchKernels::groupedDownMXFP4,
                context,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                ws.wrapMoeTiles,
                GptOssLayers.experts(w.downExperts()[l]),
                w.downExpertsBias()[l].asFloatArray(),
                ws.wrapMoeOutBatch,
                dim,
                hidden);
        task(g, p + "combine", elements(batch * dim, 256));
        g.task(
                p + "combine",
                GptOssBatchKernels::combine,
                context,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapXBatch,
                ws.batchStartPosHolder,
                dim,
                used);

        if (l == config.numberOfLayers() - 1) {
            g.persistOnDevice(
                    ws.wrapXBatch,
                    keyCache(),
                    valueCache(),
                    ws.wrapBlockTable,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        }
    }

    private void quantize(TaskGraph g, String name, FloatArray in, int k) {
        task(g, name, elements(batch * k, 256));
        g.task(
                name,
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                in,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                k);
    }

    /** {@code out[r * ldo + c] (+)= W[c]·a[r]} for the first {@code n} output columns. */
    private void gemm(
            TaskGraph g,
            String name,
            ByteArray weight,
            FloatArray out,
            int n,
            int ldo,
            int k,
            int epilogue) {
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        (batch / Int8GemmKernels.I8_BM) * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        task(g, name, grid);
        g.task(
                name,
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                weight,
                out,
                out,
                batch,
                n,
                k,
                epilogue,
                out,
                1,
                state.workspace.batchStartPosHolder,
                ldo,
                0);
    }

    private static WorkerGrid groupedGrid(int columns, int tiles) {
        int blocks = (columns + GptOssBatchKernels.TILE_COLS - 1) / GptOssBatchKernels.TILE_COLS;
        WorkerGrid2D grid = new WorkerGrid2D(blocks * GptOssBatchKernels.GEMM_THREADS, tiles);
        grid.setLocalWork(GptOssBatchKernels.GEMM_THREADS, 1, 1);
        return grid;
    }

    private static WorkerGrid rowsGrid(int groups, int local) {
        return GptOssLayers.elements(groups * local, local);
    }

    private static WorkerGrid elements(int n, int local) {
        return GptOssLayers.elements(n, local);
    }

    private void task(TaskGraph g, String name, WorkerGrid grid) {
        grids.put(g.getTaskGraphName() + "." + name, grid);
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastGraph;
    }

    private Object keyCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private Object valueCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }
}
