package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.kernels.BatchMmaKernels;
import org.beehive.jitllm.backend.tornado.kernels.DeepSeek2BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.weights.DeepSeek2LayerWeights;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek2TornadoWeights;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The batched prefill of {@code deepseek2} layers: a chunk of prompt tokens per execution, several
 * layers to a task graph (a whole stage by default).
 *
 * <p>The projections are block-scaled int8 tensor-core GEMMs on the Q8_0 weights; the attention is
 * absorbed latent attention as FP16 tensor-core GEMMs (see {@link DeepSeek2BatchKernels}); the
 * experts are the grouped int8 GEMMs of {@code qwen35moe}, after llama.cpp's sigmoid routing.
 *
 * <p>Three weights are read in another form than the decode reads them: {@code attn_kv_a_mqa}
 * padded to 640 rows for the GEMM's tiles, and {@code attn_k_b} and {@code attn_v_b} in FP16 for
 * the tensor cores. Everything else is the decode's own arrays, bound here first, so the decode
 * graphs of the same plan take them from these graphs rather than holding a second copy.
 */
// @formatter:on
public class DeepSeek2BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    /** Layers per task graph; the whole stage at the default. */
    static final int LAYERS_PER_GRAPH = 64;

    private final DeepSeek2State state;
    private final DeepSeek2TornadoWeights weights;
    private final DeepSeek2LayerWeights<TornadoTensor> w;
    private final DeepSeek2Configuration config;
    private final KernelContext context = new KernelContext();
    private final int firstLayer;
    private final int endLayer;
    private final int batch;
    private final FloatArray zeroBias;

    private final ByteArray[] kvAPadded;
    private final HalfFloatArray[] kBF16;
    private final HalfFloatArray[] vBF16;

    /** Whether the state keeps its latent cache in half precision. */
    private final boolean fp16Cache;

    private final Map<String, WorkerGrid> grids = new LinkedHashMap<>();
    private final List<ImmutableTaskGraph> graphs = new ArrayList<>();
    private String lastGraph;

    public DeepSeek2BatchPrefillLayers(
            DeepSeek2State state,
            DeepSeek2TornadoWeights weights,
            DeepSeek2Configuration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        if (state.paddedBatch == 0 || batchSize % 128 != 0) {
            throw new UnsupportedOperationException(
                    "the deepseek2 batched prefill needs a chunk width that is a multiple of 128,"
                            + " not "
                            + batchSize);
        }
        this.state = state;
        this.fp16Cache = state.usesFp16KeyValueCache();
        this.weights = weights;
        this.w = weights.layers;
        this.config = config;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.batch = state.paddedBatch;
        this.zeroBias = new FloatArray(Math.max(1, config.expertCount()));
        int layers = config.numberOfLayers();
        this.kvAPadded = new ByteArray[layers];
        this.kBF16 = new HalfFloatArray[layers];
        this.vBF16 = new HalfFloatArray[layers];
        for (int l = firstLayer; l < endLayer; l++) {
            kvAPadded[l] = padRows(w.kvAMqa()[l].asByteArray(), config.compressedKeyValueDim());
            kBF16[l] = toHalf(w.kB()[l].asByteArray());
            vBF16[l] = toHalf(w.vB()[l].asByteArray());
        }
        for (int first = firstLayer; first < endLayer; first += LAYERS_PER_GRAPH) {
            TaskGraph graph = new TaskGraph(graphName(first));
            int last = Math.min(first + LAYERS_PER_GRAPH, endLayer) - 1;
            for (int l = first; l <= last; l++) {
                appendLayer(graph, l);
            }
            lastGraph = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
    }

    /**
     * The prefill graph that binds layer {@code l}'s arrays; the decode graphs take them from it.
     */
    public String graphName(int l) {
        int offset = l - firstLayer;
        return "batchPrefillLayer_" + (firstLayer + offset - offset % LAYERS_PER_GRAPH);
    }

    @Override
    public String describeProjections() {
        return "int8 tensor-core GEMMs on Q8_0 weights; FP16 tensor-core latent attention";
    }

    // ── host-side weight forms ────────────────────────────────────────────────

    /** {@code rows} Q8_0 rows padded with zero rows to {@link DeepSeek2State#KV_A_PADDED}. */
    private ByteArray padRows(ByteArray q8, int rows) {
        int rowBytes = config.dim() / 32 * 34;
        ByteArray out = new ByteArray(DeepSeek2State.KV_A_PADDED * rowBytes);
        for (int i = 0; i < rows * rowBytes; i++) {
            out.set(i, q8.get(i));
        }
        return out;
    }

    /** Q8_0 blocks decoded to FP16, element for element. */
    private static HalfFloatArray toHalf(ByteArray q8) {
        int blocks = q8.getSize() / 34;
        HalfFloatArray out = new HalfFloatArray(blocks * 32);
        for (int b = 0; b < blocks; b++) {
            float scale = q8.getHalfFloat(b * 34).getFloat32();
            for (int i = 0; i < 32; i++) {
                out.set(b * 32 + i, new HalfFloat(scale * q8.get(b * 34 + 2 + i)));
            }
        }
        return out;
    }

    // ── binding ───────────────────────────────────────────────────────────────

    /** The latent cache in the representation the state allocated. */
    private Object cache() {
        return fp16Cache ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object[] scratch() {
        var ws = state.workspace;
        return new Object[] {
            context,
            ws.wrapNormedBatch,
            state.workspace.queryLatentBatch,
            state.workspace.compressedKvBatch,
            state.workspace.queryBatch,
            state.workspace.queryF16Batch,
            state.workspace.absorbedBatch,
            state.workspace.absorbedF16Batch,
            state.workspace.keysF16,
            state.workspace.latentTF16,
            state.workspace.scoresBatch,
            state.workspace.probsF16Batch,
            state.workspace.latentBatch,
            state.workspace.latentF16Batch,
            state.workspace.attnOutBatch,
            state.workspace.gateBatch,
            state.workspace.hiddenBatch,
            ws.wrapQ8ActBatch,
            ws.wrapQ8ActScales,
            ws.wrapQ8SplitPartial,
            ws.wrapMoeLogitsBatch,
            ws.wrapMoeIdsBatch,
            ws.wrapMoeWeightsBatch,
            ws.wrapMoeSharedGateBatch,
            ws.wrapMoeSortedToken,
            ws.wrapMoePosition,
            ws.wrapMoeTiles,
            ws.wrapMoeHiddenBatch,
            ws.wrapMoeHiddenQ8,
            ws.wrapMoeHiddenScales,
            ws.wrapMoeOutBatch,
            ws.wrapMoeSharedGateUp,
            ws.wrapMoeSharedHidden,
            ws.wrapMoeSharedQ8,
            ws.wrapMoeSharedScales,
            ws.wrapMoeSharedOut,
            cache(),
            zeroBias,
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    /** The arrays of layer {@code l} the decode graphs share with these: the file's own. */
    public static List<Object> sharedWeights(
            DeepSeek2LayerWeights<TornadoTensor> w, DeepSeek2Configuration config, int l) {
        List<Object> arrays = new ArrayList<>();
        for (TornadoTensor t :
                new TornadoTensor[] {
                    w.attnNorm()[l],
                    w.qA()[l],
                    w.qANorm()[l],
                    w.qB()[l],
                    w.kvANorm()[l],
                    w.wo()[l],
                    w.ffnNorm()[l]
                }) {
            arrays.add(array(t));
        }
        if (config.isDenseLayer(l)) {
            arrays.add(array(w.ffnGate()[l]));
            arrays.add(array(w.ffnUp()[l]));
            arrays.add(array(w.ffnDown()[l]));
        } else {
            arrays.add(array(w.router()[l]));
            if (w.routerBias()[l] != null) {
                arrays.add(array(w.routerBias()[l]));
            }
            arrays.add(array(w.gateExperts()[l]));
            arrays.add(array(w.upExperts()[l]));
            arrays.add(array(w.downExperts()[l]));
            arrays.add(array(w.sharedGate()[l]));
            arrays.add(array(w.sharedUp()[l]));
            arrays.add(array(w.sharedDown()[l]));
        }
        return arrays;
    }

    private static Object array(TornadoTensor t) {
        return switch (t.dataType()) {
            case F32 -> t.asFloatArray();
            case F16 -> t.asHalfFloatArray();
            default -> t.asByteArray();
        };
    }

    // ── one layer ─────────────────────────────────────────────────────────────

    private String task;

    private void appendLayer(TaskGraph g, int l) {
        var ws = state.workspace;
        final int dim = config.dim();
        final int heads = config.numberOfHeads();
        final int rank = config.kvLoraRank();
        final int rope = config.ropeDim();
        final int nope = config.nopeDim();
        final int qHead = config.queryHeadDim();
        final int qLora = config.qLoraRank();
        final int qDim = config.queryDim();
        final int valueHead = config.valueHeadDim();
        final int attnOut = config.attentionOutputInputDim();
        final int keyWidth = config.keyWidth();
        final int kvPad = DeepSeek2State.KV_A_PADDED;
        final int npad = state.paddedPositions;
        final int rows = batch * heads;
        final float eps = config.rmsNormEps();
        final int cacheOffset = state.cacheLayerOffset(l);
        final String p = "l" + l + "_";

        if ((l - firstLayer) % LAYERS_PER_GRAPH == 0) {
            if (l == firstLayer) {
                g.consumeFromDevice("prefillActivation", ws.wrapXBatch);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch());
            } else {
                String previous = graphName(l - 1);
                g.consumeFromDevice(previous, ws.wrapXBatch, ws.batchStartPosHolder);
                g.consumeFromDevice(previous, scratch());
            }
        }
        List<Object> own = sharedWeights(w, config, l);
        own.add(kvAPadded[l]);
        own.add(kBF16[l]);
        own.add(vBF16[l]);
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, own.toArray());

        // ── attention ──
        rmsRows(
                g,
                p + "attn_norm",
                ws.wrapXBatch,
                ws.wrapNormedBatch,
                w.attnNorm()[l],
                dim,
                dim,
                dim,
                eps);
        quantize(g, p + "attn_q8", ws.wrapNormedBatch, dim);
        gemm(
                g,
                p + "q_a",
                w.qA()[l].asByteArray(),
                state.workspace.queryLatentBatch,
                qLora,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemm(
                g,
                p + "kv_a",
                kvAPadded[l],
                state.workspace.compressedKvBatch,
                kvPad,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        rmsRows(
                g,
                p + "q_a_norm",
                state.workspace.queryLatentBatch,
                state.workspace.queryLatentBatch,
                w.qANorm()[l],
                qLora,
                qLora,
                qLora,
                eps);
        rmsRows(
                g,
                p + "kv_a_norm",
                state.workspace.compressedKvBatch,
                state.workspace.compressedKvBatch,
                w.kvANorm()[l],
                rank,
                kvPad,
                kvPad,
                eps);
        quantize(g, p + "q_lat_q8", state.workspace.queryLatentBatch, qLora);
        gemm(
                g,
                p + "q_b",
                w.qB()[l].asByteArray(),
                state.workspace.queryBatch,
                qDim,
                qLora,
                Int8GemmKernels.EPILOGUE_STORE,
                false);

        int perRow = heads * rope / 2 + rope / 2 + rank;
        if (fp16Cache) {
            g.task(
                    p + "rope_cache",
                    DeepSeek2BatchKernels::ropeAndCacheRowsFP16,
                    context,
                    ws.batchStartPosHolder,
                    state.workspace.queryBatch,
                    state.workspace.compressedKvBatch,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCacheFP16,
                    heads,
                    qHead,
                    nope,
                    rope,
                    rank,
                    qDim,
                    kvPad,
                    cacheOffset);
        } else {
            g.task(
                    p + "rope_cache",
                    DeepSeek2BatchKernels::ropeAndCacheRows,
                    context,
                    ws.batchStartPosHolder,
                    state.workspace.queryBatch,
                    state.workspace.compressedKvBatch,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCache,
                    heads,
                    qHead,
                    nope,
                    rope,
                    rank,
                    qDim,
                    kvPad,
                    cacheOffset);
        }
        grid(g, p + "rope_cache", lanes(batch * perRow, 256));
        toHalf(
                g,
                p + "q_f16",
                state.workspace.queryBatch,
                state.workspace.queryF16Batch,
                batch * qDim);
        // Absorb: per head, absorbed[t][h][r] = k_b[h][r] . q_nope[t][h].
        mma(
                g,
                p + "absorb",
                state.workspace.queryF16Batch,
                kBF16[l],
                state.workspace.absorbedBatch,
                batch,
                rank,
                nope,
                qDim,
                nope,
                heads * keyWidth,
                qHead,
                rank * nope,
                keyWidth,
                heads);
        g.task(
                p + "queries",
                DeepSeek2BatchKernels::assembleQueries,
                context,
                state.workspace.absorbedBatch,
                state.workspace.queryBatch,
                state.workspace.absorbedF16Batch,
                heads,
                rank,
                rope,
                nope,
                qHead,
                qDim,
                rows * keyWidth);
        grid(g, p + "queries", lanes(rows * keyWidth, 256));
        if (fp16Cache) {
            g.task(
                    p + "cache_f16",
                    DeepSeek2BatchKernels::cacheToHalfFP16,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKeyCacheFP16,
                    state.workspace.keysF16,
                    state.workspace.latentTF16,
                    cacheOffset,
                    rank,
                    rope,
                    npad);
        } else {
            g.task(
                    p + "cache_f16",
                    DeepSeek2BatchKernels::cacheToHalf,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKeyCache,
                    state.workspace.keysF16,
                    state.workspace.latentTF16,
                    cacheOffset,
                    rank,
                    rope,
                    npad);
        }
        grid(g, p + "cache_f16", lanes(npad * keyWidth, 256));
        mma(
                g,
                p + "scores",
                state.workspace.absorbedF16Batch,
                state.workspace.keysF16,
                state.workspace.scoresBatch,
                rows,
                npad,
                keyWidth,
                keyWidth,
                keyWidth,
                npad,
                0,
                0,
                0,
                1,
                1,
                0);
        g.task(
                p + "softmax",
                BatchMmaKernels::causalSoftmax,
                context,
                ws.batchStartPosHolder,
                state.workspace.scoresBatch,
                state.workspace.probsF16Batch,
                heads,
                npad,
                config.attentionScale());
        grid(
                g,
                p + "softmax",
                lanes(rows * DeepSeek2BatchKernels.GROUP, DeepSeek2BatchKernels.GROUP));
        mma(
                g,
                p + "attend",
                state.workspace.probsF16Batch,
                state.workspace.latentTF16,
                state.workspace.latentBatch,
                rows,
                rank,
                npad,
                npad,
                npad,
                rank,
                0,
                0,
                0,
                1,
                0,
                1);
        toHalf(
                g,
                p + "latent_f16",
                state.workspace.latentBatch,
                state.workspace.latentF16Batch,
                rows * rank);
        // Decompress: per head, out[t][h][j] = v_b[h][j] . latent[t][h].
        mma(
                g,
                p + "decompress",
                state.workspace.latentF16Batch,
                vBF16[l],
                state.workspace.attnOutBatch,
                batch,
                valueHead,
                rank,
                heads * rank,
                rank,
                attnOut,
                rank,
                valueHead * rank,
                valueHead,
                heads);
        quantize(g, p + "attn_out_q8", state.workspace.attnOutBatch, attnOut);
        gemm(
                g,
                p + "wo",
                w.wo()[l].asByteArray(),
                ws.wrapXBatch,
                dim,
                attnOut,
                Int8GemmKernels.EPILOGUE_RESIDUAL,
                true);

        // ── feed-forward ──
        rmsRows(
                g,
                p + "ffn_norm",
                ws.wrapXBatch,
                ws.wrapNormedBatch,
                w.ffnNorm()[l],
                dim,
                dim,
                dim,
                eps);
        quantize(g, p + "ffn_q8", ws.wrapNormedBatch, dim);
        if (config.isDenseLayer(l)) {
            int hidden = config.hiddenDim();
            gemm(
                    g,
                    p + "ffn_gate",
                    w.ffnGate()[l].asByteArray(),
                    state.workspace.gateBatch,
                    hidden,
                    dim,
                    Int8GemmKernels.EPILOGUE_STORE,
                    false);
            gemmGated(
                    g,
                    p + "ffn_up",
                    w.ffnUp()[l].asByteArray(),
                    state.workspace.hiddenBatch,
                    state.workspace.gateBatch,
                    hidden,
                    dim);
            quantize(g, p + "ffn_hidden_q8", state.workspace.hiddenBatch, hidden);
            gemm(
                    g,
                    p + "ffn_down",
                    w.ffnDown()[l].asByteArray(),
                    ws.wrapXBatch,
                    dim,
                    hidden,
                    Int8GemmKernels.EPILOGUE_RESIDUAL,
                    true);
        } else {
            experts(g, p, l);
        }

        if (l == endLayer - 1 || (l - firstLayer) % LAYERS_PER_GRAPH == LAYERS_PER_GRAPH - 1) {
            g.persistOnDevice(ws.wrapXBatch, cache());
        }
    }

    private void experts(TaskGraph g, String p, int l) {
        var ws = state.workspace;
        int dim = config.dim();
        int experts = config.expertCount();
        int used = config.expertsUsed();
        int hidden = config.expertHiddenDim();
        int shared = config.sharedHiddenDim();
        int assignments = batch * used;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, experts);
        FloatArray bias = w.routerBias()[l] != null ? w.routerBias()[l].asFloatArray() : zeroBias;

        g.task(
                p + "router",
                Qwen35MoeBatchKernels::routerTiled,
                context,
                ws.wrapNormedBatch,
                w.router()[l].asFloatArray(),
                ws.wrapMoeLogitsBatch,
                ws.batchStartPosHolder,
                dim,
                experts);
        WorkerGrid router =
                new WorkerGrid2D(
                        batch / Qwen35MoeBatchKernels.ROUTER_TILE * 256,
                        experts / Qwen35MoeBatchKernels.ROUTER_TILE);
        router.setLocalWork(256, 1, 1);
        grid(g, p + "router", router);
        g.task(
                p + "topk",
                DeepSeek2BatchKernels::routerTopKSigmoidBatch,
                context,
                ws.wrapMoeLogitsBatch,
                bias,
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedGateBatch,
                ws.batchStartPosHolder,
                experts,
                used,
                config.expertWeightsNorm() ? 1 : 0,
                config.expertWeightsScale());
        grid(g, p + "topk", lanes(batch * 32, 32));
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
        grid(
                g,
                p + "group",
                lanes(Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS));
        g.task(
                p + "gate_up",
                Qwen35MoeBatchKernels::groupedGateUpQ8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                ws.wrapMoeSortedToken,
                ws.wrapMoeTiles,
                w.gateExperts()[l].asByteArray(),
                w.upExperts()[l].asByteArray(),
                ws.wrapMoeHiddenBatch,
                dim,
                hidden);
        WorkerGrid gateUp =
                new WorkerGrid2D(
                        hidden
                                / Qwen35MoeBatchKernels.TILE_COLS
                                * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        gateUp.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        grid(g, p + "gate_up", gateUp);
        g.task(
                p + "hidden_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                hidden);
        grid(g, p + "hidden_q8", lanes(assignments * hidden, 128));
        g.task(
                p + "down",
                Qwen35MoeBatchKernels::groupedDownQ8_0,
                context,
                ws.wrapMoeHiddenQ8,
                ws.wrapMoeHiddenScales,
                ws.wrapMoeTiles,
                w.downExperts()[l].asByteArray(),
                ws.wrapMoeOutBatch,
                dim,
                hidden);
        WorkerGrid down =
                new WorkerGrid2D(
                        dim / Qwen35MoeBatchKernels.TILE_COLS * Qwen35MoeBatchKernels.GEMM_THREADS,
                        maxTiles);
        down.setLocalWork(Qwen35MoeBatchKernels.GEMM_THREADS, 1, 1);
        grid(g, p + "down", down);

        // The shared expert over every row of the chunk; its weight is one.
        gemm(
                g,
                p + "shared_gate",
                w.sharedGate()[l].asByteArray(),
                ws.wrapMoeSharedGateUp,
                shared,
                dim,
                Int8GemmKernels.EPILOGUE_STORE,
                false);
        gemmGated(
                g,
                p + "shared_up",
                w.sharedUp()[l].asByteArray(),
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedGateUp,
                shared,
                dim);
        g.task(
                p + "shared_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                ws.wrapMoeSharedHidden,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                shared);
        grid(g, p + "shared_q8", lanes(batch * shared, 128));
        g.task(
                p + "shared_down",
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapMoeSharedQ8,
                ws.wrapMoeSharedScales,
                w.sharedDown()[l].asByteArray(),
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedOut,
                batch,
                dim,
                shared,
                Int8GemmKernels.EPILOGUE_STORE,
                ws.wrapMoeSharedOut,
                1,
                ws.batchStartPosHolder,
                dim,
                0);
        grid(g, p + "shared_down", gemmGrid(dim, 1));
        g.task(
                p + "combine",
                Qwen35MoeBatchKernels::combine,
                context,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapMoeSharedOut,
                ws.wrapMoeSharedGateBatch,
                ws.wrapXBatch,
                ws.batchStartPosHolder,
                dim,
                used);
        grid(g, p + "combine", lanes(batch * dim, 256));
    }

    // ── building blocks ───────────────────────────────────────────────────────

    private void grid(TaskGraph g, String task, WorkerGrid grid) {
        grids.put(g.getTaskGraphName() + "." + task, grid);
    }

    private void rmsRows(
            TaskGraph g,
            String task,
            FloatArray in,
            FloatArray out,
            TornadoTensor weight,
            int n,
            int inStride,
            int outStride,
            float eps) {
        g.task(
                task,
                DeepSeek2BatchKernels::rmsNormRows,
                context,
                in,
                out,
                weight.asFloatArray(),
                state.workspace.batchStartPosHolder,
                n,
                inStride,
                outStride,
                eps);
        grid(g, task, lanes(batch * DeepSeek2BatchKernels.GROUP, DeepSeek2BatchKernels.GROUP));
    }

    /** The chunk's {@code k}-wide rows of {@code x} in int8 blocks, for the GEMM after. */
    private void quantize(TaskGraph g, String task, FloatArray x, int k) {
        var ws = state.workspace;
        g.task(
                task,
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                x,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                k);
        grid(g, task, lanes(batch * k, 128));
    }

    private void toHalf(TaskGraph g, String task, FloatArray in, HalfFloatArray out, int n) {
        g.task(task, BatchMmaKernels::toHalf, context, in, out, n);
        grid(g, task, lanes(n, 256));
    }

    /** {@code out (+)= x8 . w} over the chunk; {@code n} outputs over {@code k} inputs. */
    private void gemm(
            TaskGraph g,
            String task,
            ByteArray weight,
            FloatArray out,
            int n,
            int k,
            int epilogue,
            boolean split) {
        var ws = state.workspace;
        int splits = split ? gemmSplits(n, k) : 1;
        FloatArray partial = splits > 1 ? ws.wrapQ8SplitPartial : out;
        g.task(
                task,
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                weight,
                out,
                out,
                batch,
                n,
                k,
                epilogue,
                partial,
                splits,
                ws.batchStartPosHolder,
                n,
                0);
        grid(g, task, gemmGrid(n, splits));
        if (splits > 1) {
            int total = batch * n;
            g.task(
                    task + "_reduce",
                    Int8GemmKernels::reduceSplitsQ8_0,
                    context,
                    ws.wrapQ8SplitPartial,
                    out,
                    total,
                    splits,
                    epilogue,
                    ws.batchStartPosHolder,
                    n);
            grid(g, task + "_reduce", lanes(total, 256));
        }
    }

    /** {@code out = silu(gate) * (x8 . w)} over the chunk. */
    private void gemmGated(
            TaskGraph g,
            String task,
            ByteArray weight,
            FloatArray out,
            FloatArray gate,
            int n,
            int k) {
        var ws = state.workspace;
        g.task(
                task,
                Int8GemmKernels::gemmInt8Q8_0,
                context,
                ws.wrapQ8ActBatch,
                ws.wrapQ8ActScales,
                weight,
                out,
                gate,
                batch,
                n,
                k,
                Int8GemmKernels.EPILOGUE_SWIGLU,
                out,
                1,
                ws.batchStartPosHolder,
                n,
                0);
        grid(g, task, gemmGrid(n, 1));
    }

    private void mma(
            TaskGraph g,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols,
            int batches) {
        mma(g, task, a, b, c, m, n, k, lda, ldb, ldc, aBatch, bBatch, cBatchCols, batches, 0, 0);
    }

    /** {@link #mma} with N or K bounded by the chunk's last position, rounded to whole tiles. */
    private void mma(
            TaskGraph g,
            String task,
            HalfFloatArray a,
            HalfFloatArray b,
            FloatArray c,
            int m,
            int n,
            int k,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols,
            int batches,
            int boundN,
            int boundK) {
        g.task(
                task,
                BatchMmaKernels::gemmMMAStrided,
                context,
                a,
                b,
                c,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                aBatch,
                bBatch,
                cBatchCols,
                state.workspace.batchStartPosHolder,
                boundN,
                boundK,
                0,
                0,
                1,
                0);
        WorkerGrid3D grid = new WorkerGrid3D((m / 128) * 256, n / 128, batches);
        grid.setLocalWork(256, 1, 1);
        grid(g, task, grid);
    }

    private WorkerGrid gemmGrid(int n, int splits) {
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        (batch / Int8GemmKernels.I8_BM) * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN * splits);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        return grid;
    }

    private int gemmSplits(int outputs, int k) {
        int sms = smCount();
        if (sms <= 0) {
            return 1;
        }
        int tiles = (batch / Int8GemmKernels.I8_BM) * (outputs / Int8GemmKernels.I8_BN);
        int rounds = k / Int8GemmKernels.I8_BK;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= 8 && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return bestCost * 10 <= unsplit * 9 ? best : 1;
    }

    /** The streaming multiprocessors of the device this plan is built for; 0 when unknown. */
    private static int smCount() {
        return org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
    }

    private static WorkerGrid lanes(int count, int local) {
        int global = (count + local - 1) / local * local;
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
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
}
