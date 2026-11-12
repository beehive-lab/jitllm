package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.GroupedExpertSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.Lfm2BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35BatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The {@code lfm2moe} layers over a prompt chunk, one task graph per layer: the chunk's rows in
 * {@code wrapXBatch}, their keys and values appended to the cache, the convolution windows carried
 * to where decode continues.
 *
 * <p>The same two choices per layer as {@link Lfm2Layers}: attention or the short convolution,
 * dense or routed feed-forward. The dense projections are SIMD-group GEMMs ({@link
 * QuantizedBatchProjections}); attention, the per-head norms and the rotation are the kernels the
 * Qwen3.5 prefill runs on Metal. The routed feed-forward sorts the chunk's assignments by expert
 * ({@code Qwen35MoeBatchKernels.groupByExpert}) and runs one GEMM per tile of an expert's rows
 * ({@link GroupedExpertSimdgroupKernels}), so each selected expert's weights are read once per tile
 * rather than once per token: a chunk of 512 tokens over 64 experts gives each about 32 rows.
 *
 * <p>On Metal the GEMMs are SIMD-group matrices; elsewhere they are the tiled batch kernels,
 * decoding each weight once per eight rows, with the experts' as grouped tiled kernels. The narrow
 * chunk family is declined.
 */
// @formatter:on
public class Lfm2BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    private static final int ELEMENTWISE_LOCAL = 128;

    /** Lanes of a tiled (non-SIMD-group) batch projection or grouped expert GEMM. */
    private static final int TILED_LOCAL = 128;

    /** Whether the GEMMs are SIMD-group matrices (Metal) rather than tiled batch kernels. */
    private final boolean simdgroup = BatchPrefillGemmPolicy.simdgroup();

    private static final int ATTENTION_LOCAL = Qwen35BatchKernels.ATTENTION_STAGE_LANES;

    private final Lfm2State state;
    private final Lfm2TornadoWeights weights;
    private final Lfm2Configuration config;
    private final int batchSize;
    private final KernelContext context = new KernelContext();
    private final QuantizedBatchProjections projections;
    private final List<ImmutableTaskGraph> graphs;
    private final String lastLayerTaskGraphID;

    public Lfm2BatchPrefillLayers(
            Lfm2State state, Lfm2TornadoWeights weights, Lfm2Configuration config, int batchSize) {
        PrefillFamily family = PrefillFamily.of(state);
        if (family.isNarrow()) {
            throw PrefillFamily.decline("the lfm2moe layers build no narrow family yet");
        }
        if (config.expertHiddenDim() > config.dim()) {
            throw new UnsupportedOperationException(
                    "the batched lfm2moe experts keep their up rows in the output rows, which"
                            + " needs an expert width no larger than the model's");
        }
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.projections =
                new QuantizedBatchProjections(
                        context, state.workspace, batchSize, simdgroup, true, TILED_LOCAL);
        this.graphs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::buildLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
        this.lastLayerTaskGraphID = GraphNames.batchPrefillLayer(config.numberOfLayers() - 1);
    }

    /**
     * The bytes {@code Lfm2State} allocates for a chunk of {@code batch} rows beyond the staging
     * every family has: the normalized chunk, the convolution's, and the mixture's routing and
     * rows. Example: LFM2.5-24B-A2B at 512 rows, about 45 MB.
     */
    public static long workspaceBytes(Lfm2Configuration config, int batch) {
        long dim = config.dim();
        long assignments = (long) batch * config.expertUsedCount();
        long hidden = config.expertHiddenDim();
        long floats =
                batch * dim
                        + (long) batch * config.shortConvProjectionDim()
                        + batch * dim
                        + (long) batch * config.expertCount()
                        + assignments
                        + assignments * hidden
                        + assignments * Math.max(dim, hidden);
        long ints =
                3 * assignments
                        + 1
                        + 3L
                                * Qwen35MoeBatchKernels.maxTiles(
                                        Math.toIntExact(assignments), config.expertCount());
        return 4 * (floats + ints);
    }

    @Override
    public String describeProjections() {
        return "JIT kernels (SIMD-group GEMMs, grouped expert GEMMs)";
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    private boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    private Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    private static TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "lfm2moe layer " + layer + " must carry " + role + " for a batched prefill");
        }
        return tensor;
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    private TaskGraph buildLayer(int layerIndex) {
        var ws = state.workspace;
        TaskGraph layer = new TaskGraph(GraphNames.batchPrefillLayer(layerIndex));
        String predecessor =
                layerIndex == 0
                        ? GraphNames.PREFILL_ACTIVATION
                        : GraphNames.batchPrefillLayer(layerIndex - 1);
        layer.consumeFromDevice(predecessor, ws.wrapXBatch);
        configureTransfers(layer, layerIndex, predecessor);
        bindLayerWeights(layer, layerIndex);

        rmsReduce(layer, "attn_rms_reduce", ws.attnScaleBatch);
        layer.task(
                "attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm").asFloatArray(),
                ws.attnScaleBatch,
                config.dim());

        if (config.isAttentionLayer(layerIndex)) {
            attentionBranch(layer, layerIndex);
        } else {
            shortConvBranch(layer, layerIndex);
        }

        rmsReduce(layer, "ffn_rms_reduce", ws.ffnScaleBatch);
        if (config.isMoeLayer(layerIndex)) {
            mixtureOfExperts(layer, layerIndex);
        } else {
            projections.gateUp(
                    layer,
                    require(weights.w1Layered, layerIndex, "ffn_gate"),
                    require(weights.w3Layered, layerIndex, "ffn_up"),
                    require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm").asFloatArray(),
                    config.dim(),
                    config.hiddenDim());
            projections.projection(
                    layer,
                    "ffn_down_proj",
                    ws.wrapHbBatch,
                    ws.wrapXBatch,
                    require(weights.w2Layered, layerIndex, "ffn_down"),
                    config.hiddenDim(),
                    config.dim(),
                    true);
        }

        layer.persistOnDevice(
                ws.wrapXBatch, keyStore(), valueStore(), ws.wrapBlockTable, ws.wrapConvState);
        return layer;
    }

    private void rmsReduce(TaskGraph layer, String name, FloatArray scale) {
        layer.task(
                name,
                Qwen35BatchKernels::rmsReduceBatchGroup,
                context,
                state.workspace.wrapXBatch,
                scale,
                config.dim(),
                config.rmsNormEps());
    }

    /** Attention over the chunk, from {@code wrapNormedBatch} back into {@code wrapXBatch}. */
    private void attentionBranch(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        final int dim = config.dim();
        final int kvDim = config.kvDim();
        final int headDim = config.headSize();
        store(
                layer,
                "attn_q_proj",
                ws.wrapNormedBatch,
                ws.wrapQBatch,
                require(weights.wqLayered, layerIndex, "attn_q"),
                dim,
                dim);
        store(
                layer,
                "attn_k_proj",
                ws.wrapNormedBatch,
                ws.wrapKBatch,
                require(weights.wkLayered, layerIndex, "attn_k"),
                dim,
                kvDim);
        store(
                layer,
                "attn_v_proj",
                ws.wrapNormedBatch,
                ws.wrapVBatch,
                require(weights.wvLayered, layerIndex, "attn_v"),
                dim,
                kvDim);
        layer.task(
                "attn_qk_norm",
                Qwen35BatchKernels::fusedQKRmsNormBatchGroup,
                context,
                ws.wrapQBatch,
                ws.wrapKBatch,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.rmsNormEps(),
                ws.batchStartPosHolder);
        layer.task(
                "attn_rope",
                Qwen35BatchKernels::ropeNeoxPartialBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapQBatch,
                ws.wrapKBatch,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim);
        int kvLayer = config.keyValueLayerIndex(layerIndex);
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchFP16Paged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchFP16Paged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchPaged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapKBatch,
                    ws.wrapVBatch,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchPaged,
                    context,
                    ws.batchStartPosHolder,
                    ws.wrapQBatch,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        }
        projections.projection(
                layer,
                "attn_output_proj",
                ws.wrapXbBatch,
                ws.wrapXBatch,
                require(weights.woLayered, layerIndex, "attn_output"),
                dim,
                dim,
                true);
    }

    /**
     * {@code out = w · x} over the chunk. A Q8_0 weight off SIMD-group matrices has only the tiled
     * GEMM that adds into its output, so the output is zeroed first.
     */
    private void store(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        if (!simdgroup && w.dataType() == DataType.Q8_0) {
            layer.task(
                    name + "_zero",
                    Lfm2BatchKernels::zeroRows,
                    context,
                    state.workspace.batchStartPosHolder,
                    out,
                    d);
            projections.projection(layer, name, x, out, w, n, d, true);
        } else {
            projections.projection(layer, name, x, out, w, n, d, false);
        }
    }

    /** The gated short convolution over the chunk, rows in order, into {@code wrapXBatch}. */
    private void shortConvBranch(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        store(
                layer,
                "conv_in_proj",
                ws.wrapNormedBatch,
                ws.wrapSsmQkvBatch,
                require(weights.convInProj, layerIndex, "shortconv.in_proj"),
                config.dim(),
                config.shortConvProjectionDim());
        layer.task(
                "conv",
                Lfm2BatchKernels::shortConvBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapSsmQkvBatch,
                require(weights.convKernel, layerIndex, "shortconv.conv").asFloatArray(),
                ws.wrapConvState,
                ws.wrapSsmConvOutBatch,
                config.dim(),
                config.shortConvCache(),
                config.convLayerIndex(layerIndex) * config.convStateSize());
        projections.projection(
                layer,
                "conv_out_proj",
                ws.wrapSsmConvOutBatch,
                ws.wrapXBatch,
                require(weights.convOutProj, layerIndex, "shortconv.out_proj"),
                config.dim(),
                config.dim(),
                true);
    }

    /** The routed feed-forward over the chunk, into {@code wrapXBatch}. */
    private void mixtureOfExperts(TaskGraph layer, int layerIndex) {
        var ws = state.workspace;
        final int dim = config.dim();
        final int hidden = config.expertHiddenDim();
        final int used = config.expertUsedCount();
        layer.task(
                "ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                ws.wrapNormedBatch,
                ws.wrapXBatch,
                require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm").asFloatArray(),
                ws.ffnScaleBatch,
                dim);
        layer.task(
                "moe_router",
                Lfm2BatchKernels::routerBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapNormedBatch,
                require(weights.ffnGateInp, layerIndex, "ffn_gate_inp").asFloatArray(),
                ws.wrapMoeLogitsBatch,
                dim,
                config.expertCount());
        layer.task(
                "moe_select",
                Lfm2BatchKernels::selectExpertsSigmoidBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeLogitsBatch,
                require(weights.expProbsBias, layerIndex, "exp_probs_b").asFloatArray(),
                ws.wrapMoeIdsBatch,
                ws.wrapMoeWeightsBatch,
                config.expertCount(),
                used);
        layer.task(
                "moe_group",
                Qwen35MoeBatchKernels::groupByExpert,
                context,
                ws.wrapMoeIdsBatch,
                ws.batchStartPosHolder,
                ws.wrapMoeSortedToken,
                ws.wrapMoePosition,
                ws.wrapMoeTiles,
                config.expertCount(),
                used,
                batchSize);
        TornadoTensor gate = require(weights.ffnGateExps, layerIndex, "ffn_gate_exps");
        TornadoTensor up = require(weights.ffnUpExps, layerIndex, "ffn_up_exps");
        TornadoTensor down = require(weights.ffnDownExps, layerIndex, "ffn_down_exps");
        grouped(layer, "moe_gate", gate, ws.wrapNormedBatch, ws.wrapMoeHiddenBatch, 1, dim, hidden);
        grouped(layer, "moe_up", up, ws.wrapNormedBatch, ws.wrapMoeOutBatch, 1, dim, hidden);
        layer.task(
                "moe_swiglu",
                Lfm2BatchKernels::siluMultiplyAssignments,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeHiddenBatch,
                ws.wrapMoeOutBatch,
                hidden,
                used);
        grouped(layer, "moe_down", down, ws.wrapMoeHiddenBatch, ws.wrapMoeOutBatch, 0, hidden, dim);
        layer.task(
                "moe_combine",
                Lfm2BatchKernels::combineBatch,
                context,
                ws.batchStartPosHolder,
                ws.wrapMoeOutBatch,
                ws.wrapMoePosition,
                ws.wrapMoeWeightsBatch,
                ws.wrapXBatch,
                dim,
                used);
    }

    /** One grouped expert GEMM over the tiles {@code moe_group} wrote. */
    private void grouped(
            TaskGraph layer,
            String name,
            TornadoTensor w,
            FloatArray in,
            FloatArray out,
            int gather,
            int n,
            int d) {
        var ws = state.workspace;
        if (!simdgroup) {
            layer.task(
                    name,
                    switch (w.dataType()) {
                        case Q4_K -> TransformerComputeKernelsQ4_K::groupedExpertTiledBatchQ4_K;
                        case Q5_K -> TransformerComputeKernelsQ5_K::groupedExpertTiledBatchQ5_K;
                        case Q6_K -> TransformerComputeKernelsQ6_K::groupedExpertTiledBatchQ6_K;
                        default ->
                                throw new UnsupportedOperationException(
                                        "no batched lfm2moe expert GEMM for " + w.dataType());
                    },
                    context,
                    in,
                    out,
                    w.asByteArray(),
                    ws.wrapMoeTiles,
                    ws.wrapMoeSortedToken,
                    gather,
                    n,
                    d,
                    TILED_LOCAL);
            return;
        }
        layer.task(
                name,
                switch (w.dataType()) {
                    case Q4_K -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ4_K;
                    case Q5_K -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ5_K;
                    case Q6_K -> GroupedExpertSimdgroupKernels::groupedExpertGemmQ6_K;
                    default ->
                            throw new UnsupportedOperationException(
                                    "no batched lfm2moe expert GEMM for " + w.dataType());
                },
                context,
                in,
                out,
                w.asByteArray(),
                ws.wrapMoeTiles,
                ws.wrapMoeSortedToken,
                gather,
                n,
                d);
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    /**
     * This layer's weights: read in place where the device shares memory with the host, uploaded
     * once otherwise and recorded, so the decode graphs bind the same buffers.
     */
    private void bindLayerWeights(TaskGraph layer, int layerIndex) {
        List<Object> tensors = new ArrayList<>();
        tensors.add(require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));
        tensors.add(require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm"));
        if (config.isMoeLayer(layerIndex)) {
            tensors.add(require(weights.ffnGateInp, layerIndex, "ffn_gate_inp"));
            tensors.add(require(weights.expProbsBias, layerIndex, "exp_probs_b"));
            tensors.add(require(weights.ffnGateExps, layerIndex, "ffn_gate_exps"));
            tensors.add(require(weights.ffnUpExps, layerIndex, "ffn_up_exps"));
            tensors.add(require(weights.ffnDownExps, layerIndex, "ffn_down_exps"));
        } else {
            tensors.add(require(weights.w1Layered, layerIndex, "ffn_gate"));
            tensors.add(require(weights.w2Layered, layerIndex, "ffn_down"));
            tensors.add(require(weights.w3Layered, layerIndex, "ffn_up"));
        }
        if (config.isAttentionLayer(layerIndex)) {
            tensors.add(require(weights.wqLayered, layerIndex, "attn_q"));
            tensors.add(require(weights.wkLayered, layerIndex, "attn_k"));
            tensors.add(require(weights.wvLayered, layerIndex, "attn_v"));
            tensors.add(require(weights.woLayered, layerIndex, "attn_output"));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm"));
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm"));
        } else {
            tensors.add(require(weights.convInProj, layerIndex, "shortconv.in_proj"));
            tensors.add(require(weights.convKernel, layerIndex, "shortconv.conv"));
            tensors.add(require(weights.convOutProj, layerIndex, "shortconv.out_proj"));
        }
        Object[] uploaded =
                UnifiedMemoryWeights.consume(
                        layer, state.workspace, WeightPrepacker.deviceArrays(tensors.toArray()));
        if (uploaded.length > 0) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, uploaded);
            for (Object w : uploaded) {
                state.workspace.weightBindings.record(w, layer.getTaskGraphName());
            }
        }
    }

    /**
     * The chunk's scratch, uploaded once by layer 0 and bound from the previous layer after it; the
     * chunk holder and block table per chunk. The convolution windows are uploaded once and never
     * read back: they are the session's history, advanced here and then by decode.
     */
    private void configureTransfers(TaskGraph layer, int layerIndex, String predecessor) {
        var ws = state.workspace;
        Object[] scratch = {
            context,
            ws.wrapNormedBatch,
            ws.wrapQBatch,
            ws.wrapKBatch,
            ws.wrapVBatch,
            ws.wrapXbBatch,
            ws.wrapHbBatch,
            ws.ffnUpResult,
            ws.attnScaleBatch,
            ws.ffnScaleBatch,
            ws.wrapSsmQkvBatch,
            ws.wrapSsmConvOutBatch,
            ws.wrapMoeLogitsBatch,
            ws.wrapMoeIdsBatch,
            ws.wrapMoeWeightsBatch,
            ws.wrapMoeSortedToken,
            ws.wrapMoePosition,
            ws.wrapMoeTiles,
            ws.wrapMoeHiddenBatch,
            ws.wrapMoeOutBatch,
            keyStore(),
            valueStore(),
            ws.wrapConvState
        };
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.batchStartPosHolder);
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.wrapBlockTable);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch);
        } else {
            layer.consumeFromDevice(predecessor, scratch);
            layer.consumeFromDevice(predecessor, ws.batchStartPosHolder, ws.wrapBlockTable);
        }
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    /** The grid of a grouped expert GEMM to {@code d} outputs of a tensor in {@code type}. */
    private WorkerGrid groupedWorker(DataType type, int d, int maxTiles) {
        if (simdgroup) {
            int threads = GroupedExpertSimdgroupKernels.THREADS;
            return WorkerGridFactory.genericWorker(((d + 63) / 64) * maxTiles * threads, threads);
        }
        int colTile =
                switch (type) {
                    case Q4_K -> TransformerComputeKernelsQ4_K.colTile();
                    case Q5_K -> TransformerComputeKernelsQ5_K.colTile();
                    default -> TransformerComputeKernelsQ6_K.colTile();
                };
        int subTiles = TransformerComputeKernelsQ4_K.GROUPED_SUB_TILES;
        return WorkerGridFactory.genericWorker(
                maxTiles * subTiles * ((d + colTile - 1) / colTile) * TILED_LOCAL, TILED_LOCAL);
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsReduce =
                WorkerGridFactory.genericWorker(
                        batchSize * Qwen35BatchKernels.RMS_GROUP_LOCAL,
                        Qwen35BatchKernels.RMS_GROUP_LOCAL);
        WorkerGrid rmsApply =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        int headDim = config.headSize();
        int qkHeads = batchSize * (config.numberOfHeads() + config.numberOfKeyValueHeads());
        WorkerGrid qkNorm = WorkerGridFactory.genericWorker(qkHeads * headDim, headDim);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * (headDim / 2), 32);
        WorkerGrid kvAppend =
                WorkerGridFactory.genericWorker(batchSize * config.kvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * ATTENTION_LOCAL, ATTENTION_LOCAL);
        WorkerGrid conv = WorkerGridFactory.genericWorker(config.dim(), ELEMENTWISE_LOCAL);

        int used = config.expertUsedCount();
        int assignments = batchSize * used;
        int maxTiles = Qwen35MoeBatchKernels.maxTiles(assignments, config.expertCount());
        int routerJobs = batchSize * config.expertCount();
        WorkerGrid router =
                WorkerGridFactory.genericWorker(
                        ((routerJobs + Lfm2BatchKernels.ROUTER_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.ROUTER_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.ROUTER_SIMDGROUPS,
                        32 * Lfm2BatchKernels.ROUTER_SIMDGROUPS);
        WorkerGrid select =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Lfm2BatchKernels.SELECT_SIMDGROUPS - 1)
                                        / Lfm2BatchKernels.SELECT_SIMDGROUPS)
                                * 32
                                * Lfm2BatchKernels.SELECT_SIMDGROUPS,
                        32 * Lfm2BatchKernels.SELECT_SIMDGROUPS);
        WorkerGrid group =
                WorkerGridFactory.genericWorker(
                        Qwen35MoeBatchKernels.GROUP_THREADS, Qwen35MoeBatchKernels.GROUP_THREADS);
        int hidden = config.expertHiddenDim();
        WorkerGrid zero =
                WorkerGridFactory.genericWorker(
                        ((batchSize * Math.max(config.dim(), config.shortConvProjectionDim()) + 255)
                                        / 256)
                                * 256,
                        256);
        WorkerGrid swiglu =
                WorkerGridFactory.genericWorker(((assignments * hidden + 255) / 256) * 256, 256);
        WorkerGrid combine =
                WorkerGridFactory.genericWorker(
                        ((batchSize * config.dim() + 255) / 256) * 256, 256);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = GraphNames.batchPrefillLayer(layer) + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            if (config.isAttentionLayer(layer)) {
                scheduler.addWorkerGrid(
                        prefix + "attn_q_proj",
                        projections.worker(weights.wqLayered[layer], config.dim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_k_proj",
                        projections.worker(weights.wkLayered[layer], config.kvDim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_v_proj",
                        projections.worker(weights.wvLayered[layer], config.kvDim()));
                for (String store : new String[] {"attn_q_proj", "attn_k_proj", "attn_v_proj"}) {
                    scheduler.addWorkerGrid(prefix + store + "_zero", zero);
                }
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                scheduler.addWorkerGrid(
                        prefix + "attn_output_proj",
                        projections.worker(weights.woLayered[layer], config.dim()));
            } else {
                scheduler.addWorkerGrid(
                        prefix + "conv_in_proj",
                        projections.worker(
                                weights.convInProj[layer], config.shortConvProjectionDim()));
                scheduler.addWorkerGrid(prefix + "conv_in_proj_zero", zero);
                scheduler.addWorkerGrid(prefix + "conv", conv);
                scheduler.addWorkerGrid(
                        prefix + "conv_out_proj",
                        projections.worker(weights.convOutProj[layer], config.dim()));
            }
            if (config.isMoeLayer(layer)) {
                scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
                scheduler.addWorkerGrid(prefix + "moe_router", router);
                scheduler.addWorkerGrid(prefix + "moe_select", select);
                scheduler.addWorkerGrid(prefix + "moe_group", group);
                WorkerGrid gateUp =
                        groupedWorker(weights.ffnGateExps[layer].dataType(), hidden, maxTiles);
                scheduler.addWorkerGrid(prefix + "moe_gate", gateUp);
                scheduler.addWorkerGrid(prefix + "moe_up", gateUp);
                scheduler.addWorkerGrid(prefix + "moe_swiglu", swiglu);
                scheduler.addWorkerGrid(
                        prefix + "moe_down",
                        groupedWorker(
                                weights.ffnDownExps[layer].dataType(), config.dim(), maxTiles));
                scheduler.addWorkerGrid(prefix + "moe_combine", combine);
            } else {
                TornadoTensor w1 = weights.w1Layered[layer];
                if (projections.fusedGateUp(w1, weights.w3Layered[layer]) && simdgroup) {
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            projections.gateUpWorker(w1, config.hiddenDim()));
                } else if (projections.fusedGateUp(w1, weights.w3Layered[layer])) {
                    // The tiled fused kernels: Q8_0's applies the norm as it stages, the others
                    // after the norm's own task.
                    if (w1.dataType() != DataType.Q8_0) {
                        scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", rmsApply);
                    }
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            projections.worker(w1, config.hiddenDim()));
                } else {
                    scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", rmsApply);
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate", projections.worker(w1, config.hiddenDim()));
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_up",
                            projections.worker(weights.w3Layered[layer], config.hiddenDim()));
                    scheduler.addWorkerGrid(
                            prefix + "batch_ffn_gate_up",
                            WorkerGridFactory.genericWorker(
                                    ((batchSize * config.hiddenDim() + 255) / 256) * 256, 256));
                }
                scheduler.addWorkerGrid(
                        prefix + "ffn_down_proj",
                        projections.worker(weights.w2Layered[layer], config.dim()));
            }
        }
    }
}
