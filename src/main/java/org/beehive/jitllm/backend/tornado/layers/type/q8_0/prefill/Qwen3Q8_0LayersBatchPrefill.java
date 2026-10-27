package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Batched-prefill transformer-layer TaskGraphs for the Qwen3 Q8_0 unified batched prefill-decode
 * plan.
 *
 * <p>Q8_0 path: wrapXbBatch (FP32) holds normalized activations; wrapXbFP16Batch is not used.
 * Mirrors {@link Qwen3FP16LayersBatchPrefill} but uses Q8_0 weights (ByteArray) and FP32 attention
 * normalization path.
 */
public class Qwen3Q8_0LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    @Override
    public String describeProjections() {
        return mlxProjections != null
                ? "MLX GEMMs"
                : BatchPrefillTransformerLayerTaskGraphs.super.describeProjections();
    }

    @Override
    public String describeAttention() {
        return mlxAttention != null
                ? "MLX"
                : BatchPrefillTransformerLayerTaskGraphs.super.describeAttention();
    }

    @Override
    public String describeNativeLibraries() {
        return mlxProjections != null || mlxAttention != null ? "MLX" : "none";
    }

    static final int LOCAL_WORK_GROUP_SIZE = 32;

    /** Lanes per workgroup of the tiled K-quant projections, as Qwen3.5's batched prefill uses. */
    private static final int TILED_LOCAL = 128;

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final MlxPrefillAttention mlxAttention;
    private final MlxPrefillProjections mlxProjections;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen3Q8_0LayersBatchPrefill(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdHead = nEmbdHeadV;
        this.qDim = nEmbdHeadK * config.numberOfHeads();
        this.kvDim = nEmbdHeadV * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
        Optional<MlxPrefill> mlx =
                MlxPrefill.resolve(state.executionPolicy(), weights.dataType(), config);
        this.mlxAttention =
                MlxPrefillAttention.ifPresent(
                        mlx,
                        state,
                        context,
                        batchSize,
                        config.contextLength(),
                        config.numberOfHeads(),
                        nHeadKv,
                        nEmbdHead);
        this.mlxProjections =
                MlxPrefillProjections.ifPresent(
                        mlx,
                        state,
                        weights,
                        context,
                        batchSize,
                        config.numberOfLayers(),
                        config.dim(),
                        qDim,
                        kvDim,
                        config.hiddenDim());
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = GraphNames.batchPrefillLayer(layerIndex);
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph layer = new TaskGraph(graphName);
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache());
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
        } else {
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch);
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        if (mlxAttention != null) {
            mlxAttention.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }

        // Per-layer weights (Q8_0 format)
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());

        // ── Attention Block ────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

        if (mlxProjections != null) {
            mlxProjections.qkv(layer, layerIndex);
        } else {
            // FP32 normalize into wrapXbBatch (Q8_0 path: no FP16 quantize step)
            layer.task(
                    "batch_attn_rms_apply",
                    TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.attnScaleBatch,
                    dim);

            if (kQuantLayer(layerIndex)) {
                // One projection each: wq, wk and wv can be different representations.
                kQuantProjection(
                        layer,
                        "batch_q",
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapQBatch,
                        weights.wqLayered[layerIndex],
                        dim,
                        qDim,
                        false);
                kQuantProjection(
                        layer,
                        "batch_k",
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapKBatch,
                        weights.wkLayered[layerIndex],
                        dim,
                        kvDim,
                        false);
                kQuantProjection(
                        layer,
                        "batch_v",
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapVBatch,
                        weights.wvLayered[layerIndex],
                        dim,
                        kvDim,
                        false);
            } else {
                layer.task(
                        "batch_qkv",
                        tiledQkv()
                                ? (simdgroupGemm()
                                        ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8
                                        : TransformerBatchPrefillKernels::batchedGemmQKVQ8)
                                : Qwen3Kernels::batchedFusedQKVMatmulQ8_0,
                        context,
                        state.workspace.wrapXbBatch,
                        state.workspace.wrapQBatch,
                        state.workspace.wrapKBatch,
                        state.workspace.wrapVBatch,
                        weights.wqLayered[layerIndex].asByteArray(),
                        weights.wkLayered[layerIndex].asByteArray(),
                        weights.wvLayered[layerIndex].asByteArray(),
                        dim,
                        qDim,
                        kvDim,
                        tiledQkv() ? batchSize : LOCAL_WORK_GROUP_SIZE);
            }
        }

        layer.task(
                "batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNorm,
                context,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                nHeadKv,
                nEmbdHead,
                qDim,
                kvDim,
                config.rmsNormEps());

        if (useFp16KVCache()) {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3FP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_rope_kv",
                    Qwen3PagedKvKernels::batchedRopeWithKVCacheQwen3Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    config.ropeTheta(),
                    kvDim,
                    nEmbdHead,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        // Reuses batchedFlashAttention; passes qDim as the 'dim' stride (valid: qDim==dim
        // typically).
        if (mlxAttention != null) {
            mlxAttention.attention(layer, "batch_attention", layerIndex);
        } else if (useFp16KVCache()) {
            layer.task(
                    "batch_attention",
                    simdgroupAttention()
                            ? (nEmbdHead == 64
                                    ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                            ::batchedFlashAttentionKVFP16PagedHead64
                                    : nEmbdHead == 96
                                            ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead96
                                            : TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead128)
                            : TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        } else {
            layer.task(
                    "batch_attention",
                    simdgroupAttention()
                            ? TransformerPagedKvBatchPrefillSimdgroupKernels
                                    ::batchedFlashAttentionPagedHead128
                            : laneAttention()
                                    ? TransformerPagedKvBatchPrefillKernels
                                            ::batchedFlashAttentionPagedLaneHead128
                                    : TransformerPagedKvBatchPrefillKernels
                                            ::batchedFlashAttentionPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    nEmbdHead,
                    kvDim,
                    gqa,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    qDim);
        }

        // Output projection (Q8_0): n=qDim, d=dim
        if (mlxProjections != null) {
            mlxProjections.attentionOutput(layer, layerIndex);
        } else if (kQuantLayer(layerIndex)) {
            kQuantProjection(
                    layer,
                    "batch_attn_out",
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex],
                    qDim,
                    dim,
                    true);
        } else {
            layer.task(
                    "batch_attn_out",
                    tiledGemm()
                            ? (simdgroupGemm()
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmQ8WithResidual
                                    : TransformerBatchPrefillKernels::batchedGemmQ8WithResidual)
                            : TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asByteArray(),
                    qDim,
                    dim,
                    tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);
        }

        // ── FFN Block ──────────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        if (mlxProjections != null) {
            mlxProjections.ffn(layer, layerIndex);
        } else if (kQuantLayer(layerIndex)) {
            kQuantGateUp(layer, layerIndex, dim, hidDim);
            kQuantProjection(
                    layer,
                    "batch_ffn_down",
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex],
                    hidDim,
                    dim,
                    true);
        } else {
            layer.task(
                    "batch_ffn_gate_up",
                    tiledGemm()
                            ? (simdgroupGemm()
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmRmsNormFFNGateUpQ8
                                    : TransformerBatchPrefillKernels::batchedGemmRmsNormFFNGateUpQ8)
                            : TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUpQ8,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    hidDim,
                    tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);

            layer.task(
                    "batch_ffn_down",
                    tiledGemm()
                            ? (simdgroupGemm()
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmQ8WithResidual
                                    : TransformerBatchPrefillKernels::batchedGemmQ8WithResidual)
                            : TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex].asByteArray(),
                    hidDim,
                    dim,
                    tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);
        }

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        if (mlxAttention != null) {
            mlxAttention.persist(layer);
        }
        if (mlxProjections != null) {
            mlxProjections.persist(layer);
        }

        return layer;
    }

    // @formatter:on

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        int qkvRows = qDim + 2 * kvDim;
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (config.numberOfHeads() + nHeadKv) * nEmbdHead, nEmbdHead);

        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int optLocal =
                laneAttention()
                        ? 32 * LaneAttentionPolicy.PREFILL_WARPS_PER_GROUP
                        : WorkerGridFactory.findOptimalLocalSize(nEmbdHead);
        int attnTiles =
                (batchSize + TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE - 1)
                        / TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE;
        WorkerGrid attnWorker =
                simdgroupAttention()
                        ? WorkerGridFactory.genericWorker(
                                attnTiles
                                        * config.numberOfHeads()
                                        * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                                TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS)
                        : WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfHeads() * optLocal, optLocal);

        int gemmLocal = simdgroupGemm() ? TransformerBatchPrefillSimdgroupKernels.THREADS : 256;
        int hidTile = simdgroupGemm() ? 32 : 64;
        WorkerGrid gemmDimWorker =
                WorkerGridFactory.genericWorker(
                        ((dim + 63) / 64) * ((batchSize + 63) / 64) * gemmLocal, gemmLocal);
        WorkerGrid gemmHidWorker =
                WorkerGridFactory.genericWorker(
                        ((hidDim + hidTile - 1) / hidTile) * ((batchSize + 63) / 64) * gemmLocal,
                        gemmLocal);
        WorkerGrid gemmQkvWorker =
                WorkerGridFactory.genericWorker(
                        ((qDim + 2 * kvDim) / 64) * ((batchSize + 63) / 64) * gemmLocal, gemmLocal);
        WorkerGrid matVecDimWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid matVecHidWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = GraphNames.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            if (mlxProjections != null) {
                mlxProjections.addGrids(scheduler, p);
            } else if (kQuantLayer(i)) {
                scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
                scheduler.addWorkerGrid(p + "batch_q", kQuantWorker(qDim));
                scheduler.addWorkerGrid(p + "batch_k", kQuantWorker(kvDim));
                scheduler.addWorkerGrid(p + "batch_v", kQuantWorker(kvDim));
            } else {
                scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
                scheduler.addWorkerGrid(p + "batch_qkv", tiledQkv() ? gemmQkvWorker : qkvWorker);
            }
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (mlxAttention != null) {
                mlxAttention.addGrids(scheduler, p, "batch_attention");
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            if (mlxProjections == null && kQuantLayer(i)) {
                scheduler.addWorkerGrid(p + "batch_attn_out", kQuantWorker(dim));
            } else if (mlxProjections == null) {
                scheduler.addWorkerGrid(
                        p + "batch_attn_out", tiledGemm() ? gemmDimWorker : matVecDimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            if (mlxProjections == null && kQuantLayer(i)) {
                if (simdgroupGemm()) {
                    scheduler.addWorkerGrid(p + "batch_ffn_gate_up", gemmHidWorker);
                } else {
                    scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApplyWorker);
                    scheduler.addWorkerGrid(p + "batch_ffn_gate_up", kQuantWorker(hidDim));
                }
                scheduler.addWorkerGrid(p + "batch_ffn_down", kQuantWorker(dim));
            } else if (mlxProjections == null) {
                scheduler.addWorkerGrid(
                        p + "batch_ffn_gate_up", tiledGemm() ? gemmHidWorker : matVecHidWorker);
                scheduler.addWorkerGrid(
                        p + "batch_ffn_down", tiledGemm() ? gemmDimWorker : matVecDimWorker);
            }
        }
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }

    /**
     * Whether a layer reads its projections as K-quants: a Q4_K_M file whose tensors are kept as
     * the file stores them. Each projection then runs on its own tensor's kernel, on SIMD-group
     * matrices where {@link #simdgroupGemm()} holds and as tiled batch projections elsewhere.
     */
    private boolean kQuantLayer(int layerIndex) {
        return weights.wqLayered[layerIndex].dataType() == DataType.Q4_K;
    }

    /** {@code out = w·x} over the batch, or {@code out += w·x} with {@code residual}. */
    private void kQuantProjection(
            TaskGraph layer,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d,
            boolean residual) {
        boolean q4k = w.dataType() == DataType.Q4_K;
        if (simdgroupGemm()) {
            layer.task(
                    name,
                    residual
                            ? (q4k
                                    ? TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmQ4_KWithResidual
                                    : TransformerBatchPrefillSimdgroupKernels
                                            ::batchedGemmQ6_KWithResidual)
                            : (q4k
                                    ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_K
                                    : TransformerBatchPrefillSimdgroupKernels::batchedGemmQ6_K),
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    batchSize);
        } else {
            layer.task(
                    name,
                    residual
                            ? (q4k
                                    ? TransformerComputeKernelsQ4_K
                                            ::matrixVectorTiledBatchWithResidualQ4_K
                                    : TransformerComputeKernelsQ6_K
                                            ::matrixVectorTiledBatchWithResidualQ6_K)
                            : (q4k
                                    ? TransformerComputeKernelsQ4_K::matrixVectorTiledBatchQ4_K
                                    : TransformerComputeKernelsQ6_K::matrixVectorTiledBatchQ6_K),
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    batchSize,
                    TILED_LOCAL);
        }
    }

    /**
     * {@code hb = SiLU(w1·x) * (w3·x)} with {@code x} the FFN-normalized activations: the norm
     * applied as the SIMD-group kernel stages them, or by its own task before the tiled one.
     */
    private void kQuantGateUp(TaskGraph layer, int layerIndex, int dim, int hidDim) {
        if (simdgroupGemm()) {
            layer.task(
                    "batch_ffn_gate_up",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ4_K,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    hidDim,
                    batchSize);
            return;
        }
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        layer.task(
                "batch_ffn_gate_up",
                TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUTiledBatchQ4_K,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapHbBatch,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                dim,
                hidDim,
                batchSize,
                TILED_LOCAL);
    }

    /** The grid of a K-quant projection to {@code rows} outputs, SIMD-group or tiled. */
    private WorkerGrid kQuantWorker(int rows) {
        int batchTiles = (batchSize + 63) / 64;
        if (simdgroupGemm()) {
            int threads = TransformerBatchPrefillSimdgroupKernels.THREADS;
            return WorkerGridFactory.genericWorker(
                    ((rows + 63) / 64) * batchTiles * threads, threads);
        }
        int rowGroups =
                (batchSize + TransformerComputeKernelsQ4_K.rowTile() - 1)
                        / TransformerComputeKernelsQ4_K.rowTile();
        int colGroups =
                (rows + TransformerComputeKernelsQ4_K.colTile() - 1)
                        / TransformerComputeKernelsQ4_K.colTile();
        return WorkerGridFactory.genericWorker(rowGroups * colGroups * TILED_LOCAL, TILED_LOCAL);
    }

    /**
     * Whether the cache is half precision: the representation the state holds, which the decode
     * layers after this prefill bind as well.
     */
    private boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache();
    }

    /** The key cache every graph of this family binds: FP16 when the state holds one. */
    private Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache, following {@link #keyCache()}. */
    private Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    /** The tiled projection kernels; see {@link BatchPrefillGemmPolicy}. */
    private static boolean tiledGemm() {
        return BatchPrefillGemmPolicy.tiled();
    }

    /**
     * The tiled projections on SIMD-group matrices; see {@link BatchPrefillGemmPolicy}. Their
     * double-buffered k-loop needs every contraction length to be a multiple of 64.
     */
    private boolean simdgroupGemm() {
        return BatchPrefillGemmPolicy.simdgroup()
                && config.dim() % 64 == 0
                && config.hiddenDim() % 64 == 0
                && qDim % 64 == 0;
    }

    /** The tiled QKV kernel needs every 64-row tile to lie in one of Q, K and V. */
    private boolean tiledQkv() {
        return tiledGemm() && qDim % 64 == 0 && kvDim % 64 == 0;
    }

    /** The lane-cooperative kernel over the FP32 cache; see {@link LaneAttentionPolicy}. */
    private boolean laneAttention() {
        return !useFp16KVCache()
                && LaneAttentionPolicy.laneCooperativeAttentionFp32Cache(nEmbdHead);
    }

    /**
     * Attention on SIMD-group matrices; see {@link BatchPrefillGemmPolicy}. Over the FP32 cache
     * where the lane-cooperative kernel would otherwise run; over the FP16 cache for every head
     * width the FP16 SIMD-group kernels are written for.
     */
    private boolean simdgroupAttention() {
        return useFp16KVCache()
                ? BatchPrefillGemmPolicy.simdgroupAttention(nEmbdHead)
                : laneAttention() && BatchPrefillGemmPolicy.simdgroupAttention();
    }
}
