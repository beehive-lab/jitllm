package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan ({@link
 * TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing {@code batchSize} tokens
 * simultaneously via {@link TransformerBatchPrefillKernels}.
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device after every
 * layer so the subsequent single-token decode layers can consume it.
 */
public class LlamaFP16LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

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

    // Matches the local workgroup size used by the single-token kernels.
    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final MlxPrefillAttention mlxAttention;
    private final MlxPrefillProjections mlxProjections;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaFP16LayersBatchPrefill(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
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
                        config.numberOfKeyValueHeads(),
                        config.headSize());
        this.mlxProjections =
                MlxPrefillProjections.ifPresent(
                        mlx,
                        state,
                        weights,
                        context,
                        batchSize,
                        config.numberOfLayers(),
                        config.dim(),
                        config.numberOfHeads() * config.headSize(),
                        config.kvDim(),
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

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            // Allocate persistent GPU-side intermediates once
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache());
            // wrapXBatch produced by the prefillActivation graph and persists in device memory
            // to consume it from there we should use the explicit uniqueTaskGraph name
            // the no-arg form would use current graph name, which causes NPE without CUDA Graphs
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
        } else {
            // for the same reasons as above, we should use the explicit uniqueTaskGraph name to
            // consume
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache(),
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch);
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex - 1), state.workspace.wrapBlockTable);
        }

        if (mlxAttention != null) {
            mlxAttention.declare(
                    batchPrefillLayer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(
                    batchPrefillLayer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }

        // Per-layer weights: upload once
        batchPrefillLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        batchPrefillLayer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

        if (mlxProjections != null) {
            mlxProjections.qkv(batchPrefillLayer, layerIndex);
        } else {
            batchPrefillLayer.task(
                    "batch_attn_rms_apply",
                    TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                    context,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapXBatch,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.attnScaleBatch,
                    dim);

            if (simdgroupGemm()) {

                batchPrefillLayer.task(
                        "batch_qkv",
                        TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFP16,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.wrapQBatch,
                        state.workspace.wrapKBatch,
                        state.workspace.wrapVBatch,
                        weights.wqLayered[layerIndex].asHalfFloatArray(),
                        weights.wkLayered[layerIndex].asHalfFloatArray(),
                        weights.wvLayered[layerIndex].asHalfFloatArray(),
                        dim,
                        dim,
                        kvDim,
                        batchSize);

            } else {

                batchPrefillLayer.task(
                        "batch_qkv",
                        TransformerBatchPrefillKernels::batchedFusedQKVMatmul,
                        context,
                        state.workspace.wrapXbFP16Batch,
                        state.workspace.wrapQBatch,
                        state.workspace.wrapKBatch,
                        state.workspace.wrapVBatch,
                        weights.wqLayered[layerIndex].asHalfFloatArray(),
                        weights.wkLayered[layerIndex].asHalfFloatArray(),
                        weights.wvLayered[layerIndex].asHalfFloatArray(),
                        dim,
                        kvDim,
                        LOCAL_WORK_GROUP_SIZE);
            }
        }

        if (useFp16KVCache()) {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCacheFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        } else {
            batchPrefillLayer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        }

        if (mlxAttention != null) {
            mlxAttention.attention(batchPrefillLayer, "batch_attention", layerIndex);
        } else if (useFp16KVCache()) {
            if (simdgroupAttention()) {
                if (headSize() == 64) {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                    ::batchedFlashAttentionKVFP16PagedHead64,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCacheFP16,
                            state.workspace.wrapValueCacheFP16,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else if (headSize() == 96) {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                    ::batchedFlashAttentionKVFP16PagedHead96,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCacheFP16,
                            state.workspace.wrapValueCacheFP16,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                    ::batchedFlashAttentionKVFP16PagedHead128,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCacheFP16,
                            state.workspace.wrapValueCacheFP16,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                }
            } else {
                batchPrefillLayer.task(
                        "batch_attention",
                        TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionKVFP16Paged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapQBatch,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapXbBatch,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        dim);
            }
        } else {
            if (simdgroupAttention()) {
                if (headSize() == 64) {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupKernels
                                    ::batchedFlashAttentionPagedHead64,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCache,
                            state.workspace.wrapValueCache,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else if (headSize() == 96) {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupKernels
                                    ::batchedFlashAttentionPagedHead96,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCache,
                            state.workspace.wrapValueCache,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else {
                    batchPrefillLayer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupKernels
                                    ::batchedFlashAttentionPagedHead128,
                            context,
                            state.workspace.batchStartPosHolder,
                            state.workspace.wrapQBatch,
                            state.workspace.wrapKeyCache,
                            state.workspace.wrapValueCache,
                            state.workspace.wrapXbBatch,
                            config.numberOfHeads(),
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                }
            } else {
                batchPrefillLayer.task(
                        "batch_attention",
                        TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionPaged,
                        context,
                        state.workspace.batchStartPosHolder,
                        state.workspace.wrapQBatch,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        state.workspace.wrapXbBatch,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        dim);
            }
        }

        if (mlxProjections != null) {
            mlxProjections.attentionOutput(batchPrefillLayer, layerIndex);
        } else if (simdgroupGemm()) {

            batchPrefillLayer.task(
                    "batch_attn_out",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmFP16WithResidual,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    dim,
                    dim,
                    batchSize);

        } else {

            batchPrefillLayer.task(
                    "batch_attn_out",
                    TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    dim,
                    dim,
                    LOCAL_WORK_GROUP_SIZE);
        }

        // ── FFN Block ──────────────────────────────────────────────────────────
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        if (mlxProjections != null) {
            mlxProjections.ffn(batchPrefillLayer, layerIndex);
        } else if (simdgroupGemm()) {

            batchPrefillLayer.task(
                    "batch_ffn_gate_up",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpFP16,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    dim,
                    hidDim,
                    batchSize);

        } else {

            batchPrefillLayer.task(
                    "batch_ffn_gate_up",
                    TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUp,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    dim,
                    hidDim,
                    LOCAL_WORK_GROUP_SIZE);
        }

        if (mlxProjections == null && simdgroupGemm()) {

            batchPrefillLayer.task(
                    "batch_ffn_down",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmFP16WithResidual,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    hidDim,
                    dim,
                    batchSize);

        } else if (mlxProjections == null) {

            batchPrefillLayer.task(
                    "batch_ffn_down",
                    TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    hidDim,
                    dim,
                    LOCAL_WORK_GROUP_SIZE);
        }

        // Persist wrapXBatch for the next layer, and KV cache so the decode
        // layers can consume it via the activation graph pass-through.
        batchPrefillLayer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        if (mlxAttention != null) {
            mlxAttention.persist(batchPrefillLayer);
        }
        if (mlxProjections != null) {
            mlxProjections.persist(batchPrefillLayer);
        }

        return batchPrefillLayer;
    }

    // @formatter:on

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        // RMS: one thread per batch token
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);

        // RMS apply: B*dim threads, local=256 (dim is always a multiple of 256 for LLaMA)
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // QKV: B*(dim+2*kvDim) workgroups × LOCAL_WORK_GROUP_SIZE
        int qkvRows = dim + 2 * kvDim;
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention (flash): B*nHeads workgroups × optimalLocalSize
        int optLocal = findOptimalLocalSize(headSz);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(batchSize * nHeads * optLocal, optLocal);

        // Mat-vec (Wo, W2): B*d workgroups × LOCAL_WORK_GROUP_SIZE
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
            } else {
                scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
                scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
            }
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (mlxAttention != null) {
                mlxAttention.addGrids(scheduler, p, "batch_attention");
            } else {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
            if (mlxProjections == null) {
                scheduler.addWorkerGrid(p + "batch_attn_out", matVecDimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            if (mlxProjections == null) {
                scheduler.addWorkerGrid(p + "batch_ffn_gate_up", matVecHidWorker);
                scheduler.addWorkerGrid(p + "batch_ffn_down", matVecDimWorker);
            }
        }

        if (simdgroupGemm() && mlxProjections == null) {
            int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
            int tokenTiles = (batchSize + 63) / 64;
            int sgQkvRows = dim + 2 * kvDim;
            WorkerGrid sgQkv =
                    WorkerGridFactory.genericWorker((sgQkvRows / 64) * tokenTiles * local, local);
            WorkerGrid sgDim =
                    WorkerGridFactory.genericWorker(((dim + 63) / 64) * tokenTiles * local, local);
            WorkerGrid sgHid =
                    WorkerGridFactory.genericWorker(
                            ((hidDim + 31) / 32) * tokenTiles * local, local);
            for (int i = 0; i < config.numberOfLayers(); i++) {
                String p = GraphNames.batchPrefillLayer(i) + ".";
                scheduler.addWorkerGrid(p + "batch_qkv", sgQkv);
                scheduler.addWorkerGrid(p + "batch_attn_out", sgDim);
                scheduler.addWorkerGrid(p + "batch_ffn_gate_up", sgHid);
                scheduler.addWorkerGrid(p + "batch_ffn_down", sgDim);
            }
        }
        if (simdgroupAttention() && mlxAttention == null) {
            WorkerGrid sgAttn =
                    WorkerGridFactory.genericWorker(
                            ((batchSize
                                                    + TransformerPagedKvBatchPrefillSimdgroupKernels
                                                            .QUERY_TILE
                                                    - 1)
                                            / TransformerPagedKvBatchPrefillSimdgroupKernels
                                                    .QUERY_TILE)
                                    * config.numberOfHeads()
                                    * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                            TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS);
            for (int i = 0; i < config.numberOfLayers(); i++) {
                scheduler.addWorkerGrid(
                        GraphNames.batchPrefillLayer(i) + ".batch_attention", sgAttn);
            }
        }
    }

    private static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) {
                    optimal = s;
                    break;
                }
            }
        }
        return optimal;
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

    /**
     * The SIMD-group projections; see {@link BatchPrefillGemmPolicy#simdgroup(int, int, int, int)}.
     */
    private boolean simdgroupGemm() {
        return BatchPrefillGemmPolicy.simdgroup(
                config.dim(), config.dim(), config.kvDim(), config.hiddenDim());
    }

    /**
     * SIMD-group attention over either cache; see {@link
     * BatchPrefillGemmPolicy#simdgroupAttention(int)}.
     */
    private boolean simdgroupAttention() {
        return BatchPrefillGemmPolicy.simdgroupAttention(headSize());
    }

    private int headSize() {
        return config.headSize();
    }
}
