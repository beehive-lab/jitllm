package org.beehive.jitllm.backend.tornado.layers.phi3;

import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.BatchPrefillSupport;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * Phi-3 batched-prefill transformer-layer TaskGraphs (Phi3Q8_0Layers weights), on the Metal
 * SIMD-group kernels only ({@link BatchPrefillSupport} refuses the family elsewhere). The fused
 * {@code wqkv} and {@code wUp} weights are read by the fused projection kernels; RoPE rotates
 * split-half pairs with Phi-3's base of 10000; attention runs the kernel for the model's head
 * width.
 */
// @formatter:on
public class Phi3Q8_0LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

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

    /** Phi-3's RoPE base, fixed in its single-token kernel too. */
    private static final float ROPE_THETA = 10000.0f;

    private final Phi3State state;
    private final Phi3TornadoWeights weights;
    private final Phi3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final MlxPrefillAttention mlxAttention;
    private final MlxPrefillProjections mlxProjections;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Phi3Q8_0LayersBatchPrefill(
            Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config, int batchSize) {
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
                        config.dim(),
                        config.kvDim(),
                        config.hiddenDim(),
                        1.0f,
                        i -> Phi3MlxLayerWeights.of(weights, i));
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = GraphNames.batchPrefillLayer(layerIndex);
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyCache(),
                    valueCache());
            layer.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
            // EVERY_EXECUTION: a lease change rewrites the table, and a stale index is still valid.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
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
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        if (mlxAttention != null) {
            mlxAttention.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asByteArray(),
                weights.wDownLayered[layerIndex].asByteArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

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
            layer.task(
                    "batch_attn_rms_apply",
                    TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.attnScaleBatch,
                    dim);
            layer.task(
                    "batch_qkv",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFusedQ8,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    weights.wqkvLayered[layerIndex].asByteArray(),
                    dim,
                    dim,
                    kvDim,
                    batchSize);
        }
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
                    ROPE_THETA,
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
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
                    ROPE_THETA,
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        }
        if (mlxAttention != null) {
            mlxAttention.attention(layer, "batch_attention", layerIndex);
        } else {
            addAttention(layer, layerIndex);
        }
        if (mlxProjections != null) {
            mlxProjections.attentionOutput(layer, layerIndex);
        } else {
            layer.task(
                    "batch_attn_out",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asByteArray(),
                    dim,
                    dim,
                    batchSize);
        }
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
        } else {
            layer.task(
                    "batch_ffn_gate_up",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpFusedQ8,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.wUpLayered[layerIndex].asByteArray(),
                    dim,
                    hidDim,
                    batchSize);
            layer.task(
                    "batch_ffn_down",
                    TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.wDownLayered[layerIndex].asByteArray(),
                    hidDim,
                    dim,
                    batchSize);
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

    private void addAttention(TaskGraph layer, int layerIndex) {
        if (useFp16KVCache()) {
            addAttentionFP16(layer, layerIndex);
            return;
        }
        int h = config.headSize();
        if (h == 64) {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        } else if (h == 96) {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        } else {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        }
    }

    /** {@link #addAttention} over the FP16 cache. */
    private void addAttentionFP16(TaskGraph layer, int layerIndex) {
        int h = config.headSize();
        if (h == 64) {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        } else if (h == 96) {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        } else {
            layer.task(
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
                    h,
                    config.kvDim(),
                    config.kvMul(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.dim());
        }
    }

    /** Registers the batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
        int tokenTiles = (batchSize + 63) / 64;
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        ((dim + 2 * kvDim) / 64) * tokenTiles * local, local);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        WorkerGrid attnWorker =
                WorkerGridFactory.genericWorker(
                        ((batchSize + TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE - 1)
                                        / TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE)
                                * config.numberOfHeads()
                                * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                        TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS);
        WorkerGrid dimWorker =
                WorkerGridFactory.genericWorker(((dim + 63) / 64) * tokenTiles * local, local);
        WorkerGrid hidWorker =
                WorkerGridFactory.genericWorker(((hidDim + 31) / 32) * tokenTiles * local, local);
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
                scheduler.addWorkerGrid(p + "batch_attn_out", dimWorker);
            }
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            if (mlxProjections == null) {
                scheduler.addWorkerGrid(p + "batch_ffn_gate_up", hidWorker);
                scheduler.addWorkerGrid(p + "batch_ffn_down", dimWorker);
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
}
