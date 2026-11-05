package org.beehive.jitllm.backend.tornado.layers.llama;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.BatchPrefillSupport;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.LlamaPrefillRegions;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.ops.Implementation;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.backend.tornado.ops.WorkPlan;
import org.beehive.jitllm.backend.tornado.ops.jit.LayerKernels;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxAttention;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxMatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxPrefill;
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
 * Llama batched-prefill transformer-layer TaskGraphs for retained Q4_0 weights, on the Metal
 * SIMD-group kernels only ({@link BatchPrefillSupport} refuses it elsewhere): the projections
 * dequantize each Q4_0 block to {@code float} as they stage it. The host decodes the embeddings
 * into {@code wrapXBatch}, as for Q8_0.
 */
public class LlamaQ4_0LayersBatchPrefill
        implements BatchPrefillTransformerLayerTaskGraphs, LayerKernels.Tasks {

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

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final MlxPrefillAttention mlxAttention;
    private final MlxPrefillProjections mlxProjections;

    /** Each layer's operations before and after attention. */
    private final List<LlamaPrefillRegions.Whole> regions;

    /** Each layer's attention, between its regions before and after attention. */
    private final List<Region> attention;

    /** Runs those operations: on MLX where it runs, otherwise on the JIT kernels. */
    private final WorkPlan plan;

    /** The workspace buffers behind the activation names the operations use. */
    private final Operands operands;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaQ4_0LayersBatchPrefill(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        Optional<MlxPrefill> mlx = MlxPrefill.resolve(state.executionPolicy(), weights, config);
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
                        config.hiddenDim());
        this.regions =
                LlamaPrefillRegions.whole(
                        config,
                        batchSize,
                        batchSize,
                        false,
                        (role, layer) -> weights.tensor(role, layer).dataType());
        this.attention = LlamaPrefillRegions.attention(config, batchSize, false);
        List<Region> all = new ArrayList<>(LlamaPrefillRegions.allWhole(regions));
        all.addAll(attention);
        List<Implementation> candidates =
                mlxProjections != null
                        ? List.of(
                                new MlxMatMul(mlxProjections),
                                new MlxAttention(mlxAttention),
                                new LayerKernels(this))
                        : List.of(new LayerKernels(this));
        this.plan =
                WorkPlan.of(
                        all,
                        candidates,
                        Target.of(state.executionPolicy()),
                        context,
                        state.workspace.prepacking);
        this.operands =
                Operands.of(
                        Map.ofEntries(
                                Map.entry("x", state.workspace.wrapXBatch),
                                Map.entry("attnIn", state.workspace.wrapXbBatch),
                                Map.entry("attnIn.scale", state.workspace.attnScaleBatch),
                                Map.entry("q", state.workspace.wrapQBatch),
                                Map.entry("k", state.workspace.wrapKBatch),
                                Map.entry("v", state.workspace.wrapVBatch),
                                Map.entry("attn", state.workspace.wrapXbBatch),
                                Map.entry("o", state.workspace.wrapXBatch),
                                Map.entry("ffnIn", state.workspace.wrapXbBatch),
                                Map.entry("ffnIn.scale", state.workspace.ffnScaleBatch),
                                Map.entry("gate", state.workspace.wrapHbBatch),
                                Map.entry("up", state.workspace.wrapHbBatch),
                                Map.entry("hidden", state.workspace.wrapHbBatch),
                                Map.entry("down", state.workspace.wrapXBatch)),
                        weights::tensor,
                        all);
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

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            // Allocate GPU-side batch intermediates once.
            // wrapXBatch is filled with dequantized FP32 by the host, persisted by
            // prefillActivation.
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
        }
        if (mlxAttention != null) {
            mlxAttention.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(layer, layerIndex, GraphNames.batchPrefillLayer(layerIndex - 1));
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            layer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex - 1), state.workspace.wrapBlockTable);
        }

        // Per-layer weights: upload once (Q8_0 format)
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].deviceArray(),
                weights.wkLayered[layerIndex].deviceArray(),
                weights.wvLayered[layerIndex].deviceArray(),
                weights.woLayered[layerIndex].deviceArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].deviceArray(),
                weights.w2Layered[layerIndex].deviceArray(),
                weights.w3Layered[layerIndex].deviceArray(),
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        // RMS norm, then q, k and v.
        plan.emit(
                layer,
                regions.get(layerIndex).beforeAttention(),
                operands,
                state.workspace.weightBindings);

        if (useFp16KVCache()) {
            layer.task(
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
            layer.task(
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

        // Overwrites wrapXbBatch with attention output
        plan.emit(layer, attention.get(layerIndex), operands, state.workspace.weightBindings);

        // Output projection, residual add, RMS norm, gate|up, SwiGLU, down, residual add.
        plan.emit(
                layer,
                regions.get(layerIndex).afterAttention(),
                operands,
                state.workspace.weightBindings);

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

    /** RMS norm, then q, k and v. */
    @Override
    public void attentionInput(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

        // Writes FP32 normalized xb into wrapXbBatch (reused later by flash attention)
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
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ4_0,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim,
                dim,
                kvDim,
                batchSize);
    }

    /** The output projection with the residual add. */
    @Override
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        layer.task(
                "batch_attn_out",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0WithResidual,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim,
                dim,
                batchSize);
    }

    /** RMS norm, gate|up and SwiGLU. */
    @Override
    public void feedForwardUp(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        layer.task(
                "batch_ffn_gate_up",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmRmsNormFFNGateUpQ4_0,
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
    }

    /** The down projection with the residual add. */
    @Override
    public void feedForwardDown(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        layer.task(
                "batch_ffn_down",
                TransformerBatchPrefillSimdgroupKernels::batchedGemmQ4_0WithResidual,
                context,
                state.workspace.wrapHbBatch,
                state.workspace.wrapXBatch,
                weights.w2Layered[layerIndex].asByteArray(),
                hidDim,
                dim,
                batchSize);
    }

    @Override
    public void attention(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        if (useFp16KVCache()) {
            if (simdgroupAttention()) {
                if (headSize() == 64) {
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
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else if (headSize() == 96) {
                    layer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                    ::batchedFlashAttentionKVFP16PagedHead96Rows8,
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
                    layer.task(
                            "batch_attention",
                            TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                    ::batchedFlashAttentionKVFP16PagedHead128Rows8,
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
                layer.task(
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
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
                } else if (headSize() == 96) {
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
                            config.headSize(),
                            kvDim,
                            config.kvMul(),
                            layerIndex,
                            state.workspace.wrapBlockTable,
                            state.kvBlockCfg,
                            state.kvBlockStride,
                            dim);
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
                layer.task(
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
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int optLocal = WorkerGridFactory.findOptimalLocalSize(headSz);
        WorkerGrid attnWorker =
                simdgroupAttention()
                        ? WorkerGridFactory.genericWorker(
                                attentionTiles()
                                        * nHeads
                                        * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                                TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS)
                        : WorkerGridFactory.genericWorker(batchSize * nHeads * optLocal, optLocal);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = GraphNames.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (!plan.runs(attention.get(i), "MLX")) {
                scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            }
        }
        plan.addGrids(scheduler);
    }

    @Override
    public void addGrids(GridScheduler scheduler, String p, int i) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
        int tokenTiles = (batchSize + 63) / 64;
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid sgQkv =
                WorkerGridFactory.genericWorker(
                        ((dim + 2 * kvDim) / 64) * tokenTiles * local, local);
        WorkerGrid sgDim =
                WorkerGridFactory.genericWorker(((dim + 63) / 64) * tokenTiles * local, local);
        WorkerGrid sgHid =
                WorkerGridFactory.genericWorker(((hidDim + 31) / 32) * tokenTiles * local, local);
        scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(p + "batch_qkv", sgQkv);
        scheduler.addWorkerGrid(p + "batch_attn_out", sgDim);
        scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_gate_up", sgHid);
        scheduler.addWorkerGrid(p + "batch_ffn_down", sgDim);
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

    /**
     * The attention workgroups along the batch: 8 queries each over the FP16 cache for heads wider
     * than 64 ({@code ...Rows8} kernels), 32 otherwise.
     */
    private int attentionTiles() {
        int tile =
                useFp16KVCache() && headSize() != 64
                        ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels.ROWS8_QUERY_TILE
                        : TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE;
        return (batchSize + tile - 1) / tile;
    }
}
