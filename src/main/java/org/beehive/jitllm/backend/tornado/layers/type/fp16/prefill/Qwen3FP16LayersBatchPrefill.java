package org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupFP16Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillAttention;
import org.beehive.jitllm.backend.tornado.layers.MlxPrefillProjections;
import org.beehive.jitllm.backend.tornado.layers.PrefillFamily;
import org.beehive.jitllm.backend.tornado.layers.Qwen3PrefillRegions;
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
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen3State;
import org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batched-prefill transformer-layer TaskGraphs for the Qwen3 FP16 unified batched prefill-decode
 * plan.
 *
 * <p>Mirrors {@link LlamaFP16LayersBatchPrefill} but adapts to Qwen3's GQA layout and
 * Qwen3-specific kernels (fused Q/K RMSNorm, RoPE theta = 1 000 000).
 *
 * <p>Under native libraries on Metal the attention and the four projections run on MLX ({@link
 * MlxPrefill}) through the shared helpers, exactly as in {@code Qwen3Q8_0LayersBatchPrefill}: the
 * helpers' QKV GEMM writes {@code wrapQBatch}/{@code wrapKBatch}/{@code wrapVBatch}, Qwen3's Q/K
 * RMSNorm and RoPE run on them as on the JIT path, and the helpers' attention reads the rotated
 * queries and the cache RoPE wrote.
 */
public class Qwen3FP16LayersBatchPrefill
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

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context;
    private final PrefillFamily family;
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

    /** Each layer's operations before and after attention. */
    private final List<Qwen3PrefillRegions.Layer> regions;

    /** Each layer's attention, between its regions before and after attention. */
    private final List<Region> attention;

    /** Runs those operations: on MLX where it runs, otherwise on the JIT kernels. */
    private final WorkPlan plan;

    /** The workspace buffers behind the activation names the operations use. */
    private final Operands operands;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen3FP16LayersBatchPrefill(
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            int batchSize) {
        this.state = state;
        this.family = PrefillFamily.of(state);
        // The narrow family takes its primary's context, MLX helpers and projections.
        Qwen3FP16LayersBatchPrefill primary = family.primary(Qwen3FP16LayersBatchPrefill.class);
        this.context = primary != null ? primary.context : new KernelContext();
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
        Optional<MlxPrefill> mlx = MlxPrefill.resolve(state.executionPolicy(), weights, config);
        this.mlxAttention =
                primary != null
                        ? MlxPrefillAttention.narrowed(primary.mlxAttention, batchSize, family)
                        : MlxPrefillAttention.ifPresent(
                                mlx,
                                state,
                                context,
                                batchSize,
                                config.contextLength(),
                                config.numberOfHeads(),
                                nHeadKv,
                                nEmbdHead);
        this.mlxProjections =
                primary != null
                        ? MlxPrefillProjections.narrowed(primary.mlxProjections, batchSize, family)
                        : MlxPrefillProjections.ifPresent(
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
        this.regions =
                Qwen3PrefillRegions.of(
                        config,
                        batchSize,
                        batchSize,
                        (role, layer) -> weights.tensor(role, layer).dataType());
        this.attention = Qwen3PrefillRegions.attention(config, batchSize);
        List<Region> all = new ArrayList<>(Qwen3PrefillRegions.all(regions));
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
                        primary != null ? primary.plan : null,
                        all,
                        candidates,
                        Target.of(state.executionPolicy()),
                        context,
                        state.workspace.prepacking);
        this.operands =
                Operands.of(
                        Map.ofEntries(
                                Map.entry("x", state.workspace.wrapXBatch),
                                Map.entry("attnIn", state.workspace.wrapXbFP16Batch),
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

        TaskGraph layer = family.taskGraph(graphName);
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
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
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
            String pred = family.batchPrefillLayer(layerIndex - 1);
            layer.consumeFromDevice(
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
            layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }

        if (mlxAttention != null) {
            mlxAttention.declare(layer, layerIndex, family.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(layer, layerIndex, family.batchPrefillLayer(layerIndex - 1));
        }

        // Per-layer weights: upload once
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

        // ── Attention Block ────────────────────────────────────────────────────
        // RMS norm, then q, k and v.
        plan.emit(
                layer,
                regions.get(layerIndex).beforeAttention(),
                operands,
                state.workspace.weightBindings);

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

        // Reuses batchedFlashAttention: passes qDim as the 'dim' stride parameter.
        // Valid because qDim == dim for all standard Qwen3 models (nEmbdHeadK = dim/nHeads).
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

    /** RMS norm into wrapXbFP16Batch, then q, k and v. */
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

        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        layer.task(
                "batch_qkv",
                tiledQkv()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVFP16
                                : TransformerBatchPrefillKernels::batchedGemmQKVFP16)
                        : Qwen3Kernels::batchedFusedQKVMatmulFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                dim,
                qDim,
                kvDim,
                tiledQkv() ? batchSize : LOCAL_WORK_GROUP_SIZE);
    }

    /** The output projection with the residual add in the same kernel. */
    @Override
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        // Output projection: n=qDim (input), d=dim (output)
        layer.task(
                "batch_attn_out",
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmFP16WithResidual
                                : TransformerBatchPrefillKernels::batchedGemmFP16WithResidual)
                        : TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                qDim,
                dim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);
    }

    /** RMS norm, gate|up and SwiGLU in one kernel. */
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
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmRmsNormFFNGateUpFP16
                                : TransformerBatchPrefillKernels::batchedGemmRmsNormFFNGateUpFP16)
                        : TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUp,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                dim,
                hidDim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);
    }

    /** The down projection with the residual add in the same kernel. */
    @Override
    public void feedForwardDown(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        layer.task(
                "batch_ffn_down",
                tiledGemm()
                        ? (simdgroupGemm()
                                ? TransformerBatchPrefillSimdgroupKernels
                                        ::batchedGemmFP16WithResidual
                                : TransformerBatchPrefillKernels::batchedGemmFP16WithResidual)
                        : TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context,
                state.workspace.wrapHbBatch,
                state.workspace.wrapXBatch,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                hidDim,
                dim,
                tiledGemm() ? batchSize : LOCAL_WORK_GROUP_SIZE);
    }

    @Override
    public void attention(TaskGraph layer, int layerIndex) {
        if (useFp16KVCache()) {
            layer.task(
                    "batch_attention",
                    simdgroupAttention()
                            ? (nEmbdHead == 64
                                    ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                            ::batchedFlashAttentionKVFP16PagedHead64
                                    : nEmbdHead == 96
                                            ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead96Rows8
                                            : TransformerPagedKvBatchPrefillSimdgroupFP16Kernels
                                                    ::batchedFlashAttentionKVFP16PagedHead128Rows8)
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
    }

    public void updateGridScheduler(GridScheduler scheduler) {
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
        int attnTiles = attentionTiles();
        WorkerGrid attnWorker =
                simdgroupAttention()
                        ? WorkerGridFactory.genericWorker(
                                attnTiles
                                        * config.numberOfHeads()
                                        * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                                TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS)
                        : WorkerGridFactory.genericWorker(
                                batchSize * config.numberOfHeads() * optLocal, optLocal);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = family.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm", qkRmsNormWorker);
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
        int hidDim = config.hiddenDim();
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (qDim + 2 * kvDim) * LOCAL_WORK_GROUP_SIZE,
                        LOCAL_WORK_GROUP_SIZE);
        // Wo: B*dim output rows (n=qDim, d=dim)
        WorkerGrid matVecDimWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
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
        WorkerGrid matVecHidWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(p + "batch_qkv", tiledQkv() ? gemmQkvWorker : qkvWorker);
        scheduler.addWorkerGrid(
                p + "batch_attn_out", tiledGemm() ? gemmDimWorker : matVecDimWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
        scheduler.addWorkerGrid(
                p + "batch_ffn_gate_up", tiledGemm() ? gemmHidWorker : matVecHidWorker);
        scheduler.addWorkerGrid(
                p + "batch_ffn_down", tiledGemm() ? gemmDimWorker : matVecDimWorker);
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

    /** The tiled QKV kernel needs every 64-row tile to lie in one of Q, K and V. */
    private boolean tiledQkv() {
        return tiledGemm() && qDim % 64 == 0 && kvDim % 64 == 0;
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

    /**
     * The attention workgroups along the batch: 8 queries each over the FP16 cache for heads wider
     * than 64 ({@code ...Rows8} kernels), 32 otherwise.
     */
    private int attentionTiles() {
        int tile =
                useFp16KVCache() && nEmbdHead != 64
                        ? TransformerPagedKvBatchPrefillSimdgroupFP16Kernels.ROWS8_QUERY_TILE
                        : TransformerPagedKvBatchPrefillSimdgroupKernels.QUERY_TILE;
        return (batchSize + tile - 1) / tile;
    }
}
