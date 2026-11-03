package org.beehive.jitllm.backend.tornado.layers.qwen2;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
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
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxMatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.apple.MlxPrefill;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Qwen2 batched-prefill transformer-layer TaskGraphs (Llama's, with the Q/K/V biases and NeoX RoPE)
 * for the unified batched prefill-decode plan ({@link TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing {@code batchSize} tokens
 * simultaneously via {@link TransformerBatchPrefillKernels}.
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device after every
 * layer so the subsequent single-token decode layers can consume it.
 */
public class Qwen2FP16LayersBatchPrefill
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

    // Matches the local workgroup size used by the single-token kernels.
    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final Qwen2State state;
    private final Qwen2TornadoWeights weights;
    private final Qwen2Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final MlxPrefillAttention mlxAttention;
    private final MlxPrefillProjections mlxProjections;

    /** Each layer's operations before and after attention. */
    private final List<LlamaPrefillRegions.Whole> regions;

    /** Runs those operations: on MLX where it runs, otherwise on the JIT kernels. */
    private final WorkPlan plan;

    /** The workspace buffers behind the activation names the operations use. */
    private final Operands operands;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen2FP16LayersBatchPrefill(
            Qwen2State state,
            Qwen2TornadoWeights weights,
            Qwen2Configuration config,
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
        List<Region> all = LlamaPrefillRegions.allWhole(regions);
        List<Implementation> candidates =
                mlxProjections != null
                        ? List.of(new MlxMatMul(mlxProjections), new LayerKernels(this))
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
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        // RMS norm, then q, k and v.
        plan.emit(
                batchPrefillLayer,
                regions.get(layerIndex).beforeAttention(),
                operands,
                state.workspace.weightBindings);

        batchPrefillLayer.task(
                "batch_qkv_bias",
                Qwen2BatchPrefillKernels::batchedQKVBias,
                context,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                dim,
                kvDim,
                batchSize);

        if (useFp16KVCache()) {
            batchPrefillLayer.task(
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
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        } else {
            batchPrefillLayer.task(
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
                    batchPrefillLayer.task(
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

        // Output projection, residual add, RMS norm, gate|up, SwiGLU, down, residual add.
        plan.emit(
                batchPrefillLayer,
                regions.get(layerIndex).afterAttention(),
                operands,
                state.workspace.weightBindings);

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

    /** RMS norm, then q, k and v. */
    @Override
    public void attentionInput(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        batchPrefillLayer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps());

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

    /** The output projection with the residual add. */
    @Override
    public void attentionOutput(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        if (simdgroupGemm()) {

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
    }

    /** RMS norm, gate|up and SwiGLU. */
    @Override
    public void feedForwardUp(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps());

        if (simdgroupGemm()) {

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
    }

    /** The down projection with the residual add. */
    @Override
    public void feedForwardDown(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        if (simdgroupGemm()) {

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

        } else {

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
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();
        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        // Attention: SIMD-group tiles, or B*nHeads workgroups × optimalLocalSize
        int optLocal = WorkerGridFactory.findOptimalLocalSize(headSz);
        WorkerGrid attnWorker =
                simdgroupAttention()
                        ? WorkerGridFactory.genericWorker(
                                attentionTiles()
                                        * nHeads
                                        * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS,
                                TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS)
                        : WorkerGridFactory.genericWorker(batchSize * nHeads * optLocal, optLocal);
        int biasRows = batchSize * (dim + 2 * kvDim);
        WorkerGrid biasWorker =
                WorkerGridFactory.genericWorker(((biasRows + 255) / 256) * 256, 256);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = GraphNames.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_qkv_bias", biasWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            if (mlxAttention != null) {
                mlxAttention.addGrids(scheduler, p, "batch_attention");
            } else {
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
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker;
        WorkerGrid dimWorker;
        WorkerGrid hidWorker;
        if (simdgroupGemm()) {
            int local = TransformerBatchPrefillSimdgroupKernels.THREADS;
            int tokenTiles = (batchSize + 63) / 64;
            qkvWorker =
                    WorkerGridFactory.genericWorker(
                            ((dim + 2 * kvDim) / 64) * tokenTiles * local, local);
            dimWorker =
                    WorkerGridFactory.genericWorker(((dim + 63) / 64) * tokenTiles * local, local);
            hidWorker =
                    WorkerGridFactory.genericWorker(
                            ((hidDim + 31) / 32) * tokenTiles * local, local);
        } else {
            qkvWorker =
                    WorkerGridFactory.genericWorker(
                            batchSize * (dim + 2 * kvDim) * LOCAL_WORK_GROUP_SIZE,
                            LOCAL_WORK_GROUP_SIZE);
            dimWorker =
                    WorkerGridFactory.genericWorker(
                            batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
            hidWorker =
                    WorkerGridFactory.genericWorker(
                            batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        }
        scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
        scheduler.addWorkerGrid(p + "batch_attn_out", dimWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_gate_up", hidWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_down", dimWorker);
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
