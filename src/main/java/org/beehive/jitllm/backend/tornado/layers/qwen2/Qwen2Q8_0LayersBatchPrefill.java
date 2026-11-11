package org.beehive.jitllm.backend.tornado.layers.qwen2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;
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
import org.beehive.jitllm.backend.tornado.layers.PrefillFamily;
import org.beehive.jitllm.backend.tornado.layers.QuantizedBatchProjections;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill;
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
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Qwen2 batched-prefill transformer-layer TaskGraphs (Llama's, with the Q/K/V biases and NeoX RoPE)
 * for the unified batched prefill-decode plan (Q8_0).
 *
 * <p>Mirrors {@link LlamaFP16LayersBatchPrefill} but uses Q8_0 kernels with inline dequantization.
 * Key differences from the FP16 path:
 *
 * <ul>
 *   <li>{@code wrapXBatch} is filled with dequantized FP32 embeddings by the host before the
 *       activation graph runs (no on-device FP16→FP32 conversion).
 *   <li>{@code wrapXbBatch} (FP32) is reused as the normalized xb intermediate: written by {@code
 *       batchedRmsApplyFP32}, read by {@code batchedFusedQKVMatmulQ8}, then overwritten by flash
 *       attention output.
 *   <li>{@code wrapXbFP16Batch} is not used.
 *   <li>Weight matrices are {@code ByteArray} (Q8_0 format).
 * </ul>
 */
public class Qwen2Q8_0LayersBatchPrefill
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

    private final Qwen2State state;
    private final Qwen2TornadoWeights weights;
    private final Qwen2Configuration config;
    private final KernelContext context;
    private final PrefillFamily family;
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

    public Qwen2Q8_0LayersBatchPrefill(
            Qwen2State state,
            Qwen2TornadoWeights weights,
            Qwen2Configuration config,
            int batchSize) {
        this.state = state;
        this.family = PrefillFamily.of(state);
        // The narrow family takes its primary's context, MLX helpers and projections.
        Qwen2Q8_0LayersBatchPrefill primary = family.primary(Qwen2Q8_0LayersBatchPrefill.class);
        this.context = primary != null ? primary.context : new KernelContext();
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
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
                                config.numberOfKeyValueHeads(),
                                config.headSize());
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
                                Map.entry(
                                        "up",
                                        state.workspace.ffnUpResult != null
                                                ? state.workspace.ffnUpResult
                                                : state.workspace.wrapHbBatch),
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
            if (splitsGateUp()) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.ffnUpResult);
            }
        } else {
            String pred = family.batchPrefillLayer(layerIndex - 1);
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
            if (splitsGateUp()) {
                layer.consumeFromDevice(pred, state.workspace.ffnUpResult);
            }
        }

        if (layerIndex == 0) {
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            layer.consumeFromDevice(
                    family.batchPrefillLayer(layerIndex - 1), state.workspace.wrapBlockTable);
        }
        if (mlxAttention != null) {
            mlxAttention.declare(layer, layerIndex, family.batchPrefillLayer(layerIndex - 1));
        }
        if (mlxProjections != null) {
            mlxProjections.declare(layer, layerIndex, family.batchPrefillLayer(layerIndex - 1));
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
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray());

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

        layer.task(
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
                    config.ropeTheta(),
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

        if (nativeLayer(layerIndex) && !q8_0Qkv(layerIndex)) {
            // One projection each: wq, wk and wv can be different representations.
            batchProjections()
                    .projection(
                            layer,
                            "batch_q",
                            state.workspace.wrapXbBatch,
                            state.workspace.wrapQBatch,
                            weights.wqLayered[layerIndex],
                            dim,
                            dim,
                            false);
            batchProjections()
                    .projection(
                            layer,
                            "batch_k",
                            state.workspace.wrapXbBatch,
                            state.workspace.wrapKBatch,
                            weights.wkLayered[layerIndex],
                            dim,
                            kvDim,
                            false);
            batchProjections()
                    .projection(
                            layer,
                            "batch_v",
                            state.workspace.wrapXbBatch,
                            state.workspace.wrapVBatch,
                            weights.wvLayered[layerIndex],
                            dim,
                            kvDim,
                            false);
        } else if (simdgroupGemm()) {

            layer.task(
                    "batch_qkv",
                    halfTilesQkv()
                            ? TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8HalfTiles
                            : TransformerBatchPrefillSimdgroupKernels::batchedGemmQKVQ8,
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

        } else {

            layer.task(
                    "batch_qkv",
                    TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    dim,
                    kvDim,
                    LOCAL_WORK_GROUP_SIZE);
        }
    }

    /** The output projection with the residual add. */
    @Override
    public void attentionOutput(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        if (nativeLayer(layerIndex)) {
            batchProjections()
                    .projection(
                            layer,
                            "batch_attn_out",
                            state.workspace.wrapXbBatch,
                            state.workspace.wrapXBatch,
                            weights.woLayered[layerIndex],
                            dim,
                            dim,
                            true);
        } else if (simdgroupGemm()) {

            layer.task(
                    "batch_attn_out",
                    halfTilesDim()
                            ? TransformerBatchPrefillSimdgroupKernels
                                    ::batchedGemmQ8WithResidualHalfTiles
                            : TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asByteArray(),
                    dim,
                    dim,
                    batchSize);

        } else {

            layer.task(
                    "batch_attn_out",
                    TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                    context,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapXBatch,
                    weights.woLayered[layerIndex].asByteArray(),
                    dim,
                    dim,
                    LOCAL_WORK_GROUP_SIZE);
        }
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

        if (nativeLayer(layerIndex)) {
            batchProjections()
                    .gateUp(
                            layer,
                            weights.w1Layered[layerIndex],
                            weights.w3Layered[layerIndex],
                            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                            dim,
                            hidDim);
        } else if (simdgroupGemm()) {

            layer.task(
                    "batch_ffn_gate_up",
                    halfTilesHid()
                            ? TransformerBatchPrefillSimdgroupKernels
                                    ::batchedGemmRmsNormFFNGateUpQ8HalfTiles
                            : TransformerBatchPrefillSimdgroupKernels
                                    ::batchedGemmRmsNormFFNGateUpQ8,
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

        } else {

            layer.task(
                    "batch_ffn_gate_up",
                    TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUpQ8,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapHbBatch,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.ffnScaleBatch,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    dim,
                    hidDim,
                    LOCAL_WORK_GROUP_SIZE);
        }
    }

    /** The down projection with the residual add. */
    @Override
    public void feedForwardDown(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        if (nativeLayer(layerIndex)) {
            batchProjections()
                    .projection(
                            layer,
                            "batch_ffn_down",
                            state.workspace.wrapHbBatch,
                            state.workspace.wrapXBatch,
                            weights.w2Layered[layerIndex],
                            hidDim,
                            dim,
                            true);
        } else if (simdgroupGemm()) {

            layer.task(
                    "batch_ffn_down",
                    halfTilesDim()
                            ? TransformerBatchPrefillSimdgroupKernels
                                    ::batchedGemmQ8WithResidualHalfTiles
                            : TransformerBatchPrefillSimdgroupKernels::batchedGemmQ8WithResidual,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex].asByteArray(),
                    hidDim,
                    dim,
                    batchSize);

        } else {

            layer.task(
                    "batch_ffn_down",
                    TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    weights.w2Layered[layerIndex].asByteArray(),
                    hidDim,
                    dim,
                    LOCAL_WORK_GROUP_SIZE);
        }
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
        int kvDim = config.kvDim();
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
        int biasRows = batchSize * (dim + 2 * kvDim);
        WorkerGrid biasWorker =
                WorkerGridFactory.genericWorker(((biasRows + 255) / 256) * 256, 256);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = family.batchPrefillLayer(i) + ".";
            scheduler.addWorkerGrid(p + "batch_qkv_bias", biasWorker);
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
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (dim + 2 * kvDim) * LOCAL_WORK_GROUP_SIZE,
                        LOCAL_WORK_GROUP_SIZE);
        WorkerGrid dimWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid hidWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        if (simdgroupGemm()) {
            int qkvLocal = TransformerBatchPrefillSimdgroupKernels.threads(halfTilesQkv());
            int dimLocal = TransformerBatchPrefillSimdgroupKernels.threads(halfTilesDim());
            int hidLocal = TransformerBatchPrefillSimdgroupKernels.threads(halfTilesHid());
            int tokenTiles = (batchSize + 63) / 64;
            qkvWorker =
                    WorkerGridFactory.genericWorker(
                            ((dim + 2 * kvDim) / 64) * tokenTiles * qkvLocal, qkvLocal);
            dimWorker =
                    WorkerGridFactory.genericWorker(
                            ((dim + 63) / 64) * tokenTiles * dimLocal, dimLocal);
            hidWorker =
                    WorkerGridFactory.genericWorker(
                            ((hidDim + 31) / 32) * tokenTiles * hidLocal, hidLocal);
        }
        scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
        scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
        if (!nativeLayer(i)) {
            scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(p + "batch_attn_out", dimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", hidWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_down", dimWorker);
            return;
        }
        QuantizedBatchProjections projections = batchProjections();
        if (q8_0Qkv(i)) {
            scheduler.addWorkerGrid(p + "batch_qkv", qkvWorker);
        } else {
            scheduler.addWorkerGrid(p + "batch_q", projections.worker(weights.wqLayered[i], dim));
            scheduler.addWorkerGrid(p + "batch_k", projections.worker(weights.wkLayered[i], kvDim));
            scheduler.addWorkerGrid(p + "batch_v", projections.worker(weights.wvLayered[i], kvDim));
        }
        scheduler.addWorkerGrid(
                p + "batch_attn_out", projections.worker(weights.woLayered[i], dim));
        TornadoTensor w1 = weights.w1Layered[i];
        TornadoTensor w3 = weights.w3Layered[i];
        if (!projections.fusedGateUp(w1, w3)) {
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate", projections.worker(w1, hidDim));
            scheduler.addWorkerGrid(p + "batch_ffn_up", projections.worker(w3, hidDim));
            scheduler.addWorkerGrid(
                    p + "batch_ffn_gate_up",
                    WorkerGridFactory.genericWorker((batchSize * hidDim + 255) / 256 * 256, 256));
        } else if (simdgroupGemm()) {
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", projections.gateUpWorker(w1, hidDim));
        } else if (w1.dataType() == DataType.Q8_0) {
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", hidWorker);
        } else {
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up", projections.worker(w1, hidDim));
        }
        scheduler.addWorkerGrid(
                p + "batch_ffn_down", projections.worker(weights.w2Layered[i], dim));
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

    /** Whether a layer holds any projection in a representation other than Q8_0. */
    private boolean nativeLayer(int layerIndex) {
        for (TornadoTensor w :
                new TornadoTensor[] {
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
                }) {
            if (w.dataType() != DataType.Q8_0) {
                return true;
            }
        }
        return false;
    }

    /** Whether a layer's wq, wk and wv are all Q8_0, so the Q8_0 QKV task serves it. */
    private boolean q8_0Qkv(int layerIndex) {
        return weights.wqLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.wkLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.wvLayered[layerIndex].dataType() == DataType.Q8_0;
    }

    /** The per-weight batched projections of a layer that is not entirely Q8_0. */
    private QuantizedBatchProjections batchProjections() {
        return new QuantizedBatchProjections(
                context, state.workspace, batchSize, simdgroupGemm(), false, LOCAL_WORK_GROUP_SIZE);
    }

    /** Whether some layer runs its gate and up apart, which needs {@code ffnUpResult}. */
    private boolean splitsGateUp() {
        if (mlxProjections != null) {
            return false;
        }
        for (int i = 0; i < config.numberOfLayers(); i++) {
            if (nativeLayer(i)
                    && !batchProjections()
                            .fusedGateUp(weights.w1Layered[i], weights.w3Layered[i])) {
                return true;
            }
        }
        return false;
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

    /**
     * Whether the Q8_0 q|k|v GEMM runs the {@code ...HalfTiles} kernel; see {@link
     * TransformerBatchPrefillSimdgroupKernels#halfTiles}.
     */
    private boolean halfTilesQkv() {
        return TransformerBatchPrefillSimdgroupKernels.halfTiles(
                (config.dim() + 2 * config.kvDim()) / 64, batchSize);
    }

    /** Whether the Q8_0 projections into the model width run the {@code ...HalfTiles} kernels. */
    private boolean halfTilesDim() {
        return TransformerBatchPrefillSimdgroupKernels.halfTiles(
                (config.dim() + 63) / 64, batchSize);
    }

    /** Whether the Q8_0 gate|up GEMM runs the {@code ...HalfTiles} kernel. */
    private boolean halfTilesHid() {
        return TransformerBatchPrefillSimdgroupKernels.halfTiles(
                (config.hiddenDim() + 31) / 32, batchSize);
    }
}
