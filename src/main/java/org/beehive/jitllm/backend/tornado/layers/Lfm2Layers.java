package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.kernels.Lfm2Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35AttentionKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.plan.FusedOperandSupport;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Lfm2State;
import org.beehive.jitllm.inference.weights.tornado.Lfm2TornadoWeights;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The {@code lfm2moe} transformer layers, one task graph per layer, for single-token decode.
 *
 * <h2>Two independent choices per layer</h2>
 *
 * <p>The mixer is attention or the gated short convolution ({@link
 * Lfm2Configuration#isAttentionLayer(int)}), and the feed-forward is dense or a routed mixture of
 * experts ({@link Lfm2Configuration#isMoeLayer(int)}) — the same answers the host path uses. Both
 * mixers and both feed-forwards share one outer residual shape.
 *
 * <h2>The mixture of experts runs entirely on the device</h2>
 *
 * <p>The router's projection, the sigmoid top-k selection and the selected experts' projections are
 * consecutive tasks of one graph. The selection is written to a device buffer and the expert
 * kernels read their expert ids from it, so a token's routing never returns to the host, and only
 * the selected experts' rows are read from the stacked expert tensors.
 *
 * <h2>Every task is selected by the representation of the tensor it reads</h2>
 *
 * <p>The 24B mixes Q4_K, Q5_K and Q6_K experts with Q8_0 attention and dense layers. Nothing is
 * materialized; a representation with no kernel for a task is refused by name at construction.
 */
// @formatter:on
public class Lfm2Layers
        extends AbstractTransformerLayerTaskGraphs<Lfm2TornadoWeights, Lfm2Configuration> {

    /** Lanes per workgroup for a matrix-vector task: one workgroup reduces one output row. */
    private static final int MATVEC_LOCAL = 128;

    /** Lanes per workgroup for an elementwise task. */
    private static final int ELEMENTWISE_LOCAL = 128;

    private final Lfm2State lfm2State;

    /** The graph layer 0 consumes its activation from. */
    private final String activationGraphName;

    public Lfm2Layers(
            String taskGraphName,
            Lfm2State state,
            Lfm2TornadoWeights weights,
            Lfm2Configuration config,
            SchedulerType schedulerType) {
        this(taskGraphName, state, weights, config, schedulerType, "activationUpdate");
    }

    public Lfm2Layers(
            String taskGraphName,
            Lfm2State state,
            Lfm2TornadoWeights weights,
            Lfm2Configuration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.lfm2State = state;
        this.activationGraphName = activationGraphName;
        setupFFNLayers();
    }

    // ── validation ────────────────────────────────────────────────────────────

    private TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "lfm2moe layer "
                            + layer
                            + " must carry "
                            + role
                            + ", which this model does not hold for it");
        }
        return tensor;
    }

    private static UnsupportedOperationException unsupported(
            int layer, String task, String role, DataType type) {
        return new UnsupportedOperationException(
                "lfm2moe layer "
                        + layer
                        + " task '"
                        + task
                        + "' reads "
                        + role
                        + " as "
                        + type
                        + ", for which this backend has no kernel. It is not converted to another"
                        + " representation to get one.");
    }

    private static void requireWholeBlocks(int layer, String role, DataType type, int n) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1, Q8_0 -> 32;
                    case Q4_K, Q5_K, Q6_K -> 256;
                    default -> 1;
                };
        if (n % blockSize != 0) {
            throw new UnsupportedOperationException(
                    "lfm2moe layer "
                            + layer
                            + " reads "
                            + role
                            + " as "
                            + type
                            + " with rows of "
                            + n
                            + " weights, not a whole number of "
                            + blockSize
                            + "-weight blocks");
        }
    }

    // ── per-representation matrix-vector dispatch ─────────────────────────────

    /** {@code out = w · x}, or {@code out += w · x}, by the representation {@code w} is in. */
    private void matVec(
            TaskGraph graph,
            int layer,
            String task,
            String role,
            TornadoTensor w,
            FloatArray x,
            FloatArray out,
            int n,
            int d,
            boolean residual) {
        requireWholeBlocks(layer, role, w.dataType(), n);
        switch (w.dataType()) {
            case F32 -> {
                if (residual) {
                    throw unsupported(layer, task, role, w.dataType());
                }
                graph.task(
                        task,
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        x,
                        out,
                        w.asFloatArray(),
                        n,
                        d,
                        MATVEC_LOCAL);
            }
            case Q8_0 -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered
                                    ::matrixVectorGenericWithResidualQ8_0Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q5_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q6_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            default -> throw unsupported(layer, task, role, w.dataType());
        }
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph("layer_" + layerIndex);
        String predecessor = layerIndex == 0 ? activationGraphName : "layer_" + (layerIndex - 1);
        layer.consumeFromDevice(predecessor, lfm2State.workspace.wrapX);
        configureLayerDataTransfers(layer, layerIndex);
        transferLayerWeights(layer, layerIndex);

        normalize(
                layer,
                "attn_rms_reduce",
                "attn_rms_finalize",
                "attn_rms_apply",
                lfm2State.workspace.temp,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));

        if (config.isAttentionLayer(layerIndex)) {
            attentionBranch(layer, layerIndex);
        } else {
            shortConvBranch(layer, layerIndex);
        }

        normalize(
                layer,
                "ffn_rms_reduce",
                "ffn_rms_finalize",
                "ffn_rms_apply",
                lfm2State.workspace.tempFFN,
                require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm"));

        if (config.isMoeLayer(layerIndex)) {
            mixtureOfExperts(layer, layerIndex);
        } else {
            denseFeedForward(layer, layerIndex);
        }

        layer.persistOnDevice(
                lfm2State.workspace.wrapX,
                keyStore(),
                valueStore(),
                lfm2State.workspace.wrapConvState);
        return layer;
    }

    /** {@code xb = weight ⊙ rms(x)}: the reduction, its finalize where needed, and the apply. */
    private void normalize(
            TaskGraph layer,
            String reduce,
            String finalize,
            String apply,
            FloatArray scratch,
            TornadoTensor weight) {
        layer.task(
                reduce,
                rmsReduceKernel(),
                context,
                scratch,
                lfm2State.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                lfm2State.localSize);
        if (shouldUseFinalNormalization()) {
            layer.task(
                    finalize,
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    scratch,
                    config.dim(),
                    config.rmsNormEps());
        }
        layer.task(
                apply,
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                lfm2State.workspace.wrapXb,
                lfm2State.workspace.wrapX,
                weight.asFloatArray(),
                scratch);
    }

    /** Whether this state's key/value store is half precision. */
    private boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    private Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    /**
     * Grouped-query attention with per-head query/key RMS norms and NEOX RoPE over the whole head,
     * from {@code wrapXb} back into {@code wrapX}.
     */
    private void attentionBranch(TaskGraph layer, int layerIndex) {
        final int headDim = config.headSize();
        final int kvDim = config.kvDim();
        var ws = lfm2State.workspace;

        matVec(
                layer,
                layerIndex,
                "attn_q_proj",
                "attn_q",
                require(weights.wqLayered, layerIndex, "attn_q"),
                ws.wrapXb,
                ws.wrapQ,
                config.dim(),
                config.dim(),
                false);
        matVec(
                layer,
                layerIndex,
                "attn_k_proj",
                "attn_k",
                require(weights.wkLayered, layerIndex, "attn_k"),
                ws.wrapXb,
                ws.wrapK,
                config.dim(),
                kvDim,
                false);
        matVec(
                layer,
                layerIndex,
                "attn_v_proj",
                "attn_v",
                require(weights.wvLayered, layerIndex, "attn_v"),
                ws.wrapXb,
                ws.wrapV,
                config.dim(),
                kvDim,
                false);

        layer.task(
                "attn_qk_norm",
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                ws.wrapQ,
                ws.wrapK,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim,
                config.rmsNormEps());

        // NEOX rotation over the whole head: the partial-rotation kernel with the rotary width
        // equal to the head's.
        layer.task(
                "attn_rope",
                Qwen35AttentionKernels::ropeNeoxPartial,
                context,
                ws.positionHolder,
                ws.wrapQ,
                ws.wrapK,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim);

        // The store is sized by the layers that attend; this layer addresses it by its dense
        // index.
        int kvLayer = config.keyValueLayerIndex(layerIndex);
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35AttentionKernels::appendKeyValueFP16Paged,
                    context,
                    ws.positionHolder,
                    ws.wrapK,
                    ws.wrapV,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    lfm2State.kvBlockCfg,
                    lfm2State.kvBlockStride);
            layer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    ws.wrapQ,
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    ws.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    ws.positionHolder,
                    kvLayer,
                    ws.wrapBlockTable,
                    lfm2State.kvBlockCfg,
                    lfm2State.kvBlockStride);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35AttentionKernels::appendKeyValuePaged,
                    context,
                    ws.positionHolder,
                    ws.wrapK,
                    ws.wrapV,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    lfm2State.kvBlockCfg,
                    lfm2State.kvBlockStride);
            layer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    ws.wrapQ,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    ws.positionHolder,
                    kvLayer,
                    ws.wrapBlockTable,
                    lfm2State.kvBlockCfg,
                    lfm2State.kvBlockStride);
        }

        matVec(
                layer,
                layerIndex,
                "attn_output_proj",
                "attn_output",
                require(weights.woLayered, layerIndex, "attn_output"),
                ws.wrapXb,
                ws.wrapX,
                config.dim(),
                config.dim(),
                true);
    }

    /**
     * The gated short convolution, from {@code wrapXb} back into {@code wrapX}: {@code in_proj},
     * the convolution with both gates, and {@code out_proj} folding the residual.
     */
    private void shortConvBranch(TaskGraph layer, int layerIndex) {
        var ws = lfm2State.workspace;
        matVec(
                layer,
                layerIndex,
                "conv_in_proj",
                "shortconv.in_proj",
                require(weights.convInProj, layerIndex, "shortconv.in_proj"),
                ws.wrapXb,
                ws.wrapSsmQkv,
                config.dim(),
                config.shortConvProjectionDim(),
                false);
        layer.task(
                "conv",
                Lfm2Kernels::shortConv,
                context,
                ws.wrapSsmQkv,
                require(weights.convKernel, layerIndex, "shortconv.conv").asFloatArray(),
                ws.wrapConvState,
                ws.wrapSsmConvOut,
                config.dim(),
                config.shortConvCache(),
                config.convLayerIndex(layerIndex) * config.convStateSize());
        matVec(
                layer,
                layerIndex,
                "conv_out_proj",
                "shortconv.out_proj",
                require(weights.convOutProj, layerIndex, "shortconv.out_proj"),
                ws.wrapSsmConvOut,
                ws.wrapX,
                config.dim(),
                config.dim(),
                true);
    }

    /** The dense SwiGLU of the leading layers, from {@code wrapXb} back into {@code wrapX}. */
    private void denseFeedForward(TaskGraph layer, int layerIndex) {
        var ws = lfm2State.workspace;
        TornadoTensor gate = require(weights.w1Layered, layerIndex, "ffn_gate");
        TornadoTensor up = require(weights.w3Layered, layerIndex, "ffn_up");
        FusedOperandSupport.requireUniform(
                "lfm2moe layer " + layerIndex + " fused gate/up feed-forward",
                List.of("ffn_gate", "ffn_up"),
                gate,
                up);
        switch (gate.dataType()) {
            case Q8_0 ->
                    layer.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsLayered
                                    ::fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case Q4_K ->
                    layer.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_K,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            default ->
                    throw unsupported(
                            layerIndex, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType());
        }
        matVec(
                layer,
                layerIndex,
                "ffn_down_proj",
                "ffn_down",
                require(weights.w2Layered, layerIndex, "ffn_down"),
                ws.wrapHb,
                ws.wrapX,
                config.hiddenDim(),
                config.dim(),
                true);
    }

    /**
     * The routed feed-forward: router projection, sigmoid top-k selection, the selected experts'
     * gate/up with SwiGLU, and their down projections weighted into {@code wrapX}.
     */
    private void mixtureOfExperts(TaskGraph layer, int layerIndex) {
        var ws = lfm2State.workspace;
        final int dim = config.dim();
        final int expertHidden = config.expertHiddenDim();
        final int used = config.expertUsedCount();

        matVec(
                layer,
                layerIndex,
                "moe_router",
                "ffn_gate_inp",
                require(weights.ffnGateInp, layerIndex, "ffn_gate_inp"),
                ws.wrapXb,
                ws.wrapRouterLogits,
                dim,
                config.expertCount(),
                false);
        layer.task(
                "moe_select",
                Lfm2Kernels::selectExpertsSigmoid,
                context,
                ws.wrapRouterLogits,
                require(weights.expProbsBias, layerIndex, "exp_probs_b").asFloatArray(),
                ws.wrapSelectedExperts,
                ws.wrapSelectedExpertWeights,
                config.expertCount(),
                used);

        TornadoTensor gate = require(weights.ffnGateExps, layerIndex, "ffn_gate_exps");
        TornadoTensor up = require(weights.ffnUpExps, layerIndex, "ffn_up_exps");
        TornadoTensor down = require(weights.ffnDownExps, layerIndex, "ffn_down_exps");
        FusedOperandSupport.requireUniform(
                "lfm2moe layer " + layerIndex + " expert gate/up",
                List.of("ffn_gate_exps", "ffn_up_exps"),
                gate,
                up);
        requireWholeBlocks(layerIndex, "ffn_gate_exps", gate.dataType(), dim);
        requireWholeBlocks(layerIndex, "ffn_down_exps", down.dataType(), expertHidden);

        switch (gate.dataType()) {
            case Q4_K ->
                    layer.task(
                            "moe_gate_up",
                            Lfm2Kernels::expertGateUpSiLUQ4_K,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            ws.wrapSelectedExperts,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q5_K ->
                    layer.task(
                            "moe_gate_up",
                            Lfm2Kernels::expertGateUpSiLUQ5_K,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            ws.wrapSelectedExperts,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q6_K ->
                    layer.task(
                            "moe_gate_up",
                            Lfm2Kernels::expertGateUpSiLUQ6_K,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            ws.wrapSelectedExperts,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q8_0 ->
                    layer.task(
                            "moe_gate_up",
                            Lfm2Kernels::expertGateUpSiLUQ8_0,
                            context,
                            ws.wrapXb,
                            ws.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            ws.wrapSelectedExperts,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            default ->
                    throw unsupported(layerIndex, "moe_gate_up", "ffn_gate_exps", gate.dataType());
        }
        switch (down.dataType()) {
            case Q4_K ->
                    layer.task(
                            "moe_down",
                            Lfm2Kernels::expertDownResidualQ4_K,
                            context,
                            ws.wrapHb,
                            ws.wrapX,
                            down.asByteArray(),
                            ws.wrapSelectedExperts,
                            ws.wrapSelectedExpertWeights,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q5_K ->
                    layer.task(
                            "moe_down",
                            Lfm2Kernels::expertDownResidualQ5_K,
                            context,
                            ws.wrapHb,
                            ws.wrapX,
                            down.asByteArray(),
                            ws.wrapSelectedExperts,
                            ws.wrapSelectedExpertWeights,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q6_K ->
                    layer.task(
                            "moe_down",
                            Lfm2Kernels::expertDownResidualQ6_K,
                            context,
                            ws.wrapHb,
                            ws.wrapX,
                            down.asByteArray(),
                            ws.wrapSelectedExperts,
                            ws.wrapSelectedExpertWeights,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            case Q8_0 ->
                    layer.task(
                            "moe_down",
                            Lfm2Kernels::expertDownResidualQ8_0,
                            context,
                            ws.wrapHb,
                            ws.wrapX,
                            down.asByteArray(),
                            ws.wrapSelectedExperts,
                            ws.wrapSelectedExpertWeights,
                            dim,
                            expertHidden,
                            used,
                            MATVEC_LOCAL);
            default -> throw unsupported(layerIndex, "moe_down", "ffn_down_exps", down.dataType());
        }
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    /** This layer's weights, uploaded once in the graph that reads them. */
    private void transferLayerWeights(TaskGraph layer, int layerIndex) {
        List<Object> tensors = new ArrayList<>();
        tensors.add(require(weights.rms_att_weightLayered, layerIndex, "attn_norm").asFloatArray());
        tensors.add(require(weights.rms_ffn_weightLayered, layerIndex, "ffn_norm").asFloatArray());
        if (config.isMoeLayer(layerIndex)) {
            tensors.add(deviceArray(require(weights.ffnGateInp, layerIndex, "ffn_gate_inp")));
            tensors.add(require(weights.expProbsBias, layerIndex, "exp_probs_b").asFloatArray());
            tensors.add(deviceArray(require(weights.ffnGateExps, layerIndex, "ffn_gate_exps")));
            tensors.add(deviceArray(require(weights.ffnUpExps, layerIndex, "ffn_up_exps")));
            tensors.add(deviceArray(require(weights.ffnDownExps, layerIndex, "ffn_down_exps")));
        } else {
            tensors.add(deviceArray(require(weights.w1Layered, layerIndex, "ffn_gate")));
            tensors.add(deviceArray(require(weights.w2Layered, layerIndex, "ffn_down")));
            tensors.add(deviceArray(require(weights.w3Layered, layerIndex, "ffn_up")));
        }
        if (config.isAttentionLayer(layerIndex)) {
            tensors.add(deviceArray(require(weights.wqLayered, layerIndex, "attn_q")));
            tensors.add(deviceArray(require(weights.wkLayered, layerIndex, "attn_k")));
            tensors.add(deviceArray(require(weights.wvLayered, layerIndex, "attn_v")));
            tensors.add(deviceArray(require(weights.woLayered, layerIndex, "attn_output")));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray());
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray());
        } else {
            tensors.add(deviceArray(require(weights.convInProj, layerIndex, "shortconv.in_proj")));
            tensors.add(require(weights.convKernel, layerIndex, "shortconv.conv").asFloatArray());
            tensors.add(
                    deviceArray(require(weights.convOutProj, layerIndex, "shortconv.out_proj")));
        }
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, tensors.toArray());
    }

    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        var ws = lfm2State.workspace;
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    ws.positionHolder,
                    ws.temp,
                    ws.tempFFN,
                    ws.wrapBlockTable);
            // The convolution windows persist across tokens and are updated in place: uploaded
            // once, zeroed, and never read back.
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    ws.wrapXb,
                    ws.wrapQ,
                    ws.wrapK,
                    ws.wrapV,
                    keyStore(),
                    valueStore(),
                    ws.wrapAtt,
                    ws.wrapHb,
                    ws.wrapSsmQkv,
                    ws.wrapSsmConvOut,
                    ws.wrapConvState,
                    ws.wrapRouterLogits,
                    ws.wrapSelectedExperts,
                    ws.wrapSelectedExpertWeights);
        } else {
            String predecessor = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    ws.wrapXb,
                    ws.wrapQ,
                    ws.wrapK,
                    ws.wrapV,
                    keyStore(),
                    valueStore(),
                    ws.wrapAtt,
                    ws.wrapHb,
                    ws.wrapSsmQkv,
                    ws.wrapSsmConvOut,
                    ws.wrapConvState,
                    ws.wrapRouterLogits,
                    ws.wrapSelectedExperts,
                    ws.wrapSelectedExpertWeights,
                    ws.positionHolder,
                    ws.temp,
                    ws.tempFFN,
                    ws.wrapBlockTable);
        }
        return layer;
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsReduce =
                rmsReduceWorker(
                        WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize));
        WorkerGrid rmsApply = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);
        WorkerGrid rmsFinalize =
                WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        final int headDim = config.headSize();
        WorkerGrid qkNorm =
                WorkerGridFactory.genericWorker(
                        (config.numberOfHeads() + config.numberOfKeyValueHeads()) * headDim,
                        headDim);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(config.numberOfHeads() * (headDim / 2), 32);
        WorkerGrid kvAppend = WorkerGridFactory.genericWorker(config.kvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid attention =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), headDim);
        WorkerGrid conv = WorkerGridFactory.genericWorker(config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid single = WorkerGridFactory.genericWorker(1, 1);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "layer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            if (shouldUseFinalNormalization()) {
                scheduler.addWorkerGrid(prefix + "attn_rms_finalize", rmsFinalize);
                scheduler.addWorkerGrid(prefix + "ffn_rms_finalize", rmsFinalize);
            }
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);

            if (config.isAttentionLayer(layer)) {
                scheduler.addWorkerGrid(prefix + "attn_q_proj", matVecWorker(config.dim()));
                scheduler.addWorkerGrid(prefix + "attn_k_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_v_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                scheduler.addWorkerGrid(prefix + "attn_output_proj", matVecWorker(config.dim()));
            } else {
                scheduler.addWorkerGrid(
                        prefix + "conv_in_proj", matVecWorker(config.shortConvProjectionDim()));
                scheduler.addWorkerGrid(prefix + "conv", conv);
                scheduler.addWorkerGrid(prefix + "conv_out_proj", matVecWorker(config.dim()));
            }

            if (config.isMoeLayer(layer)) {
                scheduler.addWorkerGrid(prefix + "moe_router", matVecWorker(config.expertCount()));
                scheduler.addWorkerGrid(prefix + "moe_select", single);
                scheduler.addWorkerGrid(
                        prefix + "moe_gate_up",
                        matVecWorker(config.expertUsedCount() * config.expertHiddenDim()));
                scheduler.addWorkerGrid(prefix + "moe_down", matVecWorker(config.dim()));
            } else {
                scheduler.addWorkerGrid(prefix + "ffn_gate_up", matVecWorker(config.hiddenDim()));
                scheduler.addWorkerGrid(prefix + "ffn_down_proj", matVecWorker(config.dim()));
            }
        }
        return scheduler;
    }

    /** One workgroup per output row, which is how every matrix-vector kernel here is written. */
    private static WorkerGrid matVecWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * MATVEC_LOCAL, MATVEC_LOCAL);
    }
}
