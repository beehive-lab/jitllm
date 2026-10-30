package org.beehive.jitllm.backend.tornado.layers.type.q4_k;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.MlxDecodeProjections;
import org.beehive.jitllm.backend.tornado.layers.MlxWeightCache;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.layers.devstral.DevstralLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Q4_K transformer-layer TaskGraphs for Devstral 2 models, reading the weights <b>in the file's own
 * representation</b> rather than a Q8_0 materialization of them.
 *
 * <p>Structurally the Q8_0 sibling with three kernels swapped for their Q4_K counterparts: the
 * fused non-square QKV projection, the two residual matvecs, and the FFN gate/up. Everything else —
 * the RMS reductions, RoPE with the precomputed YaRN frequencies, attention, the transfer and
 * persistence structure — is dtype-independent and unchanged.
 *
 * <p>One deliberate difference: the Q8_0 path uses {@code fullyFusedRmsNormFFNGateUpQ8}, which
 * folds the FFN RMS norm into the gate/up projection. Here the norm is applied by the existing
 * {@code reductionOneBlock2WithLayer} task, exactly as the attention block already does, and the
 * Q4_K gate/up reads the normalized activation. That reuses a verified kernel rather than restating
 * the normalization inside a new one, at the cost of one extra task per layer.
 *
 * <p>Where the device runs packed integer dot products ({@link PackedActivationTasks#enabled()})
 * each projection instead reads the activation quantized once to Q8 blocks, with the {@code dp4a}
 * kernel of its own representation (Q4_K, Q6_K or Q8_0), and gate and up run as one fused SwiGLU
 * launch when both are Q4_K.
 */
public class DevstralQ4_KFFNLayers extends DevstralLayerTaskGraphs {

    public DevstralQ4_KFFNLayers(
            String taskGraphName,
            State state,
            LlamaTornadoWeights weights,
            DevstralConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        MlxDecodeProjections mlx = mlxDecode(layerIndex);
        // On MLX the projections are MLX's form of them, bound from the prefill graph below; the
        // file's own blocks are never uploaded.
        Object[] layerWeights =
                mlx != null
                        ? new Object[] {
                            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                            weights.rms_ffn_weightLayered[layerIndex].asFloatArray()
                        }
                        : new Object[] {
                            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                            weights.wqLayered[layerIndex].asByteArray(),
                            weights.wkLayered[layerIndex].asByteArray(),
                            weights.wvLayered[layerIndex].asByteArray(),
                            weights.woLayered[layerIndex].asByteArray(),
                            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                            weights.w1Layered[layerIndex].asByteArray(),
                            weights.w2Layered[layerIndex].asByteArray(),
                            weights.w3Layered[layerIndex].asByteArray()
                        };
        // From the batched prefill's graph where it uploaded them; the projections an MLX prefill
        // replaced with its own form are uploaded here (weightsNotProvidedBySource).
        bindLayerWeights(unifiedLayer, weightSourceGraphName(layerIndex), layerIndex, layerWeights);
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        bindSharedScratch(unifiedLayer, layerIndex);
        if (mlx != null) {
            mlx.declare(
                    unifiedLayer,
                    layerIndex,
                    layerIndex == firstLayer,
                    "layer_" + (layerIndex - 1),
                    weightSourceGraphName(layerIndex));
        }
        // MLX reads the FP32 normed activation, not the packed int8 one.
        boolean packed = packed() && mlx == null;

        unifiedLayer.task(
                "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.temp,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "attn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
        } else {
            unifiedLayer.task(
                    "attn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
        }

        if (mlx != null) {
            mlx.qkv(unifiedLayer, layerIndex);
        } else {
            // Three projections rather than one fused kernel: a K-quant file's wq, wk and wv can be
            // different representations in the same layer, and one kernel cannot decode two block
            // formats. Selected per tensor, at graph-construction time.
            matvec(
                    unifiedLayer,
                    "q_proj",
                    state.workspace.wrapXb,
                    state.workspace.wrapQ,
                    weights.wqLayered[layerIndex],
                    config.dim(),
                    config.qDim());
            matvec(
                    unifiedLayer,
                    "k_proj",
                    state.workspace.wrapXb,
                    state.workspace.wrapK,
                    weights.wkLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
            matvec(
                    unifiedLayer,
                    "v_proj",
                    state.workspace.wrapXb,
                    state.workspace.wrapV,
                    weights.wvLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
        }

        // Precomputed RoPE frequencies (YaRN-scaled)
        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        configureAttention(unifiedLayer, layerIndex);

        // O projection: n=qDim (input), d=dim (output)
        if (mlx != null) {
            mlx.attentionOutput(unifiedLayer, layerIndex);
        } else {
            if (packed) {
                // The attention output is a new activation: quantized fresh for the projection.
                PackedActivationTasks.quantize(
                        unifiedLayer,
                        "attn_out_quantize",
                        context,
                        state.workspace,
                        state.workspace.wrapXb);
            }
            residualMatvec(
                    unifiedLayer,
                    "attn_output_proj",
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex],
                    config.qDim(),
                    config.dim());
        }

        unifiedLayer.task(
                "ffn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempFFN,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempFFN,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "ffn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
        } else {
            unifiedLayer.task(
                    "ffn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
        }

        if (mlx != null) {
            mlx.ffn(unifiedLayer, layerIndex);
            unifiedLayer.persistOnDevice(state.workspace.wrapX);
            mlx.persist(unifiedLayer);
            return unifiedLayer;
        }
        if (fusedGateUp(layerIndex)) {
            // Both Q4_K: gate, up and SwiGLU in one launch over the quantized activation.
            unifiedLayer.task(
                    "ffn_gate_up",
                    TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_KDP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapXbSums,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
        } else {
            // Gate and up separately, for the same reason as q/k/v, then SwiGLU as its own task.
            matvec(
                    unifiedLayer,
                    "ffn_gate",
                    state.workspace.wrapXb,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex],
                    config.dim(),
                    config.hiddenDim());
            matvec(
                    unifiedLayer,
                    "ffn_up",
                    state.workspace.wrapXb,
                    state.workspace.wrapHb2,
                    weights.w3Layered[layerIndex],
                    config.dim(),
                    config.hiddenDim());
            unifiedLayer.task(
                    "ffn_silu",
                    TransformerComputeKernelsQ6_K::siluAndMultiply,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapHb2,
                    config.hiddenDim());
        }

        if (packed) {
            // SwiGLU's output, quantized fresh for the down projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    "ffn_down_quantize",
                    context,
                    state.workspace,
                    state.workspace.wrapHb);
        }
        residualMatvec(
                unifiedLayer,
                "ffn_down_proj",
                state.workspace.wrapHb,
                state.workspace.wrapX,
                weights.w2Layered[layerIndex],
                config.hiddenDim(),
                config.dim());

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    /** Built on first use: the graphs are built from the superclass's constructor. */
    private MlxDecodeProjections mlxDecode;

    private boolean mlxDecodeResolved;

    // @formatter:off
    /**
     * The MLX decode projections, when these are the decode layers of a batched plan whose prefill
     * runs on MLX in the affine form: the decode then reads the prefill's weights and the file's
     * projection blocks are never uploaded. Null otherwise — the single-token plan, or a prefill on
     * the tensor cores — and the Q4_K kernels run.
     */
    // @formatter:on
    private MlxDecodeProjections mlxDecode(int layerIndex) {
        if (!mlxDecodeResolved) {
            mlxDecodeResolved = true;
            mlxDecode =
                    weightSourceGraphName(layerIndex) == null || mlxPrefill.isEmpty()
                            ? null
                            : MlxDecodeProjections.ifAffine(
                                    mlxPrefill.get(),
                                    MlxWeightCache.standard(),
                                    state,
                                    weights,
                                    context,
                                    config.numberOfLayers(),
                                    config.dim(),
                                    config.qDim(),
                                    config.kvDim(),
                                    config.hiddenDim());
        }
        return mlxDecode;
    }

    /** Whether gate, up and SwiGLU run as one packed launch: both projections Q4_K. */
    private boolean fusedGateUp(int layerIndex) {
        return packed()
                && weights.w1Layered[layerIndex].dataType() == DataType.Q4_K
                && weights.w3Layered[layerIndex].dataType() == DataType.Q4_K;
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.positionHolder,
                    state.workspace.temp,
                    state.workspace.tempFFN);
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.wrapHb2,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            unifiedLayer.consumeFromDevice(
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.wrapHb2,
                    state.workspace.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        addSharedWorkerGrids(tornadoForwardScheduler);

        boolean packed = packed();
        int local = packed ? PACKED_LOCAL : LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid rmsApplyWorker =
                packed
                        ? PackedActivationTasks.quantizeWorker(config.dim())
                        : WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        WorkerGrid dimRowsWorker = WorkerGridFactory.genericWorker(config.dim() * local, local);
        WorkerGrid hiddenRowsWorker =
                WorkerGridFactory.genericWorker(config.hiddenDim() * local, local);
        WorkerGrid qProjWorker = WorkerGridFactory.genericWorker(config.qDim() * local, local);
        WorkerGrid kvProjWorker = WorkerGridFactory.genericWorker(config.kvDim() * local, local);
        // SwiGLU is elementwise over the hidden dimension, one thread per element.
        WorkerGrid siluWorker = WorkerGridFactory.genericWorker(config.hiddenDim(), 256);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String l = "layer_" + i + ".";
            tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_apply", rmsApplyWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "q_proj", qProjWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "k_proj", kvProjWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "v_proj", kvProjWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "attn_output_proj", dimRowsWorker);
            // This path applies the FFN norm as its own task rather than folding it into the
            // gate/up kernel, so it needs its own worker grid — the same one attn_rms_apply uses.
            tornadoForwardScheduler.addWorkerGrid(l + "ffn_rms_apply", rmsApplyWorker);
            if (fusedGateUp(i)) {
                tornadoForwardScheduler.addWorkerGrid(l + "ffn_gate_up", hiddenRowsWorker);
            } else {
                tornadoForwardScheduler.addWorkerGrid(l + "ffn_gate", hiddenRowsWorker);
                tornadoForwardScheduler.addWorkerGrid(l + "ffn_up", hiddenRowsWorker);
                tornadoForwardScheduler.addWorkerGrid(l + "ffn_silu", siluWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(l + "ffn_down_proj", dimRowsWorker);
            if (mlxDecode != null) {
                mlxDecode.addGrids(tornadoForwardScheduler, l);
            }
            if (packed) {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_out_quantize",
                        PackedActivationTasks.quantizeWorker(config.qDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_quantize",
                        PackedActivationTasks.quantizeWorker(config.hiddenDim()));
            }
        }

        return tornadoForwardScheduler;
    }

    /**
     * {@code out = w·x}, with the kernel chosen by what {@code w} actually holds.
     *
     * <p>The whole point of the K-quant path: a "Q4_K_M" file is mixed per tensor and per layer, so
     * the representation is a property of each weight rather than of the model. Selected here, at
     * graph-construction time, from the tensor's own neutral {@link DataType} — no runtime branch
     * reaches a kernel, and a tensor materialized as Q8_0 (Q6_K's siblings, or any format without a
     * device kernel) is read by the Q8_0 kernel it actually is.
     */
    private void matvec(
            TaskGraph graph,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        DataType type = w.dataType();
        if (packed()) {
            // The activation was quantized by the task before this one, from x.
            switch (type) {
                case Q4_K ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KDP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                state.workspace.wrapXbSums,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
                case Q6_K ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                state.workspace.wrapXbSums,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
                default ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
            }
            return;
        }
        if (type == DataType.Q4_K) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else if (type == DataType.Q6_K) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }
    }

    /** {@code out += w·x}, selected the same way as {@link #matvec}. */
    private void residualMatvec(
            TaskGraph graph,
            String name,
            FloatArray x,
            FloatArray out,
            TornadoTensor w,
            int n,
            int d) {
        DataType type = w.dataType();
        if (packed()) {
            switch (type) {
                case Q4_K ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ4_K
                                        ::matrixVectorGenericWithResidualQ4_KDP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                state.workspace.wrapXbSums,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
                case Q6_K ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ6_K
                                        ::matrixVectorGenericWithResidualQ6_KDP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                state.workspace.wrapXbSums,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
                default ->
                        graph.task(
                                name,
                                TransformerComputeKernelsQ8_0DP4A
                                        ::matrixVectorGenericWithResidualQ8_0DP4A,
                                context,
                                state.workspace.wrapXbQuants,
                                state.workspace.wrapXbScales,
                                out,
                                w.asByteArray(),
                                n,
                                d,
                                PACKED_LOCAL);
            }
            return;
        }
        if (type == DataType.Q4_K) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else if (type == DataType.Q6_K) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    x,
                    out,
                    w.asByteArray(),
                    n,
                    d,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }
    }

    // @formatter:on

}
