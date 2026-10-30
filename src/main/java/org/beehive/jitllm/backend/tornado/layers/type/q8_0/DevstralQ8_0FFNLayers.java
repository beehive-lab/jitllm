package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.layers.QuantizedProjections;
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

/**
 * Q8_0 transformer-layer TaskGraphs for Devstral 2 models. Uses precomputed RoPE frequencies (YaRN
 * scaling) instead of on-the-fly computation.
 *
 * <p>Where the device runs packed integer dot products the projections are Llama's {@code dp4a}
 * kernels over the activation quantized to Q8 blocks, sized for Devstral's non-square attention
 * ({@code qDim != dim}); elsewhere the floating-point kernels this family always used.
 */
public class DevstralQ8_0FFNLayers extends DevstralLayerTaskGraphs {

    public DevstralQ8_0FFNLayers(
            State state,
            LlamaTornadoWeights weights,
            DevstralConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
        this.projections =
                new QuantizedProjections(
                        context, state.workspace, LOCAL_WORK_GROUP_SIZE_ALLOC, PACKED_LOCAL);
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = GraphNames.layer(layerIndex);
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
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
        String weightSrc = weightSourceGraphName(layerIndex);
        if (weightSrc != null) {
            unifiedLayer.consumeFromDevice(weightSrc, layerWeights);
        } else {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights);
        }
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        bindSharedScratch(unifiedLayer, layerIndex);
        boolean packed = packed();

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

        if (nativeLayer(layerIndex)) {
            projections.rmsApply(
                    unifiedLayer,
                    "attn_rms_apply",
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
            // Three projections rather than one fused kernel: wq, wk and wv can be different
            // representations in the same layer.
            projections.matvec(
                    unifiedLayer,
                    "q_proj",
                    state.workspace.wrapQ,
                    weights.wqLayered[layerIndex],
                    config.dim(),
                    config.qDim());
            projections.matvec(
                    unifiedLayer,
                    "k_proj",
                    state.workspace.wrapK,
                    weights.wkLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
            projections.matvec(
                    unifiedLayer,
                    "v_proj",
                    state.workspace.wrapV,
                    weights.wvLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
        } else if (packed) {
            // RMS apply and the Q8 quantization of its output in one task, then Q, K and V in
            // one launch over the quantized activation.
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "attn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsQ8_0DP4A::fusedQKVMatmulQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.qDim(),
                    config.kvDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    "attn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);

            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsLayered::fusedQKVMatmulQ8NonSquare,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.qDim(),
                    config.kvDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Precomputed RoPE frequencies (YaRN-scaled)
        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        configureAttention(unifiedLayer, layerIndex);

        // O projection: n=qDim (input), d=dim (output)
        if (nativeLayer(layerIndex)) {
            projections.residualMatvec(
                    unifiedLayer,
                    "attn_out_quantize",
                    "attn_output_proj",
                    state.workspace.wrapXb,
                    weights.woLayered[layerIndex],
                    config.qDim(),
                    config.dim());
        } else if (packed) {
            // The attention output is a new activation: quantized fresh for the projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    "attn_out_quantize",
                    context,
                    state.workspace,
                    state.workspace.wrapXb);
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.qDim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.qDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
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

        if (nativeLayer(layerIndex)) {
            projections.rmsApply(
                    unifiedLayer,
                    "ffn_rms_apply",
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
            projections.gateUp(
                    unifiedLayer,
                    "ffn_gate_up",
                    weights.w1Layered[layerIndex],
                    weights.w3Layered[layerIndex],
                    config.dim(),
                    config.hiddenDim());
            projections.residualMatvec(
                    unifiedLayer,
                    "ffn_down_quantize",
                    "ffn_down_proj",
                    state.workspace.wrapHb,
                    weights.w2Layered[layerIndex],
                    config.hiddenDim(),
                    config.dim());
        } else if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    "ffn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
            unifiedLayer.task(
                    "ffn_gate_up",
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
            // SwiGLU's output, quantized fresh for the down projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    "ffn_down_quantize",
                    context,
                    state.workspace,
                    state.workspace.wrapHb);
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                    context,
                    state.workspace.wrapX,
                    state.workspace.wrapHb,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);

            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
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
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            if (splitsGateUp()) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapHb2);
            }
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
                    state.workspace.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            if (splitsGateUp()) {
                unifiedLayer.consumeFromDevice(state.workspace.wrapHb2);
            }
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        addSharedWorkerGrids(tornadoForwardScheduler);
        if (packed()) {
            int qkvRows = config.qDim() + 2 * config.kvDim();
            for (int i = 0; i < config.numberOfLayers(); i++) {
                String l = GraphNames.layer(i) + ".";
                if (nativeLayer(i)) {
                    addNativeWorkerGrids(tornadoForwardScheduler, l, i);
                    continue;
                }
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "qkv_projection",
                        PackedActivationTasks.projectionWorker(qkvRows, PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_out_quantize",
                        PackedActivationTasks.quantizeWorker(config.qDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_gate_up",
                        PackedActivationTasks.projectionWorker(config.hiddenDim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_quantize",
                        PackedActivationTasks.quantizeWorker(config.hiddenDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
            }
            return tornadoForwardScheduler;
        }

        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker =
                WorkerGridFactory.genericWorker(
                        configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedQkvGlobal = (config.qDim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String l = GraphNames.layer(i) + ".";
            if (nativeLayer(i)) {
                addNativeWorkerGrids(tornadoForwardScheduler, l, i);
                continue;
            }
            tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }

    // @formatter:on

    /** The per-weight projections of a layer that is not entirely Q8_0. */
    private final QuantizedProjections projections;

    /**
     * Whether a layer holds any projection in a representation other than Q8_0. Such a layer
     * applies each norm once and projects one weight at a time, each through the kernel for its own
     * type ({@link QuantizedProjections}); an all-Q8_0 layer keeps its fused kernels. A file whose
     * projections are all Q4_K and Q6_K runs the K-quant layers instead.
     */
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

    /** Whether some layer runs its gate and up projections apart, which needs {@code wrapHb2}. */
    protected boolean splitsGateUp() {
        for (int i = 0; i < config.numberOfLayers(); i++) {
            if (nativeLayer(i)
                    && !projections.fusedGateUp(weights.w1Layered[i], weights.w3Layered[i])) {
                return true;
            }
        }
        return false;
    }

    /** The grids of a layer that projects one weight at a time: one workgroup per output row. */
    private void addNativeWorkerGrids(GridScheduler scheduler, String l, int layerIndex) {
        WorkerGrid rmsApplyWorker = projections.rmsApplyWorker(config.dim());
        scheduler.addWorkerGrid(l + "attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(l + "q_proj", projections.worker(config.qDim()));
        scheduler.addWorkerGrid(l + "k_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(l + "v_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(l + "attn_output_proj", projections.worker(config.dim()));
        scheduler.addWorkerGrid(l + "ffn_rms_apply", rmsApplyWorker);
        projections.addGateUpGrids(
                scheduler,
                l,
                "ffn_gate_up",
                weights.w1Layered[layerIndex],
                weights.w3Layered[layerIndex],
                config.hiddenDim());
        scheduler.addWorkerGrid(l + "ffn_down_proj", projections.worker(config.dim()));
        if (projections.packed()) {
            scheduler.addWorkerGrid(
                    l + "attn_out_quantize", PackedActivationTasks.quantizeWorker(config.qDim()));
            scheduler.addWorkerGrid(
                    l + "ffn_down_quantize",
                    PackedActivationTasks.quantizeWorker(config.hiddenDim()));
        }
    }
}
