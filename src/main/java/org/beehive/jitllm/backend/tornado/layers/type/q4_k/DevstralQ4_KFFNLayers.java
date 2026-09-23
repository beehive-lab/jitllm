package org.beehive.jitllm.backend.tornado.layers.type.q4_k;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
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
 */
public class DevstralQ4_KFFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, DevstralConfiguration> {

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

        unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

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

        unifiedLayer.task(
                "attn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.temp);

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

        // Use precomputed RoPE frequencies (YaRN-scaled)
        unifiedLayer.task(
                "rope_and_kv_cache",
                TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedPaged,
                context,
                state.workspace.positionHolder,
                state.workspace.wrapQ,
                state.workspace.wrapK,
                state.workspace.wrapV,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.kvDim(),
                config.headSize(),
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride);

        configureAttention(unifiedLayer, layerIndex);

        // O projection: n=qDim (input), d=dim (output)
        residualMatvec(
                unifiedLayer,
                "attn_output_proj",
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.woLayered[layerIndex],
                config.qDim(),
                config.dim());

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

        unifiedLayer.task(
                "ffn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.tempFFN);

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
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
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
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
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
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker =
                WorkerGridFactory.genericWorker(
                        configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid qProjWorker =
                WorkerGridFactory.genericWorker(
                        config.qDim() * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid kvProjWorker =
                WorkerGridFactory.genericWorker(
                        config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        // SwiGLU is elementwise over the hidden dimension, one thread per element.
        WorkerGrid siluWorker = WorkerGridFactory.genericWorker(config.hiddenDim(), 256);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.qDim() / 2, 512);
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".q_proj", qProjWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".k_proj", kvProjWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".v_proj", kvProjWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            // This path applies the FFN norm as its own task rather than folding it into the
            // gate/up kernel, so it needs its own worker grid — the same one attn_rms_apply uses.
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_gate", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_silu", siluWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
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

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (schedulerType == SchedulerType.NVIDIA) {
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsParallelPaged,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    config.contextLength(),
                    state.workspace.positionHolder,
                    state.workspace.wrapAtt,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
    }
    // @formatter:on

}
