package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Q8_0 transformer-layer TaskGraphs for Devstral 2 models. Uses precomputed RoPE frequencies (YaRN
 * scaling) instead of on-the-fly computation.
 */
public class DevstralQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, DevstralConfiguration> {

    public DevstralQ8_0FFNLayers(
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

        int fusedQkvGlobal = (config.qDim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.qDim() / 2, 512);
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
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
