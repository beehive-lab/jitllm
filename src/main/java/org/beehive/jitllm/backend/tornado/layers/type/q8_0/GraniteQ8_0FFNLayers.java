package org.beehive.jllm.backend.tornado.layers.type.q8_0;

import org.beehive.jllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jllm.backend.tornado.kernels.GranitePagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.GraniteState;
import org.beehive.jllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class GraniteQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<GraniteTornadoWeights, GraniteConfiguration> {

    public GraniteQ8_0FFNLayers(
            String taskGraphName,
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                // Copy-in weights per layer for batched-layered layout (Q8 format)
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

        // === Attention Block ===
        // RMS Normalization
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

        // QKV Projection (fused with Q8 dequantization)
        unifiedLayer.task(
                "qkv_projection",
                TransformerComputeKernelsLayered::fusedQKVMatmulQ8,
                context,
                state.workspace.wrapXb, // input (FP32)
                state.workspace.wrapQ, // output Q
                state.workspace.wrapK, // output K
                state.workspace.wrapV, // output V
                weights.wqLayered[layerIndex].asByteArray(), // Wq (Q8)
                weights.wkLayered[layerIndex].asByteArray(), // Wk (Q8)
                weights.wvLayered[layerIndex].asByteArray(), // Wv (Q8)
                config.dim(), // dim
                config.kvDim(), // kvDim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // RoPE + KV Cache
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    GranitePagedKvKernels::ropeRotationWithCacheCopyFP16Paged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCacheFP16, // Key cache (out)
                    state.workspace.wrapValueCacheFP16, // Value cache (out)
                    config.kvDim(),
                    config.headSize(),
                    config.ropeTheta(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    GranitePagedKvKernels::ropeRotationWithCacheCopyPaged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCache, // Key cache (out)
                    state.workspace.wrapValueCache, // Value cache (out)
                    config.kvDim(),
                    config.headSize(),
                    config.ropeTheta(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }

        // Attention
        configureAttention(unifiedLayer, layerIndex, config);

        // Output Projection (Wo) with residual (Q8 dequantization)
        unifiedLayer.task(
                "attn_output_proj",
                GraniteKernels::matrixVectorGenericWithResidualQ8_0ByteWithGraniteScale,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC,
                config.residualScale());

        // === FFN Block ===
        // RMS Normalization
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

        // Fully fused: RMS apply + Gate/Up projections + SiLU + GLU (Q8 dequantization)
        unifiedLayer.task(
                "rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                context,
                state.workspace.wrapX, // raw input (FP32)
                state.workspace.wrapHb, // output
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                weights.w1Layered[layerIndex].asByteArray(), // W1 (Q8)
                weights.w3Layered[layerIndex].asByteArray(), // W3 (Q8)
                config.dim(), // input dimension
                config.hiddenDim(), // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down projection (W2) with residual (Q8 dequantization)
        unifiedLayer.task(
                "ffn_down_proj",
                GraniteKernels::matrixVectorGenericWithResidualQ8_0ByteWithGraniteScale,
                context,
                state.workspace.wrapHb,
                state.workspace.wrapX,
                weights.w2Layered[layerIndex].asByteArray(),
                config.hiddenDim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC,
                config.residualScale());

        // Keep activation X on device for next layer
        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    // @formatter:on

    // @formatter:off
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
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
                    state.workspace.wrapHb);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: Consume data already on device from previous layer
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
                    state.workspace.positionHolder);
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
    }

    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        // === Worker Grid Definitions ===
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

        // Fused QKV: dim rows for Q + kvDim rows for K + kvDim rows for V
        int fusedQkvGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);

        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        // === Per-Layer Grid Assignments (ordered by TaskGraph flow) ===
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // --- Attention Block ---
            // RMS Normalization
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
            // --- FFN Block ---
            // RMS Normalization
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            // Fused RMS + Gate/Up Projections
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            // Down Projection
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }

    // @formatter:off
    private TaskGraph configureAttention(
            TaskGraph unifiedLayer, int layerIndex, GraniteConfiguration config) {
        if (schedulerType == SchedulerType.NVIDIA) {
            // Flash Attention (optimized for NVIDIA GPUs)
            if (useFp16KVCache()) {
                return unifiedLayer.task(
                        "attention",
                        GranitePagedKvKernels::processHeadsFlashAttentionWithGraniteScaleFP16Paged,
                        context,
                        state.workspace.wrapQ, // Query
                        state.workspace.wrapKeyCacheFP16, // Key cache
                        state.workspace.wrapValueCacheFP16, // Value cache
                        state.workspace.wrapXb, // Output
                        config.numberOfHeads(),
                        config.headSize(),
                        config.kvDim(),
                        config.kvMul(),
                        state.workspace.positionHolder,
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        config.attentionScale());
            }
            return unifiedLayer.task(
                    "attention",
                    GranitePagedKvKernels::processHeadsFlashAttentionWithGraniteScalePaged,
                    context,
                    state.workspace.wrapQ, // Query
                    state.workspace.wrapKeyCache, // Key cache
                    state.workspace.wrapValueCache, // Value cache
                    state.workspace.wrapXb, // Output
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.workspace.positionHolder,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.attentionScale());
        } else {
            // Standard parallel attention (for non-NVIDIA backends)
            return unifiedLayer.task(
                    "attention",
                    GranitePagedKvKernels::processHeadsParallelGranitePaged,
                    state.workspace.wrapQ, // Query
                    state.workspace.wrapKeyCache, // Key cache
                    state.workspace.wrapValueCache, // Value cache
                    state.workspace.wrapXb, // Output
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    config.contextLength(), // seqLen parameter
                    state.workspace.positionHolder,
                    state.workspace.wrapAtt, // Attention weights buffer
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    config.attentionScale());
        }
    }

    // @formatter:on

    /** The key cache every graph of this family binds: FP16 when the state holds one. */
    protected Object keyCache() {
        return useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    /** The value cache, following {@link #keyCache()}. */
    protected Object valueCache() {
        return useFp16KVCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }
}
