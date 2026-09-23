package org.beehive.jllm.backend.tornado.layers.type.fp16;

import org.beehive.jllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jllm.backend.tornado.kernels.GranitePagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class GraniteFP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<GraniteTornadoWeights, GraniteConfiguration> {

    /**
     * @see org.beehive.jllm.backend.tornado.layers.type.fp16.LlamaFP16FFNLayers#useSimd32QkvFusion
     */
    private final boolean useSimd32QkvFusion =
            SchedulerDetectionService.isSubgroupShuffle32Supported();

    /**
     * @see LlamaFP16FFNLayers#packedHalf2Math
     */
    private final boolean packedHalf2Math = SchedulerDetectionService.isPackedHalf2MathSupported();

    public GraniteFP16FFNLayers(
            String taskGraph,
            State state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
        setupFFNLayers();
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

        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        int fusedQKVRows = config.dim() + 2 * config.kvDim();
        int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker =
                WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);

        // Map workers to tasks
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // === Attention Block ===
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_apply_fp16", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQKVWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            // === FFN Block ===
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }
        return tornadoForwardScheduler;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (LlamaFP16FFNLayers)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (partial sums)
     * └────────┬────────┘ │ ▼ (optional: NON_NVIDIA only) ┌──────────────────┐ │
     * attn_rms_finalize│──▶ temp (final scale) └────────┬─────────┘ │ ▼ ┌─────────────────────┐ │
     * attn_rms_apply_fp16 │──▶ wrapXbFP16 (normalized, FP16) └──────────┬──────────┘ │ ▼
     * ┌────────────────┐ ┌─────────────────────────────┐ │ qkv_projection │──────▶│ wrapQ, wrapK,
     * wrapV (FP32) │ └───────┬────────┘ └─────────────────────────────┘ │ ▼ ┌───────────────────┐
     * ┌─────────────────────────────────────┐ │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache,
     * ValueCache │ └─────────┬─────────┘ └─────────────────────────────────────┘ │ ▼ ┌───────────┐
     * │ attention │──▶ wrapXb (attention output) └─────┬─────┘ │ ▼ ┌──────────────────┐ │
     * attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (partial sums) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌─────────────────┐ │ ffn_rms_finalize│──▶ tempFFN (final scale)
     * └────────┬────────┘ │ ▼ ┌─────────────────┐ │ rms_ffn_gate_up │──▶ wrapHb =
     * SiLU(RMSNorm(x)·W1) ⊙ (RMSNorm(x)·W3) └────────┬────────┘ (fused: RMS apply + W1/W3 matmuls +
     * SiLU + GLU) │ ▼ ┌──────────────┐ │ ffn_down_proj│──▶ wrapX += W2 · wrapHb (residual
     * connection) └──────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 9 tasks (7 if NVIDIA, skipping rms_finalize steps)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points: • qkv_projection: Fused Q/K/V matmuls (3→1 kernel) • rope_and_kv_cache:
     * Fused RoPE rotation + cache write (2→1 kernel) • rms_ffn_gate_up: Fused RMS apply + W1/W3
     * matmuls + SiLU + GLU (4→1 kernel)
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());
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
                "attn_rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantize,
                context,
                state.workspace.wrapXbFP16,
                state.workspace.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.temp);

        // QKV Projection (fused). Same logical operation, same task name, same Q/K/V output
        // contract either way; only the reduction kernel differs, by device capability.
        if (useSimd32QkvFusion) {
            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsLayered::fusedQKVMatmulXSimd32,
                    context,
                    state.workspace.wrapXbFP16,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.kvDim());
        } else {
            unifiedLayer.task(
                    "qkv_projection",
                    packedHalf2Math
                            ? TransformerComputeKernelsLayered::fusedQKVMatmulX
                            : TransformerComputeKernelsLayered::fusedQKVMatmulXFp32Products,
                    context,
                    state.workspace.wrapXbFP16, // input (FP32)
                    state.workspace.wrapQ, // output Q
                    state.workspace.wrapK, // output K
                    state.workspace.wrapV, // output V
                    weights.wqLayered[layerIndex].asHalfFloatArray(), // Wq
                    weights.wkLayered[layerIndex].asHalfFloatArray(), // Wk
                    weights.wvLayered[layerIndex].asHalfFloatArray(), // Wv
                    config.dim(), // dim
                    config.kvDim(), // kvDim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

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
                    config.ropeTheta(), // needs to load it from model
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
                    config.ropeTheta(), // needs to load it from model
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
        // Attention
        configureAttention(unifiedLayer, layerIndex, config);
        // Output Projection (Wo) with residual
        unifiedLayer.task(
                "attn_output_proj",
                GraniteKernels::matrixVectorGenericWithResidualGranite,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.woLayered[layerIndex].asHalfFloatArray(),
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

        unifiedLayer.task(
                "rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUp,
                context,
                state.workspace.wrapX, // raw input (FP32)
                state.workspace.wrapHb, // output
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                state.workspace.tempFFN, // RMS scale factor
                weights.w1Layered[layerIndex].asHalfFloatArray(), // W1
                weights.w3Layered[layerIndex].asHalfFloatArray(), // W3
                config.dim(), // input dimension
                config.hiddenDim(), // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down projection (W2) with residual
        unifiedLayer.task(
                "ffn_down_proj",
                GraniteKernels::matrixVectorGenericWithResidualGranite,
                context,
                state.workspace.wrapHb,
                state.workspace.wrapX,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                config.hiddenDim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC,
                config.residualScale());

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer initial data to device (one-time transfer)
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.positionHolder,
                    state.workspace.temp,
                    state.workspace.tempFFN);
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    // Kernel context
                    context,
                    // Intermediate buffers
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    // QKV vectors
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    // KV cache
                    keyCache(),
                    valueCache(),
                    // Attention & FFN buffers
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.wrapXbFP16);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(
                    // Kernel context
                    context,
                    // Intermediate buffers
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2,
                    // QKV vectors
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    // KV cache
                    keyCache(),
                    valueCache(),
                    // Attention & FFN buffers
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    // Position & misc
                    state.workspace.positionHolder,
                    state.workspace.wrapXbFP16);
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
    }

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
