package org.beehive.jitllm.backend.tornado.layers.type.fp16;

import org.beehive.jitllm.backend.tornado.kernels.Phi3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Phi3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Phi3FP16FFNLayers: FP16 transformer-layer TaskGraphs for Phi3 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Qwen2/Qwen3: - Uses combined QKV matrix (wqkv) instead of separate Q, K,
 * V matrices - Includes splitQKV task to separate combined buffer - Uses ropeRotationPhi3 kernel
 * for position embeddings - FFN uses single wUp matrix that outputs both Gate and Up (2 *
 * hiddenDim) - Includes splitGateUpAndSiLU task for FFN activation - Uses wDown for final FFN
 * projection - No Q, K, V bias terms
 *
 * <p>Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3FP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Phi3TornadoWeights, Phi3Configuration> {

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;
    private final boolean useSimd32 = SchedulerDetectionService.isSubgroupShuffle32Supported();

    public Phi3FP16FFNLayers(
            String taskGraphName,
            Phi3State state,
            Phi3TornadoWeights weights,
            Phi3Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.phi3State = state;
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());
        setupFFNLayers();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        // RMS norm worker
        WorkerGrid rmsNormWorker =
                WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);

        // Fused RMS + QKV matmul worker
        int fusedQkvGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused RoPE + cache copy worker (Phi3 uses dim/2 pattern)
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 128);

        // Parallel attention worker
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        // Output projection worker
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker =
                WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused RMS + FFN gate/up worker
        int fusedFFNGlobal = (2 * config.hiddenDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNWorker =
                WorkerGridFactory.genericWorker(fusedFFNGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN down projection worker
        int ffnDownGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnDownWorker =
                WorkerGridFactory.genericWorker(ffnDownGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
        // Same worker as before - total rows = dim + 2*kvDim = opSize

        for (int i = 0; i < config.numberOfLayers(); i++) {
            // === Attention Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQkvWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", matmul1Worker);
            // === FFN Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_silu", fusedFFNWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", ffnDownWorker);
        }
        return gridScheduler;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Phi3FP16FFNLayers - Fully Optimized)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (scale factor for
     * RMSNorm) └────────┬────────┘ │ ▼ ┌────────────────────────┐ │ attn_rms_qkv_projection│──▶
     * wrapQ, wrapK, wrapV (direct output) └───────────┬────────────┘ (fused: RMS apply + QKV matmul
     * + split) │ ▼ ┌───────────────────┐ ┌─────────────────────────────────────┐ │
     * rope_and_kv_cache │───▶│ Q,K rotated + KeyCache, ValueCache │ └─────────┬─────────┘
     * └─────────────────────────────────────┘ │ (fused: Phi3 RoPE + cache write) ▼ ┌───────────┐ │
     * attention │──▶ wrapXb (attention output) └─────┬─────┘ │ ▼ ┌──────────────────┐ │
     * attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (scale factor) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌──────────────────┐ │ ffn_rms_finalize │──▶ tempFFN (final
     * scale) └────────┬─────────┘ │ ▼ ┌──────────────┐ │ rms_ffn_silu │──▶ wrapHbU =
     * SiLU(RMSNorm(x)·Wgate) ⊙ (RMSNorm(x)·Wup) └──────┬───────┘ (fused: RMS apply + gate/up matmul
     * + SiLU + GLU) │ ▼ ┌──────────────┐ │ ffn_down_proj│──▶ wrapX += wDown · wrapHbU (residual
     * connection) └──────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 8 tasks (NVIDIA) / 9 tasks (non-NVIDIA) Original: 13 tasks Reduction: 5 tasks
     * eliminated (38% fewer kernel launches)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points (vs original 13 tasks): • attn_rms_qkv_projection: Fused RMS apply + QKV
     * matmul + direct split (3→1 kernel) • rope_and_kv_cache: Fused Phi3 RoPE rotation + cache
     * write (2→1 kernel) • rms_ffn_silu: Fused RMS apply + gate/up matmul + SiLU + GLU (3→1 kernel)
     *
     * <p>Phi3-Specific: • Combined wqkv: Single [opSize × dim] matrix for Q+K+V projection • Direct
     * QKV output: No intermediate buffer, routes by row index • Phi3 RoPE: Uses headSize/2 offset
     * pattern (different from Llama/Qwen) • Combined wUp: Single [2×hiddenDim × dim] matrix for
     * gate+up • Inline SiLU+GLU: No intermediate wrapHb buffer needed
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        Phi3TornadoWeights weights = (Phi3TornadoWeights) this.weights;
        var taskGraphName = "layer_" + layerIndex;
        var unifiedLayer = new TaskGraph(taskGraphName);

        unifiedLayer.consumeFromDevice(phi3State.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                // Attention weights
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                // FFN weights
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asHalfFloatArray(),
                weights.wDownLayered[layerIndex].asHalfFloatArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                phi3State.workspace.temp, // output: scale factor
                phi3State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                phi3State.localSize); // local memory size

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
                "attn_rms_qkv_projection",
                useSimd32
                        ? Phi3Kernels::fusedRmsNormQKVMatmulDirectSimd32
                        : Phi3Kernels::fusedRmsNormQKVMatmulDirect,
                context,
                phi3State.workspace.wrapX, // input
                phi3State.workspace.wrapQ, // output Q
                phi3State.workspace.wrapK, // output K
                phi3State.workspace.wrapV, // output V
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                phi3State.workspace.temp, // RMS scale
                weights.wqkvLayered[layerIndex].asHalfFloatArray(),
                config.dim(), // dim
                config.kvDim(), // kvDim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Phi3 RoPE Rotation + KV Cache Write
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    Phi3PagedKvKernels::ropeRotationWithCacheCopyPhi3FP16Paged,
                    context,
                    phi3State.workspace.positionHolder, // current position
                    phi3State.workspace.wrapQ, // Q vectors (in/out, rotated)
                    phi3State.workspace.wrapK, // K vectors (in/out, rotated)
                    phi3State.workspace.wrapV, // V vectors (in only)
                    phi3State.workspace.wrapKeyCacheFP16, // key cache (out)
                    phi3State.workspace.wrapValueCacheFP16, // value cache (out)
                    config.numberOfKeyValueHeads(), // nHeadKv
                    config.headSize(), // head dimension
                    config.kvDim(), // kvDim
                    layerIndex, // layer index for cache offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        } else {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    Phi3PagedKvKernels::ropeRotationWithCacheCopyPhi3Paged,
                    context,
                    phi3State.workspace.positionHolder, // current position
                    phi3State.workspace.wrapQ, // Q vectors (in/out, rotated)
                    phi3State.workspace.wrapK, // K vectors (in/out, rotated)
                    phi3State.workspace.wrapV, // V vectors (in only)
                    state.workspace.wrapKeyCache, // key cache (out)
                    state.workspace.wrapValueCache, // value cache (out)
                    config.numberOfKeyValueHeads(), // nHeadKv
                    config.headSize(), // head dimension
                    config.kvDim(), // kvDim
                    layerIndex, // layer index for cache offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        }

        // Flash Attention
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    phi3State.workspace.wrapQ, // query vectors
                    phi3State.workspace.wrapKeyCacheFP16, // key cache
                    phi3State.workspace.wrapValueCacheFP16, // value cache
                    phi3State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    config.headSize(), // headSize
                    config.kvDim(), // kvDim
                    config.kvMul(), // kvMul (nHeads / nHeadKv)
                    phi3State.workspace.positionHolder, // position
                    layerIndex, // layer index
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // context length
        } else {
            unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    phi3State.workspace.wrapQ, // query vectors
                    state.workspace.wrapKeyCache, // key cache
                    state.workspace.wrapValueCache, // value cache
                    phi3State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    config.headSize(), // headSize
                    config.kvDim(), // kvDim
                    config.kvMul(), // kvMul (nHeads / nHeadKv)
                    phi3State.workspace.positionHolder, // position
                    layerIndex, // layer index
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // context length
        }

        // Output Projection with Residual
        if (useSimd32) {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    phi3State.workspace.wrapXb,
                    phi3State.workspace.wrapX,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    phi3State.workspace.wrapXb,
                    phi3State.workspace.wrapX,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                "ffn_rms_reduce",
                rmsReduceKernel(),
                context,
                phi3State.workspace.tempFFN, // output: scale factor
                phi3State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                phi3State.localSize); // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    phi3State.workspace.tempFFN, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        unifiedLayer.task(
                "rms_ffn_silu",
                useSimd32
                        ? Phi3Kernels::fusedRmsNormFFNGateUpSiLUSimd32
                        : Phi3Kernels::fusedRmsNormFFNGateUpSiLU,
                context,
                phi3State.workspace.wrapX, // input
                phi3State.workspace.wrapHbU, // output (direct to final FFN buffer)
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                phi3State.workspace.tempFFN, // RMS scale
                weights.wUpLayered[layerIndex].asHalfFloatArray(),
                config.dim(), // input dim
                config.hiddenDim(), // output dim (hiddenDim, not 2×hiddenDim!)
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual
        if (useSimd32) {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    phi3State.workspace.wrapHbU,
                    phi3State.workspace.wrapX,
                    weights.wDownLayered[layerIndex].asHalfFloatArray(),
                    config.hiddenDim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    phi3State.workspace.wrapHbU,
                    phi3State.workspace.wrapX,
                    weights.wDownLayered[layerIndex].asHalfFloatArray(),
                    config.hiddenDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        unifiedLayer.persistOnDevice(phi3State.workspace.wrapX);
        return unifiedLayer;
    }

    /** Configure data transfers for first and subsequent layers */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and state every execution
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    phi3State.workspace.positionHolder,
                    phi3State.workspace.temp,
                    phi3State.workspace.tempFFN);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    phi3State.workspace.wrapXb,
                    phi3State.workspace.wrapXb2,
                    phi3State.workspace.wrapQ,
                    phi3State.workspace.wrapK,
                    phi3State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    phi3State.workspace.wrapAtt,
                    phi3State.workspace.wrapHb,
                    phi3State.workspace.wrapHbG,
                    phi3State.workspace.wrapHbU,
                    phi3State.workspace.wrapQkv);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(
                    context,
                    phi3State.workspace.wrapXb,
                    phi3State.workspace.wrapXb2,
                    phi3State.workspace.wrapQ,
                    phi3State.workspace.wrapK,
                    phi3State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    phi3State.workspace.wrapAtt,
                    phi3State.workspace.wrapHb,
                    phi3State.workspace.positionHolder,
                    phi3State.workspace.wrapHbG,
                    phi3State.workspace.wrapHbU,
                    phi3State.workspace.wrapQkv);
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
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
