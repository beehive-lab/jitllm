package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.Phi3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
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
 * Phi3Q8_0FFNLayers: Q8_0 transformer-layer TaskGraphs for Phi3 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Phi3FP16FFNLayers: - Uses Q8_0-quantized weights (getQuants() and
 * getScales()) - Same attention and RoPE kernels as FP16 version - 8-bit integer computations with
 * dequantization - 2x memory compression vs FP16 - Same combined QKV and gate/up FFN structure
 *
 * <p>Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3Q8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Phi3TornadoWeights, Phi3Configuration> {

    // Typed reference to Phi3-specific state
    private final Phi3State phi3State;
    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;

    public Phi3Q8_0FFNLayers(
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
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 128);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        final int opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        int qkvmatmulDimRowMajorGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid qkvDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        qkvmatmulDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());
        int ffnFusedGlobal = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnFusedWorker =
                WorkerGridFactory.genericWorker(ffnFusedGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            //                           ATTENTION BLOCK
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_qkv_projection_q8", qkvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            //                              FFN BLOCK
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_gateup_silu_q8", ffnFusedWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }
        return tornadoForwardScheduler;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Phi3Q8_0FFNLayers - Fully Optimized)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (scale factor for
     * RMSNorm) └────────┬────────┘ │ ▼ ┌─────────────────────────────┐ │ attn_rms_qkv_projection_q8
     * │──▶ wrapQ, wrapK, wrapV (direct output) └────────────┬────────────────┘ (fused: RMS apply +
     * Q8 QKV matmul + split) │ ▼ ┌───────────────────┐ ┌─────────────────────────────────────┐ │
     * rope_and_kv_cache │───▶│ Q,K rotated + KeyCache, ValueCache │ └─────────┬─────────┘
     * └─────────────────────────────────────┘ │ (fused: Phi3 RoPE + cache write) ▼ ┌───────────┐ │
     * attention │──▶ wrapXb (attention output) └─────┬─────┘ │ ▼ ┌──────────────────┐ │
     * attn_output_proj │──▶ wrapX += Wo · wrapXb (residual, Q8 dequant) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (scale factor) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌──────────────────┐ │ ffn_rms_finalize │──▶ tempFFN (final
     * scale) └────────┬─────────┘ │ ▼ ┌───────────────────────┐ │ ffn_rms_gateup_silu_q8│──▶
     * wrapHbU = SiLU(RMSNorm(x)·Wgate) ⊙ (RMSNorm(x)·Wup) └───────────┬───────────┘ (fused: RMS
     * apply + Q8 gate/up matmul + SiLU + GLU) │ ▼ ┌───────────────┐ │ ffn_down_proj │──▶ wrapX +=
     * wDown · wrapHbU (residual, Q8 dequant) └───────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or
     * logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 8 tasks (NVIDIA) / 9 tasks (non-NVIDIA) Original: 13 tasks Reduction: 5 tasks
     * eliminated (38% fewer kernel launches)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points (vs original 13 tasks): • attn_rms_qkv_projection_q8: Fused RMS apply +
     * Q8 QKV matmul + direct split (3→1 kernel) • rope_and_kv_cache: Fused Phi3 RoPE rotation +
     * cache write (2→1 kernel) • ffn_rms_gateup_silu_q8: Fused RMS apply + Q8 gate/up matmul + SiLU
     * + GLU (3→1 kernel)
     *
     * <p>Q8_0 Quantization Details: • Block size: 32 elements per quantization block • Block
     * format: 2-byte FP16 scale + 32 signed int8 quantized values (34 bytes total) •
     * Dequantization: value = quant * scale (inline during matmul) • Memory savings: ~2x
     * compression vs FP16, ~4x vs FP32
     *
     * <p>Phi3-Specific: • Combined wqkv: Single Q8 [opSize × dim] matrix for Q+K+V projection •
     * Direct QKV output: No intermediate buffer, routes by row index • Phi3 RoPE: Uses headSize/2
     * offset pattern (different from Llama/Qwen) • Combined wUp: Single Q8 [2×hiddenDim × dim]
     * matrix for gate+up • Inline SiLU+GLU: No intermediate wrapHb buffer needed
     *
     * <p>Buffers Eliminated by Fusion: • wrapXb (attention path): Not needed between RMS and QKV
     * projection • wrapQkv: Direct output to Q/K/V buffers • wrapHb: Not needed between gate/up
     * matmul and SiLU • wrapHbG: Gate output merged into final computation
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;
        var unifiedLayer = new TaskGraph(taskGraphName);

        unifiedLayer.consumeFromDevice(phi3State.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                // Copy-in quantized weights per layer (Q8_0 format: ByteArray)
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asByteArray(),
                weights.wDownLayered[layerIndex].asByteArray());
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

        // Fused: RMS apply + Q8 QKV matmul + direct Q/K/V split
        unifiedLayer.task(
                "attn_rms_qkv_projection_q8",
                TransformerComputeKernelsLayered::fusedRmsNormQKVMatmulQ8,
                context,
                phi3State.workspace.wrapX, // input: hidden state
                phi3State.workspace.wrapQ, // output Q [dim]
                phi3State.workspace.wrapK, // output K [kvDim]
                phi3State.workspace.wrapV, // output V [kvDim]
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                phi3State.workspace.temp, // RMS scale (precomputed)
                weights.wqkvLayered[layerIndex].asByteArray(), // Q8 combined QKV [opSize × dim]
                config.dim(), // input dim
                config.kvDim(), // K/V output dim
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

        // Output Projection with Residual (Q8 dequantization)
        unifiedLayer.task(
                "attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                phi3State.workspace.wrapXb, // input: attention output
                phi3State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asByteArray(), // Q8 Wo [dim × dim]
                config.dim(), // input dim
                config.dim(), // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

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

        // Fused: RMS apply + Q8 gate/up matmul + SiLU activation + GLU
        unifiedLayer.task(
                "ffn_rms_gateup_silu_q8",
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpSiLUQ8,
                context,
                phi3State.workspace.wrapX, // input: hidden state
                phi3State.workspace.wrapHbU, // output: SiLU(gate) ⊙ up [hiddenDim]
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                phi3State.workspace.tempFFN, // RMS scale (precomputed)
                weights.wUpLayered[layerIndex]
                        .asByteArray(), // Q8 combined gate+up [2×hiddenDim × dim]
                config.dim(), // input dim
                config.hiddenDim(), // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual (Q8 dequantization)
        unifiedLayer.task(
                "ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                phi3State.workspace.wrapHbU, // input: FFN intermediate
                phi3State.workspace.wrapX, // output: wrapX += wDown · wrapHbU
                weights.wDownLayered[layerIndex].asByteArray(), // Q8 wDown [dim × hiddenDim]
                config.hiddenDim(), // input dim
                config.dim(), // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(phi3State.workspace.wrapX);
        return unifiedLayer;
    }

    // @formatter:on

    /** Configure data transfers for first and subsequent layers */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.positionHolder,
                    state.workspace.temp,
                    state.workspace.tempFFN); //
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, //
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapXb2, //
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV, //
                    keyCache(),
                    valueCache(), //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb, //
                    phi3State.workspace.wrapHbG,
                    phi3State.workspace.wrapHbU,
                    phi3State.workspace.wrapQkv); //
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
                    state.workspace.wrapXb2, //
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV, //
                    keyCache(),
                    valueCache(), //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb, //
                    state.workspace.positionHolder, // /
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
