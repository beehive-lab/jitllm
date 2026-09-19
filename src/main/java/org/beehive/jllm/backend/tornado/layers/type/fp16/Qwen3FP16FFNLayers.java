package org.beehive.jllm.backend.tornado.layers.type.fp16;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Qwen3State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.jllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Qwen3FP16FFNLayers: FP16 transformer-layer TaskGraphs for Qwen3 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Llama: - Supports GQA with separate KV heads (nHeadKv) - Uses
 * Qwen3Kernels for RMSNorm with parallel offset - Custom RoPE rotation for Qwen3 - Different
 * attention computation due to GQA structure
 *
 * <p>Works directly with Qwen3State to access and mutate Qwen3-specific state fields like tempQcur
 * and tempKcur.
 */
public class Qwen3FP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen3TornadoWeights, Qwen3Configuration> {

    // Typed reference to Qwen3-specific state
    protected final Qwen3State qwen3State;
    // Qwen3-specific GQA parameters
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdVGqa;
    private final int nEmbdHead;
    private final int nEmbdGqa;
    private final int gqa;
    // Decode attention is split-KV (flash-decoding) on every backend except Metal, where TornadoVM
    // fails to JIT the multi-workgroup split-KV kernel; Metal falls back to the
    // single-workgroup-per-head
    // online-softmax kernel instead (see SchedulerDetectionService.isMetalBackend). Splits per
    // head: see State.SPLIT_KV.
    private final boolean isMetalBackend = SchedulerDetectionService.isMetalBackend();
    private final int attentionSplits = isMetalBackend ? 1 : State.SPLIT_KV;
    // GEMV reduction strategy: 32-lane warp-shuffle on CUDA, shared-memory trees elsewhere.
    // Warp is
    // faster but the OpenCL backend miscompiles simdShuffleDown, so it is auto-selected by backend.
    //
    // Metal is included via SUBGROUP_SHUFFLE_32 — a correctness fix, not an optimisation. The
    // shared-memory-tree branch below produces numerically wrong output on Metal —
    // fluent-looking token salad at a normal throughput, which is why it must be gated by
    // capability and not left to a backend-name test.
    // Three of the four kernels this selects are the same ones LlamaFP16FFNLayers already runs
    // correctly on Metal under its own isSubgroupShuffle32Supported() gate
    // (matrixVectorGenericWithResidualSimd32 twice, fusedRmsNormFFNGateUpWarp); the fourth,
    // fusedRmsNormQKVMatmulWarp, is Qwen3's own and has the same 32-lane butterfly shape as the
    // verified fusedQKVMatmulXSimd32. All four run on the existing 32-wide worker grid
    // (LOCAL_WORK_GROUP_SIZE_ALLOC == 32), so no grid-scheduler change is needed.
    //
    // SUBGROUP_SHUFFLE_32 is granted to Metal only (TornadoDevices.capabilitiesOf), so CUDA and
    // OpenCL selection is byte-for-byte unchanged.
    private final boolean useWarpMatmul =
            SchedulerDetectionService.isWarpShuffleSupported()
                    || SchedulerDetectionService.isSubgroupShuffle32Supported();

    public Qwen3FP16FFNLayers(
            String taskGraphName,
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen3State = state;

        // Initialize GQA parameters from Qwen3Config
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdVGqa = nEmbdHeadV * nHeadKv;
        this.nEmbdHead = nEmbdHeadV;
        this.nEmbdGqa = nEmbdVGqa;
        this.gqa = config.numberOfHeads() / config.numberOfKeyValueHeads();
        setupFFNLayers();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker =
                WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);
        WorkerGrid ropeWorker =
                WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), nEmbdHead);
        // Split-KV attention launches nHeads*nSplits workgroups (one per head-split) followed by a
        // combine pass over nHeads workgroups.
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(
                        config.numberOfHeads() * attentionSplits, nEmbdHead);
        WorkerGrid attentionCombineWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), nEmbdHead);
        // attn_output_proj worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker =
                WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);
        // FFN workers
        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker =
                WorkerGridFactory.genericWorker(fusedFFNW1W3Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker =
                WorkerGridFactory.genericWorker(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
        int qkRmsNormGroups = config.numberOfHeads() + config.numberOfKeyValueHeads();
        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(qkRmsNormGroups * nEmbdHead, nEmbdHead);

        int qDim0 = nEmbdHeadK * config.numberOfHeads();
        int kvDim0 = nEmbdGqa;
        int fusedQKVRows = qDim0 + 2 * kvDim0; // Q rows + K rows + V rows
        int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker =
                WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Map workers to tasks for each layer (in task execution order)
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // === Attention Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQKVWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".qk_rmsnorm", qkRmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            gridScheduler.addWorkerGrid(
                    "layer_" + i + ".attention",
                    isMetalBackend ? attentionCombineWorker : parallelAttentionWorker);
            if (!isMetalBackend) {
                gridScheduler.addWorkerGrid(
                        "layer_" + i + ".attention_combine", attentionCombineWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", matmul1Worker);
            // === FFN Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", projectionTwoWorker);
        }
        return gridScheduler;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Qwen3FP16FFNLayers)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (scale factor for
     * RMSNorm) └────────┬────────┘ │ ▼ ┌─────────────────────────┐ │ attn_rms_qkv_projection │──▶
     * wrapQ, wrapK, wrapV (FP32) └───────────┬─────────────┘ (fused: RMS apply + Q/K/V matmuls) │ ▼
     * ┌─────────────┐ │ qk_rmsnorm │──▶ wrapQ, wrapK normalized in-place └──────┬──────┘ (fused: Q
     * + K RMSNorm reduction + apply) │ ▼ ┌───────────────────┐
     * ┌─────────────────────────────────────┐ │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache,
     * ValueCache │ └─────────┬─────────┘ └─────────────────────────────────────┘ │ (fused: RoPE
     * rotation + cache write) ▼ ┌───────────┐ │ attention │──▶ wrapXb (attention output)
     * └─────┬─────┘ │ ▼ ┌──────────────────┐ │ attn_output_proj │──▶ wrapX += Wo · wrapXb (residual
     * connection) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (scale factor) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌──────────────────┐ │ ffn_rms_finalize │──▶ tempFFN (final
     * scale) └────────┬─────────┘ │ ▼ ┌─────────────────┐ │ rms_ffn_gate_up │──▶ wrapHb =
     * SiLU(RMSNorm(x)·W1) ⊙ (RMSNorm(x)·W3) └────────┬────────┘ (fused: RMS apply + W1/W3 matmuls +
     * SiLU + GLU) │ ▼ ┌──────────────┐ │ ffn_down_proj│──▶ wrapX += W2 · wrapHb (residual
     * connection) └──────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 9 tasks (NVIDIA) / 10 tasks (non-NVIDIA)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points (vs baseline 18 tasks): • attn_rms_qkv_projection: Fused RMS apply +
     * Q/K/V matmuls (4→1 kernel) • qk_rmsnorm: Fused Q + K RMSNorm (4→1 kernel) •
     * rope_and_kv_cache: Fused RoPE rotation + cache write (2→1 kernel) • rms_ffn_gate_up: Fused
     * RMS apply + W1/W3 matmuls + SiLU + GLU (4→1 kernel)
     *
     * <p>Qwen3-Specific: • GQA: nHeads (Q) != nHeadKv (K/V), with gqa = nHeads / nHeadKv • Q/K
     * RMSNorm: Additional normalization after QKV projection (qk_rmsnorm) • RoPE theta: 1,000,000
     * (vs Llama's 10,000 or 50,000)
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;

        // === Dimension Parameters ===
        int qDim = nEmbdHeadK * config.numberOfHeads(); // Q output size (full heads)
        int kvDim = nEmbdGqa; // K/V output size (reduced for GQA)
        int inputDim = config.dim(); // Model dimension

        var unifiedLayer = new TaskGraph(taskGraphName);

        // === Data Setup ===
        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
            // Attention weights
            weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS norm weights
            weights.wqLayered[layerIndex].asHalfFloatArray(), // Q projection
            weights.wkLayered[layerIndex].asHalfFloatArray(), // K projection
            weights.wvLayered[layerIndex].asHalfFloatArray(), // V projection
            weights.woLayered[layerIndex].asHalfFloatArray(), // Output projection
            // Qwen3-specific Q/K norm weights
            weights.rms_att_KNormLayered[layerIndex].asFloatArray(), // K RMSNorm weights
            weights.rms_att_QNormLayered[layerIndex].asFloatArray(), // Q RMSNorm weights
            // FFN weights
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // FFN RMS norm weights
            weights.w1Layered[layerIndex].asHalfFloatArray(), // FFN gate
            weights.w2Layered[layerIndex].asHalfFloatArray(), // FFN down
            weights.w3Layered[layerIndex].asHalfFloatArray() // FFN up
        };
        String weightSrc = weightSourceGraphName(layerIndex);
        if (weightSrc != null) {
            // consumeFromDevice only aliases a buffer the producer graph really has. TornadoVM
            // builds a graph out of its tasks' argument lists, so a weight the producer declares
            // in transferToDevice but passes to no task is never allocated or uploaded there, and
            // consuming it here yields a buffer nothing ever wrote — silently, with no diagnostic.
            // Anything the producer no longer reads is therefore uploaded by this graph instead.
            Object[] notProvided = weightsNotProvidedBySource(layerIndex);
            if (notProvided.length == 0) {
                unifiedLayer.consumeFromDevice(weightSrc, layerWeights);
            } else {
                List<Object> consumed = new ArrayList<>(layerWeights.length);
                for (Object w : layerWeights) {
                    boolean skip = false;
                    for (Object n : notProvided) {
                        if (n == w) {
                            skip = true;
                            break;
                        }
                    }
                    if (!skip) {
                        consumed.add(w);
                    }
                }
                unifiedLayer.consumeFromDevice(weightSrc, consumed.toArray());
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, notProvided);
            }
        } else {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights);
        }
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                qwen3State.workspace.temp, // output: scale factor
                qwen3State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                qwen3State.localSize); // local memory size

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        // Fused RMS Apply + QKV Projection
        if (useWarpMatmul) {
            unifiedLayer.task(
                    "attn_rms_qkv_projection",
                    Qwen3Kernels::fusedRmsNormQKVMatmulWarp,
                    context,
                    qwen3State.workspace.wrapX,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    qwen3State.workspace.temp,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    inputDim,
                    qDim,
                    kvDim,
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            unifiedLayer.task(
                    "attn_rms_qkv_projection",
                    Qwen3Kernels::fusedRmsNormQKVMatmul,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapQ, // output: Q vectors
                    qwen3State.workspace.wrapK, // output: K vectors
                    qwen3State.workspace.wrapV, // output: V vectors
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.temp, // RMS scale factor from reduction
                    weights.wqLayered[layerIndex].asHalfFloatArray(), // Wq [qDim x inputDim]
                    weights.wkLayered[layerIndex].asHalfFloatArray(), // Wk [kvDim x inputDim]
                    weights.wvLayered[layerIndex].asHalfFloatArray(), // Wv [kvDim x inputDim]
                    inputDim, // input dimension
                    qDim, // Q output dimension
                    kvDim, // K/V output dimension (GQA: reduced)
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Fused Q/K RMSNorm (Qwen3-specific)
        unifiedLayer.task(
                "qk_rmsnorm",
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                qwen3State.workspace.wrapQ, // Q vectors (in/out)
                qwen3State.workspace.wrapK, // K vectors (in/out)
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(), // Q norm weights
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(), // K norm weights
                config.numberOfHeads(), // nHeads (Q heads)
                config.numberOfKeyValueHeads(), // nHeadKv (K/V heads, GQA)
                nEmbdHead, // head dimension
                nEmbdHead, // local memory size
                config.rmsNormEps()); // epsilon

        // Fused RoPE Rotation + KV Cache Write
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    Qwen3PagedKvKernels::ropeRotationWithCacheCopyFP16Paged,
                    context,
                    qwen3State.workspace.positionHolder, // current position
                    qwen3State.workspace.wrapQ, // Q vectors (in/out, rotated)
                    qwen3State.workspace.wrapK, // K vectors (in/out, rotated)
                    qwen3State.workspace.wrapV, // V vectors (in only)
                    qwen3State.workspace.wrapKeyCacheFP16, // key cache (out, FP16)
                    qwen3State.workspace.wrapValueCacheFP16, // value cache (out, FP16)
                    config.ropeTheta(),
                    config.numberOfKeyValueHeads(), // nHeadKv
                    nEmbdHead, // head dimension
                    nEmbdGqa, // kvDim
                    layerIndex, // layer index for cache offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        } else {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    Qwen3PagedKvKernels::ropeRotationWithCacheCopyPaged,
                    context,
                    qwen3State.workspace.positionHolder, // current position
                    qwen3State.workspace.wrapQ, // Q vectors (in/out, rotated)
                    qwen3State.workspace.wrapK, // K vectors (in/out, rotated)
                    qwen3State.workspace.wrapV, // V vectors (in only)
                    qwen3State.workspace.wrapKeyCache, // key cache (out)
                    qwen3State.workspace.wrapValueCache, // value cache (out)
                    config.ropeTheta(),
                    config.numberOfKeyValueHeads(), // nHeadKv
                    nEmbdHead, // head dimension
                    nEmbdGqa, // kvDim
                    layerIndex, // layer index for cache offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        }

        if (isMetalBackend) {
            // Metal cannot JIT the multi-workgroup split-KV kernel, so use the
            // single-workgroup-per-head online-softmax kernel, which writes wrapXb directly and
            // needs no combine phase. It reads the FP32 KV cache, so it is not compatible with
            // the FP16 KV cache; useFp16KVCache() below is therefore only consulted off Metal.
            unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    qwen3State.workspace.wrapQ, // query vectors
                    qwen3State.workspace.wrapKeyCache, // key cache
                    qwen3State.workspace.wrapValueCache, // value cache
                    qwen3State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    nEmbdHead, // headSize
                    nEmbdGqa, // kvDim
                    gqa, // kvMul (nHeads / nHeadKv)
                    qwen3State.workspace.positionHolder, // position
                    layerIndex, // layer index
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // context length
        } else {
            // Split-KV (flash-decoding) attention.
            // Phase 1: split each head's KV range across attentionSplits workgroups; partials ->
            // wrapAttSplit.
            if (useFp16KVCache()) {
                unifiedLayer.task(
                        "attention",
                        packedHalf2Attention
                                ? TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PackedPaged
                                : TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16Paged,
                        context,
                        qwen3State.workspace.wrapQ, // query vectors
                        qwen3State.workspace.wrapKeyCacheFP16, // key cache (FP16)
                        qwen3State.workspace.wrapValueCacheFP16, // value cache (FP16)
                        state.workspace
                                .wrapAttSplit, // scratch: per-head split partials (compact layout)
                        config.numberOfHeads(), // nHeads
                        nEmbdHead, // headSize
                        nEmbdGqa, // kvDim
                        gqa, // kvMul (nHeads / nHeadKv)
                        qwen3State.workspace.positionHolder, // position
                        layerIndex, // layer index
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride, // context length
                        attentionSplits); // number of KV splits per head
            } else {
                unifiedLayer.task(
                        "attention",
                        TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVPaged,
                        context,
                        qwen3State.workspace.wrapQ, // query vectors
                        qwen3State.workspace.wrapKeyCache, // key cache
                        qwen3State.workspace.wrapValueCache, // value cache
                        state.workspace
                                .wrapAttSplit, // scratch: per-head split partials (compact layout)
                        config.numberOfHeads(), // nHeads
                        nEmbdHead, // headSize
                        nEmbdGqa, // kvDim
                        gqa, // kvMul (nHeads / nHeadKv)
                        qwen3State.workspace.positionHolder, // position
                        layerIndex, // layer index
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride, // context length
                        attentionSplits); // number of KV splits per head
            }
            // Phase 2: combine the per-head split partials into the final attention output ->
            // wrapXb.
            unifiedLayer.task(
                    "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    state.workspace
                            .wrapAttSplit, // scratch: per-head split partials (compact layout)
                    qwen3State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    nEmbdHead, // headSize
                    attentionSplits); // number of KV splits per head
        }

        // Output Projection with Residual
        if (useWarpMatmul) {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapX,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    nEmbdHeadK * config.numberOfHeads(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    qwen3State.workspace.wrapXb, // input: attention output
                    qwen3State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asHalfFloatArray(), // Wo [dim x qDim]
                    nEmbdHeadK * config.numberOfHeads(), // input dim (qDim)
                    config.dim(), // output dim
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
                qwen3State.workspace.tempFFN, // output: scale factor
                qwen3State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                qwen3State.localSize); // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen3State.workspace.tempFFN, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        // Fused RMS Apply + Gate/Up Projection + SiLU + GLU
        if (useWarpMatmul) {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpWarp,
                    context,
                    qwen3State.workspace.wrapX,
                    qwen3State.workspace.wrapHb,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    qwen3State.workspace.tempFFN,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.hiddenDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUp,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapHb, // output: SiLU(x·W1) ⊙ (x·W3)
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.tempFFN, // RMS scale factor
                    weights.w1Layered[layerIndex].asHalfFloatArray(), // W1 (gate)
                    weights.w3Layered[layerIndex].asHalfFloatArray(), // W3 (up)
                    config.dim(), // input dimension
                    config.hiddenDim(), // hidden dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Down Projection with Residual
        if (useWarpMatmul) {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    qwen3State.workspace.wrapHb,
                    qwen3State.workspace.wrapX,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    config.hiddenDim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    qwen3State.workspace.wrapHb, // input: FFN intermediate
                    qwen3State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                    weights.w2Layered[layerIndex].asHalfFloatArray(), // W2 (down)
                    config.hiddenDim(), // input dim
                    config.dim(), // output dim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }
        if (useFp16KVCache()) {
            unifiedLayer.persistOnDevice(
                    qwen3State.workspace.wrapX,
                    qwen3State.workspace.wrapKeyCacheFP16,
                    qwen3State.workspace.wrapValueCacheFP16);
        } else {
            unifiedLayer.persistOnDevice(
                    qwen3State.workspace.wrapX,
                    qwen3State.workspace.wrapKeyCache,
                    qwen3State.workspace.wrapValueCache);
        }

        return unifiedLayer;
    }

    // @formatter:on

    /**
     * Returns the explicit predecessor graph name for consumeFromDevice.
     *
     * <p>The single-token plan receives {@code wrapX} (and relays all persisted buffers, including
     * the KV cache) from a named predecessor graph: the activation graph for layer 0, the previous
     * layer graph otherwise. The no-arg consume form looks up the <em>current</em> graph's name as
     * the source key, which never matches in interpreter mode, so the persisted KV-cache buffer is
     * not propagated and gets re-allocated every decode token — exhausting the device-memory pool
     * (OOM) on long generations. Decode subclasses override this with their own predecessor names.
     */
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "activationUpdate" : "layer_" + (layerIndex - 1);
    }

    /** Configure data transfers for first and subsequent layers */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        Object keyCache =
                useFp16KVCache()
                        ? qwen3State.workspace.wrapKeyCacheFP16
                        : qwen3State.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? qwen3State.workspace.wrapValueCacheFP16
                        : qwen3State.workspace.wrapValueCache;
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, qwen3State.workspace.positionHolder);
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, //
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2, //
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV, //
                    keyCache,
                    valueCache, //
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
        } else {
            // Subsequent layers: consume from the previous layer graph BY NAME.
            // The no-arg consumeFromDevice form uses the current graph's own name as the
            // source key, which never matches the predecessor in interpreter mode, so the
            // persisted KV cache is not propagated and is re-allocated every token (OOM).
            String pred = "layer_" + (layerIndex - 1);
            unifiedLayer.consumeFromDevice(
                    pred,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2, //
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK, //
                    qwen3State.workspace.wrapV,
                    keyCache, //
                    valueCache,
                    qwen3State.workspace.wrapAtt, //
                    qwen3State.workspace.wrapHb,
                    qwen3State.workspace.positionHolder); //
            unifiedLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
            unifiedLayer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
        }
        return unifiedLayer;
    }

    // @formatter:on

    /**
     * The graph that has already uploaded this layer's weights, or {@code null} to upload them
     * here.
     *
     * <p>A weight array bound with {@code transferToDevice} in two task graphs of one execution
     * plan gets a device buffer in each, so the pool has to hold the whole model twice. See {@code
     * LlamaFP16FFNLayers.weightSourceGraphName} for the full note.
     */
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }

    /**
     * Weights that {@link #weightSourceGraphName} names as the producer but that no task in that
     * graph actually reads, so it never uploads them and this graph has to. Empty by default, which
     * is the case whenever the producer computes with every weight it declares.
     */
    protected Object[] weightsNotProvidedBySource(int layerIndex) {
        return NO_WEIGHTS;
    }

    private static final Object[] NO_WEIGHTS = new Object[0];
}
