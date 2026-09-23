package org.beehive.jllm.backend.tornado.layers.type.q8_0;

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
 * Qwen3Q8_0FFNLayers: Q8_0 transformer-layer TaskGraphs for Qwen3 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Qwen3FP16FFNLayers: - Uses Q8_0-quantized weights (getQuants() and
 * getScales()) - Same Qwen3Kernels for RMSNorm and RoPE - 8-bit integer computations with
 * dequantization - 2x memory compression vs FP16
 *
 * <p>Works directly with Qwen3State to access and mutate Qwen3-specific state fields like tempQcur
 * and tempKcur.
 */
public class Qwen3Q8_0FFNLayers
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
    private final boolean useWarpMatmul = SchedulerDetectionService.isWarpShuffleSupported();

    public Qwen3Q8_0FFNLayers(
            String taskGraphName,
            Qwen3State state,
            Qwen3TornadoWeights weights,
            Qwen3Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen3State = state;
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

        int qkRmsNormGroups = config.numberOfHeads() + config.numberOfKeyValueHeads();
        WorkerGrid qkRmsNormWorker =
                WorkerGridFactory.genericWorker(qkRmsNormGroups * nEmbdHead, nEmbdHead);

        WorkerGrid ropeWorker =
                WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), nEmbdHead);
        // Split-KV attention launches nHeads*nSplits workgroups, then a combine pass over nHeads
        // workgroups.
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(
                        config.numberOfHeads() * attentionSplits, nEmbdHead);
        WorkerGrid attentionCombineWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), nEmbdHead);
        // attn_output_proj worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker =
                WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker =
                WorkerGridFactory.genericWorker(fusedFFNW1W3Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker =
                WorkerGridFactory.genericWorker(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int qDim0 = nEmbdHeadK * config.numberOfHeads();
        int kvDim0 = nEmbdGqa;
        int fusedQKVRows = qDim0 + 2 * kvDim0; // Q rows + K rows + V rows
        int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker =
                WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
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
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", projectionTwoWorker);
        }
        return gridScheduler;
    }

    /** Setup a single transformer layer for Qwen3 with GQA (Q8_0 quantized) */
    // @formatter:off
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
            unifiedLayer.consumeFromDevice(wrapXSrc, qwen3State.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(qwen3State.workspace.wrapX);
        }
        Object[] layerWeights = {
            // Attention weights
            weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS norm weights
            weights.wqLayered[layerIndex].asByteArray(), // Q projection
            weights.wkLayered[layerIndex].asByteArray(), // K projection
            weights.wvLayered[layerIndex].asByteArray(), // V projection
            weights.woLayered[layerIndex].asByteArray(), // Output projection
            // Qwen3-specific Q/K norm weights
            weights.rms_att_KNormLayered[layerIndex].asFloatArray(), // K RMSNorm weights
            weights.rms_att_QNormLayered[layerIndex].asFloatArray(), // Q RMSNorm weights
            // FFN weights
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // FFN RMSNorm weights
            weights.w1Layered[layerIndex].asByteArray(), // FFN gate projection
            weights.w2Layered[layerIndex].asByteArray(), // FFN down projection
            weights.w3Layered[layerIndex].asByteArray() // FFN up projection
        };
        String weightSrc = weightSourceGraphName(layerIndex);
        if (weightSrc != null) {
            unifiedLayer.consumeFromDevice(weightSrc, layerWeights);
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
                    Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0Warp,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapQ, // output: Q vectors
                    qwen3State.workspace.wrapK, // output: K vectors
                    qwen3State.workspace.wrapV, // output: V vectors
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.temp, // RMS scale factor from reduction
                    weights.wqLayered[layerIndex].asByteArray(), // Wq (Q8_0)
                    weights.wkLayered[layerIndex].asByteArray(), // Wk (Q8_0)
                    weights.wvLayered[layerIndex].asByteArray(), // Wv (Q8_0)
                    inputDim, // input dimension
                    qDim, // Q output dimension
                    kvDim, // K/V output dimension (GQA: reduced)
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            unifiedLayer.task(
                    "attn_rms_qkv_projection",
                    Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapQ, // output: Q vectors
                    qwen3State.workspace.wrapK, // output: K vectors
                    qwen3State.workspace.wrapV, // output: V vectors
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.temp, // RMS scale factor from reduction
                    weights.wqLayered[layerIndex].asByteArray(), // Wq (Q8_0)
                    weights.wkLayered[layerIndex].asByteArray(), // Wk (Q8_0)
                    weights.wvLayered[layerIndex].asByteArray(), // Wv (Q8_0)
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

        // Fused RoPE Rotation + KV Cache Write. The cache kernels read FP32 activations, never
        // the weights, so an FP16 cache takes the kernels the FP16 weights use.
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
            // Metal: single-workgroup-per-head online-softmax attention, writing directly to
            // wrapXb.
            // No combine phase needed (TornadoVM fails to JIT the multi-workgroup split-KV kernel
            // here).
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
                        TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16Paged,
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
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0ByteSimd32,
                    context,
                    qwen3State.workspace.wrapXb, // input: attention output
                    qwen3State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asByteArray(), // Wo [dim x qDim]
                    nEmbdHeadK * config.numberOfHeads(), // input dim (qDim)
                    config.dim()); // output dim
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    qwen3State.workspace.wrapXb, // input: attention output
                    qwen3State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asByteArray(), // Wo [dim x qDim]
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
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpQ8_0Warp,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapHb, // output: SiLU(x·W1) ⊙ (x·W3)
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.tempFFN, // RMS scale factor
                    weights.w1Layered[layerIndex].asByteArray(), // W1 (gate) Q8_0
                    weights.w3Layered[layerIndex].asByteArray(), // W3 (up) Q8_0
                    config.dim(), // input dimension
                    config.hiddenDim(), // hidden dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        } else {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpQ8_0,
                    context,
                    qwen3State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen3State.workspace.wrapHb, // output: SiLU(x·W1) ⊙ (x·W3)
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen3State.workspace.tempFFN, // RMS scale factor
                    weights.w1Layered[layerIndex].asByteArray(), // W1 (gate) Q8_0
                    weights.w3Layered[layerIndex].asByteArray(), // W3 (up) Q8_0
                    config.dim(), // input dimension
                    config.hiddenDim(), // hidden dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Down Projection with Residual
        if (useWarpMatmul) {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0ByteSimd32,
                    context,
                    qwen3State.workspace.wrapHb, // input: FFN intermediate
                    qwen3State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                    weights.w2Layered[layerIndex].asByteArray(), // W2 (down)
                    config.hiddenDim(), // input dim
                    config.dim()); // output dim
        } else {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    qwen3State.workspace.wrapHb, // input: FFN intermediate
                    qwen3State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                    weights.w2Layered[layerIndex].asByteArray(), // W2 (down)
                    config.hiddenDim(), // input dim
                    config.dim(), // output dim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        unifiedLayer.persistOnDevice(state.workspace.wrapX, keyCache(), valueCache());

        return unifiedLayer;
    }

    // @formatter:on

    /**
     * Returns the explicit predecessor graph name for consumeFromDevice.
     *
     * <p>The single-token plan relays all persisted buffers (including the KV cache) from a named
     * predecessor graph: the activation graph for layer 0, the previous layer graph otherwise. The
     * no-arg consume form does not propagate the persisted KV cache in interpreter mode, so it is
     * re-allocated every decode token and exhausts the device memory pool (OOM) on long
     * generations. Decode subclasses override this.
     */
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "activationUpdate" : "layer_" + (layerIndex - 1);
    }

    /** Configure data transfers for first and subsequent layers */
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

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.positionHolder,
                    qwen3State.workspace.temp,
                    qwen3State.workspace.tempFFN);
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen3State.workspace.tempQcur,
                    qwen3State.workspace.tempKcur);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
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
            // Subsequent layers: consume from the previous layer graph BY NAME. The no-arg
            // consume form does not propagate the persisted KV cache in interpreter mode, so
            // it would be re-allocated every decode token and exhaust the memory pool (OOM).
            String pred = "layer_" + (layerIndex - 1);
            unifiedLayer.consumeFromDevice(
                    pred,
                    context,
                    qwen3State.workspace.wrapXb,
                    qwen3State.workspace.wrapXb2,
                    qwen3State.workspace.wrapQ,
                    qwen3State.workspace.wrapK,
                    qwen3State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    qwen3State.workspace.wrapAtt,
                    qwen3State.workspace.wrapHb,
                    qwen3State.workspace.positionHolder);
            unifiedLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);

            unifiedLayer.consumeFromDevice(
                    pred, qwen3State.workspace.tempQcur, qwen3State.workspace.tempKcur); //
            unifiedLayer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
        }
        return unifiedLayer;
    }

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
}
