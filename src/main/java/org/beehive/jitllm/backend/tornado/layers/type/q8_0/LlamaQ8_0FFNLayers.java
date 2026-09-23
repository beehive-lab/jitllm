package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LlamaQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, LlamaConfiguration> {

    public LlamaQ8_0FFNLayers(
            String taskGraphName,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (LlamaQ8FFNLayers)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (partial sums)
     * └────────┬────────┘ │ ▼ (optional: NON_NVIDIA only) ┌──────────────────┐ │
     * attn_rms_finalize│──▶ temp (final scale) └────────┬─────────┘ │ ▼ ┌────────────────┐ │
     * attn_rms_apply │──▶ wrapXb (normalized, FP32) └───────┬────────┘ │ ▼ ┌────────────────┐
     * ┌─────────────────────────────┐ │ qkv_projection │──────▶│ wrapQ, wrapK, wrapV (FP32) │
     * └───────┬────────┘ └─────────────────────────────┘ │ ▼ ┌───────────────────┐
     * ┌─────────────────────────────────────┐ │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache,
     * ValueCache │ └─────────┬─────────┘ └─────────────────────────────────────┘ │ ▼ ┌───────────┐
     * │ attention │──▶ wrapXb (attention output) └─────┬─────┘ │ ▼ ┌──────────────────┐ │
     * attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection) └────────┬─────────┘ │
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════ │ ▼
     * ┌────────────────┐ │ ffn_rms_reduce │──▶ tempFFN (partial sums) └───────┬────────┘ │ ▼
     * (optional: NON_NVIDIA only) ┌─────────────────┐ │ ffn_rms_finalize│──▶ tempFFN (final scale)
     * └────────┬────────┘ │ ▼ ┌─────────────────┐ │ rms_ffn_gate_up │──▶ wrapHb =
     * SiLU(RMSNorm(x)·W1) ⊙ (RMSNorm(x)·W3) └────────┬────────┘ (fully fused: RMS reduce/apply +
     * W1/W3 matmuls + SiLU + GLU) │ ▼ ┌──────────────┐ │ ffn_down_proj│──▶ wrapX += W2 · wrapHb
     * (residual connection) └──────┬───────┘ │ ▼ wrapX (FP32) ──▶ [next layer or logits]
     *
     * <p>══════════════════════════════════════════════════════════════════════════════
     *
     * <p>Task Count: 9 tasks (7 if NVIDIA, skipping rms_finalize steps)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points: • qkv_projection: Fused Q/K/V matmuls with Q8 dequantization (3→1
     * kernel) • rope_and_kv_cache: Fused RoPE rotation + cache write (2→1 kernel) •
     * rms_ffn_gate_up: Fully fused RMS norm + W1/W3 matmuls + SiLU + GLU (5→1 kernel)
     *
     * <p>Quantization: Q8_0 format (8-bit weights with block-wise scaling)
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
            // Copy-in weights per layer for batched-layered layout (Q8 format)
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
        // Precomputed RoPE tables: the frequencies come from the model's own rope_theta (and
        // any Llama 3.1 frequency scaling) instead of a constant baked into the kernel.
        // The paged twin differs in the KV index only: the block-table walk replaces
        // layer*contextLength*kvDim + pos*kvDim.
        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        // Attention
        configureAttention(unifiedLayer, layerIndex);

        // Output Projection (Wo) with residual (Q8 dequantization)
        unifiedLayer.task(
                "attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

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
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                state.workspace.wrapHb,
                state.workspace.wrapX,
                weights.w2Layered[layerIndex].asByteArray(),
                config.hiddenDim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Keep activation X on device for next layer
        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        Object keyCache = keyCache();
        Object valueCache = valueCache();
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
                    keyCache,
                    valueCache, //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray()); //
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
                    keyCache,
                    valueCache, //
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb, //
                    state.workspace.positionHolder //
                    ,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            unifiedLayer.consumeFromDevice(state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
    }

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

    /**
     * RoPE on Q and K, and K and V appended to the cache. Shared by every weight type of this
     * family, so a subclass cannot write one cache representation while attention reads another.
     */
    protected void ropeAndKeyValueCache(TaskGraph unifiedLayer, int layerIndex) {
        // The cache representation is independent of the weights: Q, K and V are FP32
        // activations here, so the FP16-cache kernels are the ones the FP16 weights use.
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedFP16Paged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCacheFP16, // Key cache (out, FP16)
                    state.workspace.wrapValueCacheFP16, // Value cache (out, FP16)
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedPaged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ, // Q (in/out)
                    state.workspace.wrapK, // K (in/out)
                    state.workspace.wrapV, // V (in only)
                    state.workspace.wrapKeyCache, // Key cache (out)
                    state.workspace.wrapValueCache, // Value cache (out)
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
    }

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

    /** Attention is dtype-independent — a Q4_0 sibling reuses this unchanged. */
    protected TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (useFp16KVCache()) {
            // Flash attention over the half-precision cache, FP32 accumulation.
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    state.workspace.wrapQ,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
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
        }
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
