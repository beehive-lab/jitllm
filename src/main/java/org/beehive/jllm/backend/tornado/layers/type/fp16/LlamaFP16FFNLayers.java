package org.beehive.jllm.backend.tornado.layers.type.fp16;

import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LlamaFP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, LlamaConfiguration> {
    /**
     * Whether this graph uses split-KV (flash-decoding) attention, resolved once from the session's
     * policy.
     *
     * <p>It was a {@code static final} read from {@code jllm.attention.splitKv} at class
     * initialization. <b>Selection is policy; the partition count is capacity</b> — the count sizes
     * {@code wrapAttSplit}, so it stays where the array is allocated, and the working value is
     * checked against it rather than assumed equal.
     */
    private final boolean splitKvAttention;

    /** Working partitions per head, never more than the scratch array was sized for. */
    private final int attentionSplits;

    /**
     * QKV-projection and residual-MatVec reduction strategy: 32-lane subgroup shuffle where
     * verified correct (Metal), shared-memory reduction elsewhere — the same
     * NVIDIA/non-NVIDIA-style capability selection {@code Qwen3FP16FFNLayers.useWarpMatmul} already
     * uses for its own GEMV kernel choice, on its own capability rather than reusing that one
     * (Metal parity task 5→6 follow-up: {@code DeviceCapability.SUBGROUP_SHUFFLE_32} is verified
     * for the QKV projection's and {@code matrixVectorGenericWithResidual}'s reduction shape
     * specifically — both a five-step butterfly over one 32-lane workgroup per output row — not for
     * warp shuffle in general). Gates {@code qkv_projection}, {@code attn_output_proj} and {@code
     * ffn_down_proj}.
     */
    private final boolean useSimd32Reduction =
            SchedulerDetectionService.isSubgroupShuffle32Supported();

    /**
     * Whether the packed FP16 multiply is accurate enough here to keep CPU parity. Where it is not,
     * the QKV projection widens each pair before multiplying instead.
     */
    private final boolean packedHalf2Math = SchedulerDetectionService.isPackedHalf2MathSupported();

    private final LlamaState llamaState;

    public LlamaFP16FFNLayers(
            String taskGraph,
            State state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
        this.llamaState = (LlamaState) state;
        var partitions = state.executionPolicy().splitKvPartitions();
        this.splitKvAttention = partitions.isPresent();
        int working = partitions.orElse(State.SPLIT_KV);
        if (working > State.SPLIT_KV) {
            // The scratch array was allocated for State.SPLIT_KV partitions. A working value above
            // it would index past the end — silently, since the kernels take the count as an
            // argument. Sizing maxima belong to whoever allocates; the working value must fit
            // inside one.
            throw new IllegalArgumentException(
                    "split-KV working partitions "
                            + working
                            + " exceed the "
                            + State.SPLIT_KV
                            + " the attention scratch was sized for;"
                            + " raise jllm.attention.splitKv.count, which is the capacity");
        }
        this.attentionSplits = working;
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

        int attentionGroups =
                splitKvAttentionEnabled()
                        ? config.numberOfHeads() * attentionSplits
                        : config.numberOfHeads();
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(attentionGroups, config.headSize());
        WorkerGrid attentionCombineWorker =
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
            if (splitKvAttentionEnabled()) {
                tornadoForwardScheduler.addWorkerGrid(
                        "layer_" + i + ".attention_combine", attentionCombineWorker);
            }
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
        // consumeFromDevice for wrapX: the no-arg form uses the current graph's own name as the
        // source key, which works in CUDA-graph mode (pointers are frozen) but fails in interpreter
        // mode (updatePersistedObjectState looks up the predecessor's name, not the current name).
        // Subclasses that receive wrapX across a graph boundary override predecessorGraphName() to
        // return the correct predecessor graph name so the XPUBuffer is propagated in both modes.
        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asHalfFloatArray(),
            weights.wkLayered[layerIndex].asHalfFloatArray(),
            weights.wvLayered[layerIndex].asHalfFloatArray(),
            weights.woLayered[layerIndex].asHalfFloatArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asHalfFloatArray(),
            weights.w2Layered[layerIndex].asHalfFloatArray(),
            weights.w3Layered[layerIndex].asHalfFloatArray()
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
                "attn_rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantize,
                context,
                state.workspace.wrapXbFP16,
                state.workspace.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.temp);

        // QKV Projection (fused). Same logical operation, same task name, same Q/K/V output
        // contract either way — only the reduction kernel differs, by device capability (Metal
        // parity task 5->6 follow-up). Both variants share the same 32-wide worker grid
        // (LOCAL_WORK_GROUP_SIZE_ALLOC == 32), so no grid-scheduler change is needed here.
        if (useSimd32Reduction) {
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

        // RoPE + KV Cache. The paged twins differ from the legacy kernels in the KV index only:
        // the block-table walk replaces layer*contextLength*kvDim + pos*kvDim.
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
        // Attention
        configureAttention(unifiedLayer, layerIndex);
        // Output Projection (Wo) with residual. Same logical operation, same task name, same
        // output contract either way; only the reduction kernel differs, by device capability.
        if (useSimd32Reduction) {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    config.dim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

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

        // Same logical operation, same task name, same output contract either way; only the
        // reduction kernel differs, by device capability (fusedRmsNormFFNGateUpWarp is the same
        // kernel Qwen3FP16FFNLayers already selects on CUDA via isWarpShuffleSupported() -
        // reused here, not duplicated, gated on the separately-verified Metal capability so
        // CUDA and OpenCL selection for Llama is unchanged).
        if (useSimd32Reduction) {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpWarp,
                    context,
                    state.workspace.wrapX,
                    state.workspace.wrapHb,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN,
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
                    state.workspace.wrapX, // raw input (FP32)
                    state.workspace.wrapHb, // output
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    state.workspace.tempFFN, // RMS scale factor
                    weights.w1Layered[layerIndex].asHalfFloatArray(), // W1
                    weights.w3Layered[layerIndex].asHalfFloatArray(), // W3
                    config.dim(), // input dimension
                    config.hiddenDim(), // output dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Down projection (W2) with residual. Same logical operation, same task name, same
        // output contract either way; only the reduction kernel differs, by device capability.
        if (useSimd32Reduction) {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualSimd32,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    config.hiddenDim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    config.hiddenDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        if (useFp16KVCache()) {
            unifiedLayer.persistOnDevice(
                    state.workspace.wrapX,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16);
        } else {
            unifiedLayer.persistOnDevice(
                    state.workspace.wrapX,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache);
        }

        // Diagnostic, default off: pulls the per-layer intermediates back so an FP16
        // divergence can be localized. The shipped task graph is unchanged.
        if (Boolean.getBoolean("jllm.diag.transfers")
                && layerIndex == Integer.getInteger("jllm.diag.layer", 0)) {
            unifiedLayer.transferToHost(
                    uk.ac.manchester.tornado.api.enums.DataTransferMode.EVERY_EXECUTION,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapX,
                    state.workspace.wrapQ,
                    state.workspace.wrapXbFP16,
                    state.workspace.temp);
        }

        return unifiedLayer;
    }

    /**
     * The graph that has already uploaded this layer's weights, or {@code null} to upload them
     * here.
     *
     * <p>A weight array bound with {@code transferToDevice} in two task graphs of one execution
     * plan gets a device buffer in each, so the pool has to hold the whole model twice. That is
     * what put batched prefill/decode at roughly 2x the weights on the device and left 8B FP16
     * models unable to allocate on a 24GB card while the same model ran in standard mode. The
     * batched decode graphs consume the copy their layer's prefill graph already uploaded, the way
     * they already consume the KV cache and the block table.
     *
     * <p>Only override this where the named graph is guaranteed to be in the same execution plan
     * and to have run first. Decode-only plans have no prefill graph to consume from and must keep
     * uploading.
     */
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }

    /**
     * Returns the name of the predecessor TaskGraph from which {@code wrapX} should be consumed, or
     * {@code null} to fall back to the no-arg form (source key = own graph name).
     *
     * <p>The no-arg form is safe in CUDA-graph mode (device pointers are frozen at capture time)
     * but fails in interpreter mode: {@code updatePersistedObjectState} looks up the predecessor's
     * graph name, not the current graph's name, so the XPUBuffer is never propagated and {@code
     * executeAlloc} NPEs on a null buffer.
     *
     * <p>Override in subclasses that receive {@code wrapX} from a named predecessor graph:
     *
     * <ul>
     *   <li>layer 0: return the activation graph name (e.g. {@code "activationUpdate"})
     *   <li>layer k &gt; 0: return {@code "layer_" + (k-1)}
     * </ul>
     */
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "activationUpdate" : "layer_" + (layerIndex - 1);
    }

    protected boolean splitKvAttentionEnabled() {
        return splitKvAttention && schedulerType == SchedulerType.NVIDIA;
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

    /**
     * Whether this graph addresses KV through the block table.
     *
     * <p>Follows the state, because the state is what chose the cache layout. Off puts the graph
     * back on the legacy contiguous kernels, which are still bound by every other family.
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        Object keyCache =
                useFp16KVCache() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
        Object valueCache =
                useFp16KVCache()
                        ? state.workspace.wrapValueCacheFP16
                        : state.workspace.wrapValueCache;
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
                    keyCache,
                    valueCache,
                    // Attention & FFN buffers
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    state.workspace.wrapXbFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            if (splitKvAttentionEnabled()) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapAttSplit);
            }
            // The block table is read by every KV kernel. A buffer a kernel reads and the graph
            // does not carry is the silent-wrong-answer bug in this codebase, so it is bound
            // explicitly. EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the
            // table, and a stale block index is still a valid index, so a table uploaded once
            // leaves the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: consume from the previous layer graph by name.
            // The no-arg consumeFromDevice form uses the current graph's own name as source key,
            // which never matches the predecessor in interpreter mode (no CUDA graphs).
            String pred = "layer_" + (layerIndex - 1);
            unifiedLayer.consumeFromDevice(
                    pred,
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
                    keyCache,
                    valueCache,
                    // Attention & FFN buffers
                    state.workspace.wrapAtt,
                    state.workspace.wrapHb,
                    // Position & misc
                    state.workspace.positionHolder,
                    state.workspace.wrapXbFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            if (splitKvAttentionEnabled()) {
                unifiedLayer.consumeFromDevice(pred, state.workspace.wrapAttSplit);
            }
            unifiedLayer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        }
        return unifiedLayer;
    }

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (splitKvAttentionEnabled()) {
            if (useFp16KVCache()) {
                unifiedLayer.task(
                        "attention",
                        packedHalf2Attention
                                ? TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16PackedPaged
                                : TransformerPagedKvKernels
                                        ::processHeadsFlashAttentionSplitKVFP16Paged,
                        context,
                        state.workspace.wrapQ,
                        state.workspace.wrapKeyCacheFP16,
                        state.workspace.wrapValueCacheFP16,
                        state.workspace.wrapAttSplit,
                        config.numberOfHeads(),
                        config.headSize(),
                        config.kvDim(),
                        config.kvMul(),
                        state.workspace.positionHolder,
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        attentionSplits);
            } else {
                unifiedLayer.task(
                        "attention",
                        TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVPaged,
                        context,
                        state.workspace.wrapQ,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache,
                        state.workspace.wrapAttSplit,
                        config.numberOfHeads(),
                        config.headSize(),
                        config.kvDim(),
                        config.kvMul(),
                        state.workspace.positionHolder,
                        layerIndex,
                        state.workspace.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride,
                        attentionSplits);
            }
            // The combine pass reads no KV, so it is the same task on both paths.
            return unifiedLayer.task(
                    "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    state.workspace.wrapAttSplit,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    attentionSplits);
        }
        if (useFp16KVCache()) {
            // Flash Attention over the half-precision KV cache (FP32 accumulation).
            // The scalar-read variant is an evaluation aid; the packed variant is the default.
            if (scalarFp16KeyValueReads) {
                return unifiedLayer.task(
                        "attention",
                        TransformerPagedKvKernels::processHeadsFlashAttentionFP16ScalarPaged,
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
        } else if (schedulerType == SchedulerType.NVIDIA) {
            // Flash Attention (optimized for NVIDIA GPUs)
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
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
                    state.kvBlockStride);
        } else {
            // Standard parallel attention (for non-NVIDIA backends)
            return unifiedLayer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsParallelPaged,
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
                    state.kvBlockStride);
        }
    }

    // @formatter:on

}
