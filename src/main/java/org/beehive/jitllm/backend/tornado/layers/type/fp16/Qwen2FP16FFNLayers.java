package org.beehive.jitllm.backend.tornado.layers.type.fp16;

import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.Qwen2Q8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.Fp16GemvUnrollPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Qwen2FP16FFNLayers: FP16 transformer-layer TaskGraphs for Qwen2 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Qwen3: - No tempQcur/tempKcur fields in Qwen2State - Includes bias terms
 * for Q, K, V projections - Standard GQA (no parallel offset RMSNorm) - Uses
 * Qwen2Kernels::processHeadsFlashAttention for attention computation - Uses
 * Qwen3Kernels::ropeRotationWithCacheCopy for position embeddings - Simpler matrix dimensions (uses
 * config.dim() and config.kvDim() directly)
 *
 * <p>Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2FP16FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen2TornadoWeights, Qwen2Configuration> {

    // Typed reference to Qwen2-specific state
    private final Qwen2State qwen2State;

    /**
     * 32-lane subgroup-shuffle GEMV reductions instead of the shared-memory trees. Metal only, and
     * a correctness fix rather than an optimisation: the shared-memory-tree kernels below produce
     * numerically wrong output on Metal — fluent-looking token salad, with the process still
     * exiting 0 and reporting throughput. It stayed hidden while TornadoVM 5.2.0 could not sketch
     * these kernels at all and the tuple failed loudly instead; TornadoVM 6.0.0 compiles them,
     * turning a loud failure into a silent one.
     *
     * <p>Same gate, same kernels and same reasoning as {@code
     * LlamaFP16FFNLayers.useSimd32Reduction} and {@code Qwen3FP16FFNLayers.useWarpMatmul}. {@code
     * SUBGROUP_SHUFFLE_32} is granted to Metal only ({@code TornadoDevices.capabilitiesOf}), so
     * CUDA and OpenCL keep the exact kernels they select today — this class had no capability
     * branch at all before.
     */
    private final boolean useSimd32Reduction =
            SchedulerDetectionService.isSubgroupShuffle32Supported();

    public Qwen2FP16FFNLayers(
            Qwen2State state,
            Qwen2TornadoWeights weights,
            Qwen2Configuration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
        this.qwen2State = state;
        setupFFNLayers();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        int h = config.numberOfHeads();
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(h / 2, ic / 2, 1);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configKvDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker =
                WorkerGridFactory.genericWorker(
                        configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 32);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);

        // Parallel attention worker configuration
        // Calculate optimal local work size based on head dimension
        int optimalLocalSize = Math.min(config.headSize(), 64); // Start with 64 threads per head
        if (config.headSize() % optimalLocalSize != 0) {
            // Find largest divisor of headSize <= 64
            for (int size = 64; size >= 1; size--) {
                if (config.headSize() % size == 0) {
                    optimalLocalSize = size;
                    break;
                }
            }
        }

        // WorkerGrid for fused QKV bias addition (dimension is dimQ)
        WorkerGrid fusedQKVBiasWorker = new WorkerGrid1D(config.dim());
        fusedQKVBiasWorker.setGlobalWork(config.dim(), 1, 1);
        fusedQKVBiasWorker.setLocalWork(32, 1, 1); // Or an optimized local size

        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * optimalLocalSize, 1, 1);
        parallelAttentionWorker.setLocalWork(optimalLocalSize, 1, 1);

        int fusedQKVGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker =
                WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused QKV bias worker (covers dimQ which is largest)
        WorkerGrid fusedQKVBiasWorkerNorm = new WorkerGrid1D(config.dim());
        fusedQKVBiasWorkerNorm.setGlobalWork(config.dim(), 1, 1);
        fusedQKVBiasWorkerNorm.setLocalWork(32, 1, 1);

        // Map workers to tasks
        boolean lane = laneAttention(config.headSize());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String l = layerGraphName(i) + "." + layerTaskPrefix(i);
            tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_qkv_projection", fusedQKVWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "fused_qkv_bias", fusedQKVBiasWorkerNorm);
            tornadoForwardScheduler.addWorkerGrid(l + "rope_and_kv_cache", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "fused_qkv_bias", fusedQKVBiasWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "ffn_down_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    l + "fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "mapContextFFN", rmsNormWorker);
            if (lane) {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attention",
                        LaneAttentionPolicy.worker(
                                config.numberOfHeads() * State.SPLIT_KV, config.headSize()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attention_combine",
                        WorkerGridFactory.createAttentionWorker(
                                config.numberOfHeads(), config.headSize()));
            } else {
                tornadoForwardScheduler.addWorkerGrid(l + "attention", parallelAttentionWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(
                    l + "rms_ffn_gate_up", configHiddenDimRowMajorWorker);
        }
        return tornadoForwardScheduler;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Qwen2FP16FFNLayers - Optimized)
     *
     * <p>══════════════════════════════════════════════════════════════════════════════ ATTENTION
     * BLOCK ══════════════════════════════════════════════════════════════════════════════
     *
     * <p>wrapX (FP32) │ ▼ ┌─────────────────┐ │ attn_rms_reduce │──▶ temp (scale factor for
     * RMSNorm) └────────┬────────┘ │ ▼ ┌─────────────────────────┐ │ attn_rms_qkv_projection │──▶
     * wrapQ, wrapK, wrapV (FP32) └───────────┬─────────────┘ (fused: RMS apply + Q/K/V matmuls) │ ▼
     * ┌────────────────┐ │ fused_qkv_bias │──▶ wrapQ, wrapK, wrapV += biases └───────┬────────┘
     * (fused: Q + K + V bias addition) │ ▼ ┌───────────────────┐
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
     * <p>Task Count: 9 tasks (NVIDIA) / 10 tasks (non-NVIDIA) Previous: 12 tasks Reduction: 3 tasks
     * eliminated (25% fewer kernel launches)
     *
     * <p>Data Flow Summary: Input: wrapX (FP32) - hidden state from previous layer Output: wrapX
     * (FP32) - updated hidden state with residual connections
     *
     * <p>Key Fusion Points (vs previous 12 tasks): • fused_qkv_bias: Fused Q + K + V bias addition
     * (3→1 kernel) • rms_ffn_gate_up: Fused RMS apply + W1/W3 matmuls + SiLU + GLU (eliminates
     * separate mapContextFFN kernel)
     *
     * <p>Qwen2-Specific: • GQA: nHeads (Q) != nHeadKv (K/V), with kvMul = nHeads / nHeadKv • Bias
     * terms: Q, K, V projections include bias (unlike Qwen3) • No Q/K RMSNorm: Unlike Qwen3, Qwen2
     * doesn't normalize Q/K after projection
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        return appendFFNLayer(null, layerIndex);
    }

    @Override
    protected TaskGraph appendFFNLayer(TaskGraph existing, int layerIndex) {
        boolean firstInGraph = existing == null;
        String tp = layerTaskPrefix(layerIndex);
        TaskGraph unifiedLayer =
                firstInGraph ? new TaskGraph(layerGraphName(layerIndex)) : existing;

        if (firstInGraph) {
            String wrapXSrc = predecessorGraphName(layerIndex);
            if (wrapXSrc != null) {
                unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
            } else {
                unifiedLayer.consumeFromDevice(state.workspace.wrapX);
            }
        }
        Object[] layerWeights = {
            // Attention weights
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asHalfFloatArray(),
            weights.wkLayered[layerIndex].asHalfFloatArray(),
            weights.wvLayered[layerIndex].asHalfFloatArray(),
            weights.woLayered[layerIndex].asHalfFloatArray(),
            // Qwen2-specific bias terms
            weights.q_biasLayered[layerIndex].asFloatArray(),
            weights.k_biasLayered[layerIndex].asFloatArray(),
            weights.v_biasLayered[layerIndex].asFloatArray(),
            // FFN weights
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asHalfFloatArray(),
            weights.w2Layered[layerIndex].asHalfFloatArray(),
            weights.w3Layered[layerIndex].asHalfFloatArray()
        };
        bindLayerWeights(unifiedLayer, weightSourceGraphName(layerIndex), layerIndex, layerWeights);
        if (firstInGraph) {
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            bindDecodeScratch(unifiedLayer, layerIndex);
        }

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                tp + "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                qwen2State.workspace.temp, // output: scale factor
                qwen2State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                qwen2State.localSize); // local memory size

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    tp + "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        // Fused RMS Apply + QKV Projection
        // Same logical operation, same task name, same output contract either way; only the
        // reduction kernel differs, by device capability. Both variants run on the existing 32-wide
        // worker grid (LOCAL_WORK_GROUP_SIZE_ALLOC == 32), so no grid-scheduler change is needed.
        unifiedLayer.task(
                tp + "attn_rms_qkv_projection",
                useSimd32Reduction
                        ? (Fp16GemvUnrollPolicy.unrolled()
                                ? Qwen3Kernels::fusedRmsNormQKVMatmulWarpUnrolled
                                : Qwen3Kernels::fusedRmsNormQKVMatmulWarp)
                        : Qwen3Kernels::fusedRmsNormQKVMatmul,
                context,
                qwen2State.workspace.wrapX, // input: raw hidden state (FP32)
                qwen2State.workspace.wrapQ, // output: Q vectors
                qwen2State.workspace.wrapK, // output: K vectors
                qwen2State.workspace.wrapV, // output: V vectors
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                qwen2State.workspace.temp, // RMS scale factor from reduction
                weights.wqLayered[layerIndex].asHalfFloatArray(), // Wq
                weights.wkLayered[layerIndex].asHalfFloatArray(), // Wk
                weights.wvLayered[layerIndex].asHalfFloatArray(), // Wv
                config.dim(), // input dimension
                config.dim(), // Q output dimension
                config.kvDim(), // K/V output dimension (GQA)
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Q/K/V Bias Addition (3→1 kernel fusion)
        unifiedLayer.task(
                tp + "fused_qkv_bias",
                TransformerComputeKernelsLayered::fusedQKvBiasAddition,
                context,
                qwen2State.workspace.wrapQ, // Q (in/out)
                qwen2State.workspace.wrapK, // K (in/out)
                weights.q_biasLayered[layerIndex].asFloatArray(), // Q bias
                qwen2State.workspace.wrapV, // V (in/out)
                weights.k_biasLayered[layerIndex].asFloatArray(), // K bias
                weights.v_biasLayered[layerIndex].asFloatArray(), // V bias
                config.dim(), // dimQ
                config.kvDim()); // dimKV

        // Fused RoPE Rotation + KV Cache Write
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    tp + "rope_and_kv_cache",
                    Qwen3PagedKvKernels::ropeRotationWithCacheCopyFP16Paged,
                    context,
                    qwen2State.workspace.positionHolder, // current sequence position
                    qwen2State.workspace.wrapQ, // Q (rotated in-place)
                    qwen2State.workspace.wrapK, // K (rotated in-place)
                    qwen2State.workspace.wrapV, // V (copied to cache)
                    qwen2State.workspace.wrapKeyCacheFP16, // key cache (write)
                    qwen2State.workspace.wrapValueCacheFP16, // value cache (write)
                    config.ropeTheta(),
                    config.numberOfKeyValueHeads(), // nHeadKv
                    config.headSize(), // per-head dimension
                    config.kvDim(), // kvDim
                    layerIndex, // layer offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        } else {
            unifiedLayer.task(
                    tp + "rope_and_kv_cache",
                    Qwen3PagedKvKernels::ropeRotationWithCacheCopyPaged,
                    context,
                    qwen2State.workspace.positionHolder, // current sequence position
                    qwen2State.workspace.wrapQ, // Q (rotated in-place)
                    qwen2State.workspace.wrapK, // K (rotated in-place)
                    qwen2State.workspace.wrapV, // V (copied to cache)
                    state.workspace.wrapKeyCache, // key cache (write)
                    state.workspace.wrapValueCache, // value cache (write)
                    config.ropeTheta(),
                    config.numberOfKeyValueHeads(), // nHeadKv
                    config.headSize(), // per-head dimension
                    config.kvDim(), // kvDim
                    layerIndex, // layer offset
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // max sequence length
        }

        // Flash Attention
        if (laneAttention(config.headSize())) {
            // Split-KV with the lane-cooperative kernel: more workgroups than heads, and the
            // partials merged by the combine pass.
            unifiedLayer.task(
                    tp + "attention",
                    LaneAttentionPolicy.kernel(config.headSize()),
                    context,
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapKeyCacheFP16,
                    qwen2State.workspace.wrapValueCacheFP16,
                    qwen2State.workspace.wrapAttSplit,
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    qwen2State.workspace.positionHolder,
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    State.SPLIT_KV);
            unifiedLayer.task(
                    tp + "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    qwen2State.workspace.wrapAttSplit,
                    qwen2State.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    State.SPLIT_KV);
        } else if (useFp16KVCache()) {
            unifiedLayer.task(
                    tp + "attention",
                    Qwen2PagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    qwen2State.workspace.wrapQ, // query vectors
                    qwen2State.workspace.wrapKeyCacheFP16, // key cache
                    qwen2State.workspace.wrapValueCacheFP16, // value cache
                    qwen2State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    config.headSize(), // headSize
                    config.kvDim(), // kvDim
                    config.kvMul(), // kvMul (nHeads / nHeadKv)
                    qwen2State.workspace.positionHolder, // position
                    layerIndex, // layer index
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // context length
        } else {
            unifiedLayer.task(
                    tp + "attention",
                    Qwen2PagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    qwen2State.workspace.wrapQ, // query vectors
                    state.workspace.wrapKeyCache, // key cache
                    state.workspace.wrapValueCache, // value cache
                    qwen2State.workspace.wrapXb, // output: attention result
                    config.numberOfHeads(), // nHeads
                    config.headSize(), // headSize
                    config.kvDim(), // kvDim
                    config.kvMul(), // kvMul (nHeads / nHeadKv)
                    qwen2State.workspace.positionHolder, // position
                    layerIndex, // layer index
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride); // context length
        }

        // Output Projection with Residual
        // The Simd32 variant takes no localWorkGroupSize (it is fixed at 32 lanes), so this is an
        // if/else rather than a ternary on the method reference.
        if (useSimd32Reduction) {
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    (Fp16GemvUnrollPolicy.unrolled()
                            ? TransformerComputeKernelsLayered
                                    ::matrixVectorGenericWithResidualSimd32Unrolled
                            : TransformerComputeKernelsLayered
                                    ::matrixVectorGenericWithResidualSimd32),
                    context,
                    qwen2State.workspace.wrapXb, // input: attention output
                    qwen2State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asHalfFloatArray(), // Wo
                    config.dim(), // input dim
                    config.dim()); // output dim
        } else {
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    qwen2State.workspace.wrapXb, // input: attention output
                    qwen2State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asHalfFloatArray(), // Wo
                    config.dim(), // input dim
                    config.dim(), // output dim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                tp + "ffn_rms_reduce",
                rmsReduceKernel(),
                context,
                qwen2State.workspace.tempFFN, // output: scale factor
                qwen2State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                qwen2State.localSize); // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    tp + "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen2State.workspace.tempFFN, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        // Fused RMS Apply + Gate/Up Projection + SiLU + GLU
        // (Replaces mapContextFFN + fusedFeedForwardWithSiLUAndGLUActivation)
        unifiedLayer.task(
                tp + "rms_ffn_gate_up",
                useSimd32Reduction
                        ? (Fp16GemvUnrollPolicy.unrolled()
                                ? TransformerComputeKernelsLayered
                                        ::fusedRmsNormFFNGateUpWarpUnrolled
                                : TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpWarp)
                        : TransformerComputeKernelsLayered::fusedRmsNormFFNGateUp,
                context,
                qwen2State.workspace.wrapX, // input: raw hidden state (FP32)
                qwen2State.workspace.wrapHb, // output: SiLU(x·W1) ⊙ (x·W3)
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                qwen2State.workspace.tempFFN, // RMS scale factor
                weights.w1Layered[layerIndex].asHalfFloatArray(), // W1 (gate)
                weights.w3Layered[layerIndex].asHalfFloatArray(), // W3 (up)
                config.dim(), // input dimension
                config.hiddenDim(), // hidden dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual
        if (useSimd32Reduction) {
            unifiedLayer
                    .task(
                            tp + "ffn_down_proj",
                            (Fp16GemvUnrollPolicy.unrolled()
                                    ? TransformerComputeKernelsLayered
                                            ::matrixVectorGenericWithResidualSimd32Unrolled
                                    : TransformerComputeKernelsLayered
                                            ::matrixVectorGenericWithResidualSimd32),
                            context,
                            qwen2State.workspace.wrapHb, // input: FFN intermediate
                            qwen2State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                            weights.w2Layered[layerIndex].asHalfFloatArray(), // W2 (down)
                            config.hiddenDim(), // input dim
                            config.dim()) // output dim
                    .persistOnDevice(state.workspace.wrapX);
        } else {
            unifiedLayer
                    .task(
                            tp + "ffn_down_proj",
                            TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                            context,
                            qwen2State.workspace.wrapHb, // input: FFN intermediate
                            qwen2State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                            weights.w2Layered[layerIndex].asHalfFloatArray(), // W2 (down)
                            config.hiddenDim(), // input dim
                            config.dim(), // output dim
                            LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .persistOnDevice(state.workspace.wrapX);
        }

        return unifiedLayer;
    }

    /** Adjacent decode layers per graph, which removes all but one host round trip per group. */
    @Override
    protected int layersPerGraph() {
        return Qwen2Q8_0FFNLayers.LAYERS_PER_GRAPH;
    }

    /**
     * Binds the split-KV attention scratch: uploaded once by layer 0 and passed along the graph
     * chain. Called by the graph that opens a group, so subclasses that replace {@link
     * #configureLayerDataTransfers} keep it.
     */
    protected void bindDecodeScratch(TaskGraph unifiedLayer, int layerIndex) {
        Object[] scratch = {state.workspace.wrapAttSplit};
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch);
            return;
        }
        String source = predecessorGraphName(layerIndex);
        if (source != null) {
            unifiedLayer.consumeFromDevice(source, scratch);
        } else {
            unifiedLayer.consumeFromDevice(scratch);
        }
    }

    /** Configure data transfers for first and subsequent layers */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen2State.workspace.positionHolder,
                    qwen2State.workspace.temp,
                    qwen2State.workspace.tempFFN); //
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION, //
                    context,
                    qwen2State.workspace.wrapXb,
                    qwen2State.workspace.wrapXb2, //
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapK,
                    qwen2State.workspace.wrapV, //
                    keyCache(),
                    valueCache(), //
                    qwen2State.workspace.wrapAtt,
                    qwen2State.workspace.wrapHb); //
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice( //
                    context,
                    qwen2State.workspace.wrapXb,
                    qwen2State.workspace.wrapXb2, //
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapK,
                    qwen2State.workspace.wrapV, //
                    keyCache(),
                    valueCache(), //
                    qwen2State.workspace.wrapAtt,
                    qwen2State.workspace.wrapHb, //
                    qwen2State.workspace.positionHolder //
                    );
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

    /**
     * The graph {@code wrapX} is consumed from, or {@code null} for the no-arg form. The batched
     * prefill/decode plan's decode layers name their predecessor; see {@code
     * LlamaFP16FFNLayers#predecessorGraphName}.
     */
    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    /**
     * The graph that already uploaded this layer's weights, or {@code null} to upload them here;
     * see {@code LlamaFP16FFNLayers#weightSourceGraphName}.
     */
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }
}
