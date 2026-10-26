package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
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
 * Qwen2Q8_0FFNLayers: Q8_0 transformer-layer TaskGraphs for Qwen2 with Group Query Attention (GQA)
 * support.
 *
 * <p>Key Differences from Qwen2FP16FFNLayers: - Uses Q8_0-quantized weights (getQuants() and
 * getScales()) - Same attention and RoPE kernels as FP16 version - 8-bit integer computations with
 * dequantization - 2x memory compression vs FP16 - Includes bias terms for Q, K, V projections
 *
 * <p>Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2Q8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen2TornadoWeights, Qwen2Configuration> {
    // Typed reference to Qwen2-specific state
    private final Qwen2State qwen2State;

    public Qwen2Q8_0FFNLayers(
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
        ropeWorker.setLocalWork(1, 1, 1);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker =
                WorkerGridFactory.genericWorker(
                        configKvDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedQKVGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker =
                WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // WorkerGrid for fused QKV bias addition (dimension is dimQ)
        WorkerGrid fusedQKVBiasWorker = new WorkerGrid1D(config.dim());
        fusedQKVBiasWorker.setGlobalWork(config.dim(), 1, 1);
        fusedQKVBiasWorker.setLocalWork(32, 1, 1);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker =
                WorkerGridFactory.genericWorker(
                        configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 32);
        // Race-free single-workgroup reduction on the NVIDIA path; see rmsReduceKernel().
        WorkerGrid rmsReduceWorker = rmsReduceWorker(rmsNormWorker);

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

        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * optimalLocalSize, 1, 1);
        parallelAttentionWorker.setLocalWork(optimalLocalSize, 1, 1);

        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(
                32, 1, 1); // Set local work size to 32 (for copying to caches)

        // Map workers to tasks
        boolean packed = PackedActivationTasks.enabled();
        boolean lane = laneAttention(config.headSize());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String l = layerGraphName(i) + "." + layerTaskPrefix(i);
            tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_reduce", rmsReduceWorker);
            if (packed) {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "qkv_projection",
                        PackedActivationTasks.projectionWorker(
                                config.dim() + 2 * config.kvDim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_out_quantize",
                        PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_rms_apply", PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_gate_up",
                        PackedActivationTasks.projectionWorker(config.hiddenDim(), PACKED_LOCAL));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_quantize",
                        PackedActivationTasks.quantizeWorker(config.hiddenDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj",
                        PackedActivationTasks.projectionWorker(config.dim(), PACKED_LOCAL));
            } else {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_rms_qkv_projection", fusedQKVWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "rms_ffn_gate_up", configHiddenDimRowMajorWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj", configDimRowMajorGlobalWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(l + "fused_qkv_bias", fusedQKVBiasWorker);
            tornadoForwardScheduler.addWorkerGrid(l + "rope_and_kv_cache", ropeWorker);
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
            tornadoForwardScheduler.addWorkerGrid(l + "ffn_rms_reduce", rmsReduceWorker);
        }
        return tornadoForwardScheduler;
    }

    /** Setup a single transformer layer for Qwen2 with Q8_0 quantization and GQA */
    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        return appendFFNLayer(null, layerIndex);
    }

    @Override
    protected TaskGraph appendFFNLayer(TaskGraph existing, int layerIndex) {
        boolean firstInGraph = existing == null;
        String tp = layerTaskPrefix(layerIndex);
        boolean packed = PackedActivationTasks.enabled();
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
            weights.wqLayered[layerIndex].asByteArray(),
            weights.wkLayered[layerIndex].asByteArray(),
            weights.wvLayered[layerIndex].asByteArray(),
            weights.woLayered[layerIndex].asByteArray(),
            // Qwen2-specific bias terms
            weights.q_biasLayered[layerIndex].asFloatArray(),
            weights.k_biasLayered[layerIndex].asFloatArray(),
            weights.v_biasLayered[layerIndex].asFloatArray(),
            // FFN weights
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asByteArray(),
            weights.w2Layered[layerIndex].asByteArray(),
            weights.w3Layered[layerIndex].asByteArray()
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

        // Final normalization (non-NVIDIA only). Not optional: on the NON_NVIDIA path
        // rmsReduceKernel() is reductionOneBlockWithLayer, which splits the sum of squares across
        // workgroups and combines them with no inter-workgroup synchronization, so the scale it
        // leaves in temp is only correct once this task recomputes it. The FFN block below has
        // always had its counterpart; the attention block did not, which made every Metal run of
        // Qwen2 Q8_0 normalize the attention input by a partial sum and emit token salad while
        // exiting 0. Invisible on CUDA and on OpenCL-over-NVIDIA, where SINGLE_PASS_RMS selects
        // the single-workgroup reduction and no finalize task exists to omit.
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    tp + "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen2State.workspace.temp, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        if (packed) {
            // Packed integer dot products over the activation quantized once to Q8 blocks.
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    tp + "attn_rms_apply",
                    context,
                    qwen2State.workspace,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    qwen2State.workspace.temp);
            unifiedLayer.task(
                    tp + "qkv_projection",
                    TransformerComputeKernelsQ8_0DP4A::fusedQKVMatmulQ8_0DP4A,
                    context,
                    qwen2State.workspace.wrapXbQuants,
                    qwen2State.workspace.wrapXbScales,
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapK,
                    qwen2State.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    config.kvDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "attn_rms_qkv_projection",
                    Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                    context,
                    qwen2State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen2State.workspace.wrapQ, // output: Q vectors
                    qwen2State.workspace.wrapK, // output: K vectors
                    qwen2State.workspace.wrapV, // output: V vectors
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen2State.workspace.temp, // RMS scale factor from reduction
                    weights.wqLayered[layerIndex].asByteArray(), // Wq (Q8_0)
                    weights.wkLayered[layerIndex].asByteArray(), // Wk (Q8_0)
                    weights.wvLayered[layerIndex].asByteArray(), // Wv (Q8_0)
                    config.dim(), // input dimension
                    config.dim(), // Q output dimension
                    config.kvDim(), // K/V output dimension (GQA: reduced)
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

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
        if (packed) {
            // The attention output is a new activation: quantized fresh for the projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    tp + "attn_out_quantize",
                    context,
                    qwen2State.workspace,
                    qwen2State.workspace.wrapXb);
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    qwen2State.workspace.wrapXbQuants,
                    qwen2State.workspace.wrapXbScales,
                    qwen2State.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    qwen2State.workspace.wrapXb, // input: attention output
                    qwen2State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                    weights.woLayered[layerIndex].asByteArray(), // Wo
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
        if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    tp + "ffn_rms_apply",
                    context,
                    qwen2State.workspace,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    qwen2State.workspace.tempFFN);
            unifiedLayer.task(
                    tp + "ffn_gate_up",
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                    context,
                    qwen2State.workspace.wrapXbQuants,
                    qwen2State.workspace.wrapXbScales,
                    qwen2State.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpQ8_0,
                    context,
                    qwen2State.workspace.wrapX, // input: raw hidden state (FP32)
                    qwen2State.workspace.wrapHb, // output: SiLU(x·W1) ⊙ (x·W3)
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                    qwen2State.workspace.tempFFN, // RMS scale factor
                    weights.w1Layered[layerIndex].asByteArray(), // W1 (gate)
                    weights.w3Layered[layerIndex].asByteArray(), // W3 (up)
                    config.dim(), // input dimension
                    config.hiddenDim(), // hidden dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Down Projection with Residual
        if (packed) {
            // SwiGLU's output, quantized fresh for the down projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    tp + "ffn_down_quantize",
                    context,
                    qwen2State.workspace,
                    qwen2State.workspace.wrapHb);
            unifiedLayer.task(
                    tp + "ffn_down_proj",
                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                    context,
                    qwen2State.workspace.wrapXbQuants,
                    qwen2State.workspace.wrapXbScales,
                    qwen2State.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "ffn_down_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                    context,
                    qwen2State.workspace.wrapHb, // input: FFN intermediate
                    qwen2State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                    weights.w2Layered[layerIndex].asByteArray(), // W2 (down)
                    config.hiddenDim(), // input dim
                    config.dim(), // output dim
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    // @formatter:on

    /** Adjacent decode layers per graph, which removes all but one host round trip per group. */
    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    public static final int LAYERS_PER_GRAPH = 8;

    /** Lanes per output row of the packed projections: a multiple of 32. */
    private static final int PACKED_LOCAL = 32;

    /**
     * Binds the split-KV attention scratch and, where the packed projections run, the quantized
     * activation buffers: uploaded once by layer 0 and passed along the graph chain. Called by the
     * graph that opens a group, so subclasses that replace {@link #configureLayerDataTransfers}
     * keep it.
     */
    protected void bindDecodeScratch(TaskGraph unifiedLayer, int layerIndex) {
        Object[] scratch =
                PackedActivationTasks.enabled()
                        ? java.util.stream.Stream.concat(
                                        java.util.stream.Stream.of(state.workspace.wrapAttSplit),
                                        java.util.Arrays.stream(
                                                PackedActivationTasks.buffers(state.workspace)))
                                .toArray()
                        : new Object[] {state.workspace.wrapAttSplit};
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
    // @formatter:off
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen2State.workspace.positionHolder,
                    qwen2State.workspace.temp,
                    qwen2State.workspace.tempFFN);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen2State.workspace.wrapXb,
                    qwen2State.workspace.wrapXb2,
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapK,
                    qwen2State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    qwen2State.workspace.wrapAtt,
                    qwen2State.workspace.wrapHb);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(
                    context,
                    qwen2State.workspace.wrapXb,
                    qwen2State.workspace.wrapXb2,
                    qwen2State.workspace.wrapQ,
                    qwen2State.workspace.wrapK,
                    qwen2State.workspace.wrapV,
                    keyCache(),
                    valueCache(),
                    qwen2State.workspace.wrapAtt,
                    qwen2State.workspace.wrapHb,
                    qwen2State.workspace.positionHolder);
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
