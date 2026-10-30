package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.GraniteKernels;
import org.beehive.jitllm.backend.tornado.kernels.GranitePagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.scheduling.LaneAttentionPolicy;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class GraniteQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<GraniteTornadoWeights, GraniteConfiguration> {

    public GraniteQ8_0FFNLayers(
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
        setupFFNLayers();
    }

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

        // === Data Setup ===
        if (firstInGraph) {
            String wrapXSrc = predecessorGraphName(layerIndex);
            if (wrapXSrc != null) {
                unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
            } else {
                unifiedLayer.consumeFromDevice(state.workspace.wrapX);
            }
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
        bindLayerWeights(unifiedLayer, weightSourceGraphName(layerIndex), layerIndex, layerWeights);
        if (firstInGraph) {
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            bindDecodeScratch(unifiedLayer, layerIndex);
        }

        // === Attention Block ===
        // RMS Normalization
        unifiedLayer.task(
                tp + "attn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.temp,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    tp + "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            // Packed integer dot products over the activation quantized once to Q8 blocks.
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    tp + "attn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
            unifiedLayer.task(
                    tp + "qkv_projection",
                    TransformerComputeKernelsQ8_0DP4A::fusedQKVMatmulQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    config.kvDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "attn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);

            // QKV Projection (fused with Q8 dequantization)
            unifiedLayer.task(
                    tp + "qkv_projection",
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
        }

        // RoPE + KV Cache
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    tp + "rope_and_kv_cache",
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
                    tp + "rope_and_kv_cache",
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

        // Output Projection (Wo) with the scaled residual
        if (packed) {
            // The attention output is a new activation: quantized fresh for the projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    tp + "attn_out_quantize",
                    context,
                    state.workspace,
                    state.workspace.wrapXb);
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    TransformerComputeKernelsQ8_0DP4A
                            ::matrixVectorGenericWithScaledResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    PACKED_LOCAL,
                    config.residualScale());
        } else {
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    GraniteKernels::matrixVectorGenericWithResidualQ8_0ByteWithGraniteScale,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.woLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC,
                    config.residualScale());
        }

        // === FFN Block ===
        // RMS Normalization
        unifiedLayer.task(
                tp + "ffn_rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempFFN,
                state.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    tp + "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempFFN,
                    config.dim(),
                    config.rmsNormEps());
        }

        if (packed) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    unifiedLayer,
                    tp + "ffn_rms_apply",
                    context,
                    state.workspace,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
            unifiedLayer.task(
                    tp + "ffn_gate_up",
                    TransformerComputeKernelsQ8_0DP4A::fusedFFNGateUpSiLUQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
            // SwiGLU's output, quantized fresh for the down projection.
            PackedActivationTasks.quantize(
                    unifiedLayer,
                    tp + "ffn_down_quantize",
                    context,
                    state.workspace,
                    state.workspace.wrapHb);
            unifiedLayer.task(
                    tp + "ffn_down_proj",
                    TransformerComputeKernelsQ8_0DP4A
                            ::matrixVectorGenericWithScaledResidualQ8_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    PACKED_LOCAL,
                    config.residualScale());
        } else {
            // Fully fused: RMS apply + Gate/Up projections + SiLU + GLU (Q8 dequantization)
            unifiedLayer.task(
                    tp + "rms_ffn_gate_up",
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
                    tp + "ffn_down_proj",
                    GraniteKernels::matrixVectorGenericWithResidualQ8_0ByteWithGraniteScale,
                    context,
                    state.workspace.wrapHb,
                    state.workspace.wrapX,
                    weights.w2Layered[layerIndex].asByteArray(),
                    config.hiddenDim(),
                    config.dim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC,
                    config.residualScale());
        }

        // Keep activation X on device for next layer
        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    // @formatter:on

    /** Whether attention runs the scaled lane-cooperative split-KV kernel. */
    private boolean laneAttention() {
        return useFp16KVCache()
                && !packedHalf2Attention
                && LaneAttentionPolicy.scaledLaneCooperativeAttention(config.headSize());
    }

    /** Adjacent decode layers per graph, which removes all but one host round trip per group. */
    @Override
    protected int layersPerGraph() {
        return LAYERS_PER_GRAPH;
    }

    static final int LAYERS_PER_GRAPH = 8;

    /** Lanes per output row of the packed projections: a multiple of 32. */
    private static final int PACKED_LOCAL = 64;

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
        boolean packed = PackedActivationTasks.enabled();
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
                tornadoForwardScheduler.addWorkerGrid(l + "attn_rms_apply", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid(l + "qkv_projection", fusedQkvWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attn_output_proj", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "rms_ffn_gate_up", configHiddenDimRowMajorWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        l + "ffn_down_proj", configDimRowMajorGlobalWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(l + "rope_and_kv_cache", ropeWithCacheWorker);
            if (laneAttention()) {
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attention",
                        LaneAttentionPolicy.worker(
                                config.numberOfHeads() * State.SPLIT_KV, config.headSize()));
                tornadoForwardScheduler.addWorkerGrid(
                        l + "attention_combine", parallelAttentionWorker);
            } else {
                tornadoForwardScheduler.addWorkerGrid(l + "attention", parallelAttentionWorker);
            }
            tornadoForwardScheduler.addWorkerGrid(l + "ffn_rms_reduce", rmsReduceWorker);
        }

        return tornadoForwardScheduler;
    }

    // @formatter:off
    private TaskGraph configureAttention(
            TaskGraph unifiedLayer, int layerIndex, GraniteConfiguration config) {
        String tp = layerTaskPrefix(layerIndex);
        if (laneAttention()) {
            // Split-KV with the lane-cooperative kernel, scored by Granite's attention multiplier
            // rather than 1/sqrt(headSize); the combine pass merges the partials.
            unifiedLayer.task(
                    tp + "attention",
                    LaneAttentionPolicy.scaledKernel(config.headSize()),
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
                    State.SPLIT_KV,
                    config.attentionScale());
            return unifiedLayer.task(
                    tp + "attention_combine",
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    state.workspace.wrapAttSplit,
                    state.workspace.wrapXb,
                    config.numberOfHeads(),
                    config.headSize(),
                    State.SPLIT_KV);
        }
        // Metal runs the flash kernel too: processHeadsParallelGranitePaged produces wrong output
        // there (Granite 3.2 2B and 4.0 1B answered with punctuation in both representations).
        if (schedulerType == SchedulerType.NVIDIA
                || SchedulerDetectionService.isSubgroupShuffle32Supported()) {
            // Flash Attention (optimized for NVIDIA GPUs)
            if (useFp16KVCache()) {
                return unifiedLayer.task(
                        tp + "attention",
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
                    tp + "attention",
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
                    tp + "attention",
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
