package org.beehive.jllm.backend.tornado.layers.type.q8_0;

import org.beehive.jllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Qwen2State;
import org.beehive.jllm.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.jllm.model.qwen2.Qwen2Configuration;
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
            String taskGraphName,
            Qwen2State state,
            Qwen2TornadoWeights weights,
            Qwen2Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
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
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int fusedQKVGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker = new WorkerGrid1D(fusedQKVGlobal);
        fusedQKVWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // WorkerGrid for fused QKV bias addition (dimension is dimQ)
        WorkerGrid fusedQKVBiasWorker = new WorkerGrid1D(config.dim());
        fusedQKVBiasWorker.setGlobalWork(config.dim(), 1, 1);
        fusedQKVBiasWorker.setLocalWork(32, 1, 1);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

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
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_qkv_projection", fusedQKVWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".fused_qkv_bias", fusedQKVBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }
        return tornadoForwardScheduler;
    }

    /** Setup a single transformer layer for Qwen2 with Q8_0 quantization and GQA */
    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);

        unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
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
                weights.w3Layered[layerIndex].asByteArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task(
                "attn_rms_reduce",
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
                    "attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen2State.workspace.temp, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        unifiedLayer.task(
                "attn_rms_qkv_projection",
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

        // Fused Q/K/V Bias Addition (3→1 kernel fusion)
        unifiedLayer.task(
                "fused_qkv_bias",
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
                    "rope_and_kv_cache",
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
                    "rope_and_kv_cache",
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
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "attention",
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
                    "attention",
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
        unifiedLayer.task(
                "attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                qwen2State.workspace.wrapXb, // input: attention output
                qwen2State.workspace.wrapX, // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asByteArray(), // Wo
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
                qwen2State.workspace.tempFFN, // output: scale factor
                qwen2State.workspace.wrapX, // input: hidden state
                config.dim(), // dimension
                config.rmsNormEps(), // epsilon
                qwen2State.localSize); // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task(
                    "ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen2State.workspace.tempFFN, // scale factor (in/out)
                    config.dim(), // dimension
                    config.rmsNormEps()); // epsilon
        }

        // Fused RMS Apply + Gate/Up Projection + SiLU + GLU
        // (Replaces mapContextFFN + fusedFeedForwardWithSiLUAndGLUActivation)
        unifiedLayer.task(
                "rms_ffn_gate_up",
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

        // Down Projection with Residual
        unifiedLayer.task(
                "ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                qwen2State.workspace.wrapHb, // input: FFN intermediate
                qwen2State.workspace.wrapX, // output: wrapX += W2 · wrapHb
                weights.w2Layered[layerIndex].asByteArray(), // W2 (down)
                config.hiddenDim(), // input dim
                config.dim(), // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    // @formatter:on

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
}
