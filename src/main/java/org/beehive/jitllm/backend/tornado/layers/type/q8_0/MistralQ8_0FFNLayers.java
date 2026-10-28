package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.layers.QuantizedProjections;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.mistral.MistralConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class MistralQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, MistralConfiguration> {

    // @formatter:off
    public MistralQ8_0FFNLayers(
            LlamaState state,
            LlamaTornadoWeights weights,
            MistralConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
        this.projections =
                new QuantizedProjections(
                        context, state.workspace, LOCAL_WORK_GROUP_SIZE_ALLOC, PACKED_LOCAL);
        setupFFNLayers();
    }

    // @formatter:on

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = GraphNames.layer(layerIndex);
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        unifiedLayer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // === Attention Block ===
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

        if (nativeLayer(layerIndex)) {
            projections.rmsApply(
                    unifiedLayer,
                    "attn_rms_apply",
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);
            // Three projections rather than one fused kernel: wq, wk and wv can be different
            // representations in the same layer.
            projections.matvec(
                    unifiedLayer,
                    "q_proj",
                    state.workspace.wrapQ,
                    weights.wqLayered[layerIndex],
                    config.dim(),
                    config.dim());
            projections.matvec(
                    unifiedLayer,
                    "k_proj",
                    state.workspace.wrapK,
                    weights.wkLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
            projections.matvec(
                    unifiedLayer,
                    "v_proj",
                    state.workspace.wrapV,
                    weights.wvLayered[layerIndex],
                    config.dim(),
                    config.kvDim());
        } else {
            unifiedLayer.task(
                    "attn_rms_apply",
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.temp);

            unifiedLayer.task(
                    "qkv_projection",
                    TransformerComputeKernelsLayered::fusedQKVMatmulQ8,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.kvDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // Precomputed RoPE tables: the frequencies come from the model's own rope_theta (and
        // any Llama 3.1 frequency scaling) instead of a constant baked into the kernel.
        if (useFp16KVCache()) {
            unifiedLayer.task(
                    "rope_and_kv_cache",
                    TransformerPagedKvKernels::ropeRotationWithCacheCopyPrecomputedFP16Paged,
                    context,
                    state.workspace.positionHolder,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
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
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    config.kvDim(),
                    config.headSize(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }

        configureAttention(unifiedLayer, layerIndex);

        if (nativeLayer(layerIndex)) {
            projections.residualMatvec(
                    unifiedLayer,
                    "attn_out_quantize",
                    "attn_output_proj",
                    state.workspace.wrapXb,
                    weights.woLayered[layerIndex],
                    config.dim(),
                    config.dim());
        } else {
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
        }

        // === FFN Block ===
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

        if (nativeLayer(layerIndex)) {
            projections.rmsApply(
                    unifiedLayer,
                    "ffn_rms_apply",
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    state.workspace.tempFFN);
            projections.gateUp(
                    unifiedLayer,
                    "ffn_gate_up",
                    weights.w1Layered[layerIndex],
                    weights.w3Layered[layerIndex],
                    config.dim(),
                    config.hiddenDim());
            projections.residualMatvec(
                    unifiedLayer,
                    "ffn_down_quantize",
                    "ffn_down_proj",
                    state.workspace.wrapHb,
                    weights.w2Layered[layerIndex],
                    config.hiddenDim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    "rms_ffn_gate_up",
                    TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                    context,
                    state.workspace.wrapX,
                    state.workspace.wrapHb,
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);

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
        }

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
    }

    // @formatter:on

    // @formatter:off
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
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
                    state.workspace.wrapHb,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            if (splitsGateUp()) {
                unifiedLayer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION, state.workspace.wrapHb2);
            }
        } else {
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
                    state.workspace.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
            unifiedLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            if (splitsGateUp()) {
                unifiedLayer.consumeFromDevice(state.workspace.wrapHb2);
            }
        }
        return unifiedLayer;
    }

    // @formatter:on

    // @formatter:off
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

        int fusedQkvGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker =
                WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);
        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".ffn_rms_reduce", rmsReduceWorker);
            if (nativeLayer(i)) {
                addNativeWorkerGrids(tornadoForwardScheduler, GraphNames.layer(i) + ".", i);
                continue;
            }
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    GraphNames.layer(i) + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }

    // @formatter:on

    // @formatter:off
    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (useFp16KVCache()) {
            // Flash attention over the half-precision cache (FP32 accumulation). Mistral runs
            // the non-NVIDIA scheduler for its FP32 path; the FP16 cache takes the flash kernel
            // the other Llama-family layers use, which only needs a CUDA device.
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

    /** The grids of a layer that projects one weight at a time: one workgroup per output row. */
    private void addNativeWorkerGrids(GridScheduler scheduler, String l, int layerIndex) {
        WorkerGrid rmsApplyWorker = projections.rmsApplyWorker(config.dim());
        scheduler.addWorkerGrid(l + "attn_rms_apply", rmsApplyWorker);
        scheduler.addWorkerGrid(l + "q_proj", projections.worker(config.dim()));
        scheduler.addWorkerGrid(l + "k_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(l + "v_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(l + "attn_output_proj", projections.worker(config.dim()));
        scheduler.addWorkerGrid(l + "ffn_rms_apply", rmsApplyWorker);
        projections.addGateUpGrids(
                scheduler,
                l,
                "ffn_gate_up",
                weights.w1Layered[layerIndex],
                weights.w3Layered[layerIndex],
                config.hiddenDim());
        scheduler.addWorkerGrid(l + "ffn_down_proj", projections.worker(config.dim()));
        if (projections.packed()) {
            scheduler.addWorkerGrid(
                    l + "attn_out_quantize", PackedActivationTasks.quantizeWorker(config.dim()));
            scheduler.addWorkerGrid(
                    l + "ffn_down_quantize",
                    PackedActivationTasks.quantizeWorker(config.hiddenDim()));
        }
    }

    /** Lanes per output row of the packed projections: a multiple of 32. */
    private static final int PACKED_LOCAL = 64;

    /** The per-weight projections of a layer that is not entirely Q8_0. */
    private final QuantizedProjections projections;

    // @formatter:off
    /**
     * Whether a layer holds any projection in a representation other than Q8_0, as a K-quant file's
     * do. Such a layer applies each norm once and projects one weight at a time, each through the
     * kernel for its own type ({@link QuantizedProjections}); an all-Q8_0 layer keeps its fused
     * kernels.
     */
    // @formatter:on
    private boolean nativeLayer(int layerIndex) {
        for (TornadoTensor w :
                new TornadoTensor[] {
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
                }) {
            if (w.dataType() != DataType.Q8_0) {
                return true;
            }
        }
        return false;
    }

    /** Whether some layer runs its gate and up apart, which needs {@code wrapHb2}. */
    private boolean splitsGateUp() {
        for (int i = 0; i < config.numberOfLayers(); i++) {
            if (nativeLayer(i)
                    && !projections.fusedGateUp(weights.w1Layered[i], weights.w3Layered[i])) {
                return true;
            }
        }
        return false;
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
     * The FP16 cache on CUDA and Metal, whatever the scheduler: Mistral keeps the non-NVIDIA
     * scheduler for its FP32 attention, and its FP16 path takes the flash kernel, which runs on
     * those devices, not that scheduler.
     */
    @Override
    protected boolean useFp16KVCache() {
        return state.usesFp16KeyValueCache()
                && (Fp16KeyValueSupport.nvidiaDevice() || metalBackend());
    }
}
