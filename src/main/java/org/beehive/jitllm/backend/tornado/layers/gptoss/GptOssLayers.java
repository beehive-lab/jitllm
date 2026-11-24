package org.beehive.jitllm.backend.tornado.layers.gptoss;

import org.beehive.jitllm.backend.tornado.kernels.GptOssKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.plan.UnifiedMemoryWeights;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.MXFP4TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.GptOssState;
import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.inference.weights.tornado.GptOssTornadoWeights;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Single-token gpt-oss layers: Q8_0 attention with biases, RoPE from the YaRN tables, attention
 * with sinks over a sliding window on alternate layers, and the MXFP4 mixture of experts. Every
 * projection runs one warp per output row ({@link GptOssKernels}).
 */
public class GptOssLayers
        extends AbstractTransformerLayerTaskGraphs<GptOssTornadoWeights, GptOssConfiguration> {

    /** Threads per work-group of the warp-per-row projections: four rows per group. */
    static final int ROWS_LOCAL = 128;

    static final int ELEMENT_LOCAL = 128;

    /** Threads per row of the packed int8 Q8_0 projections: a warp. */
    static final int PROJECTION_LOCAL = 32;

    protected final GptOssState gptState;

    /** The batched-prefill graph whose uploads these layers read; null on a single-token plan. */
    protected final String prefillGraphName;

    public GptOssLayers(
            GptOssState state,
            GptOssTornadoWeights weights,
            GptOssConfiguration config,
            SchedulerType schedulerType) {
        this(state, weights, config, schedulerType, null);
    }

    protected GptOssLayers(
            GptOssState state,
            GptOssTornadoWeights weights,
            GptOssConfiguration config,
            SchedulerType schedulerType,
            String prefillGraphName) {
        super(state, weights, config, schedulerType);
        this.gptState = state;
        this.prefillGraphName = prefillGraphName;
        if (Math.max(config.dim(), config.expertHiddenDim()) > GptOssKernels.MAX_STAGED
                || config.dim() % GptOssKernels.EXPERT_ROWS != 0
                || (config.dim() / 32) % 2 != 0
                || (config.expertHiddenDim() / 32) % 2 != 0) {
            throw new UnsupportedOperationException(
                    "the gpt-oss expert kernels stage vectors of at most "
                            + GptOssKernels.MAX_STAGED
                            + " in shared memory, in groups of "
                            + GptOssKernels.EXPERT_ROWS
                            + " rows; this model has dim "
                            + config.dim()
                            + " and expert width "
                            + config.expertHiddenDim());
        }
        if (config.headSize() != GptOssKernels.HEAD) {
            throw new UnsupportedOperationException(
                    "the gpt-oss attention kernel is written for heads of "
                            + GptOssKernels.HEAD
                            + "; this model has "
                            + config.headSize());
        }
        setupFFNLayers();
    }

    static WorkerGrid rows(int rows) {
        int groups = (rows + ROWS_LOCAL / 32 - 1) / (ROWS_LOCAL / 32);
        return WorkerGridFactory.genericWorker(groups * ROWS_LOCAL, ROWS_LOCAL);
    }

    /**
     * Work-groups of {@link GptOssKernels#GROUP} threads, {@link GptOssKernels#EXPERT_ROWS} rows
     * each.
     */
    static WorkerGrid staged(int rows) {
        int groups = (rows + GptOssKernels.EXPERT_ROWS - 1) / GptOssKernels.EXPERT_ROWS;
        return WorkerGridFactory.genericWorker(groups * GptOssKernels.GROUP, GptOssKernels.GROUP);
    }

    public static WorkerGrid elements(int n, int local) {
        return WorkerGridFactory.genericWorker((n + local - 1) / local * local, local);
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        WorkerGrid norm = elements(dim, GptOssKernels.GROUP);
        WorkerGrid combineHeads =
                elements(config.numberOfHeads() * GptOssKernels.HEAD, GptOssKernels.GROUP);
        WorkerGrid qkv =
                PackedActivationTasks.projectionWorker(
                        config.queryDim() + 2 * config.kvDim(), PROJECTION_LOCAL);
        WorkerGrid wo = PackedActivationTasks.projectionWorker(dim, PROJECTION_LOCAL);
        WorkerGrid normDim = elements(dim, ELEMENT_LOCAL);
        WorkerGrid rope = elements(config.numberOfHeads() * config.headSize() / 2, 64);
        WorkerGrid attention =
                groupedAttention()
                        ? WorkerGridFactory.genericWorker(
                                config.numberOfKeyValueHeads()
                                        * GptOssKernels.SPLITS
                                        * GptOssKernels.GROUP,
                                GptOssKernels.GROUP)
                        : WorkerGridFactory.genericWorker(
                                config.numberOfHeads() * GptOssKernels.SPLITS * GptOssKernels.HEAD,
                                GptOssKernels.HEAD);
        WorkerGrid router =
                WorkerGridFactory.genericWorker(
                        config.numberOfExperts() * GptOssKernels.GROUP, GptOssKernels.GROUP);
        WorkerGrid single = WorkerGridFactory.genericWorker(32, 32);
        WorkerGrid gateUp = staged(config.numberOfExpertsUsed() * config.expertHiddenDim());
        WorkerGrid down = staged(config.numberOfExpertsUsed() * dim);
        WorkerGrid combineDim = elements(dim, ELEMENT_LOCAL);
        for (int l = firstLayer; l < endLayer(config.numberOfLayers()); l++) {
            String p = GraphNames.layer(l) + ".";
            scheduler.addWorkerGrid(p + "attn_norm", norm);
            scheduler.addWorkerGrid(p + "qkv", qkv);
            scheduler.addWorkerGrid(p + "rope_kv", rope);
            scheduler.addWorkerGrid(p + "attention", attention);
            scheduler.addWorkerGrid(p + "attention_combine", combineHeads);
            scheduler.addWorkerGrid(p + "attn_output", wo);
            scheduler.addWorkerGrid(p + "ffn_norm", norm);
            scheduler.addWorkerGrid(p + "router", router);
            scheduler.addWorkerGrid(p + "topk", single);
            scheduler.addWorkerGrid(p + "experts_gate_up", gateUp);
            scheduler.addWorkerGrid(p + "experts_down", down);
            scheduler.addWorkerGrid(p + "experts_combine", combineDim);
        }
        return scheduler;
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph(GraphNames.layer(layerIndex));
        String predecessor = predecessorGraphName(layerIndex);
        if (predecessor == null) {
            layer.consumeFromDevice(gptState.workspace.wrapX);
        } else {
            layer.consumeFromDevice(predecessor, gptState.workspace.wrapX);
        }
        uploadWeights(layer, layerIndex);
        transfers(layer, layerIndex);
        attention(layer, layerIndex);
        experts(layer, layerIndex);
        layer.persistOnDevice(gptState.workspace.wrapX, keyCache(), valueCache());
        return layer;
    }

    /** An explicit predecessor for plans that chain graphs from more than one family. */
    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    protected void uploadWeights(TaskGraph layer, int l) {
        GptOssLayerWeights<TornadoTensor> w = weights.layers();
        UnifiedMemoryWeights.upload(
                layer,
                state.workspace,
                weights.rms_att_weightLayered[l].asFloatArray(),
                weights.wqLayered[l].deviceArray(),
                weights.wkLayered[l].deviceArray(),
                weights.wvLayered[l].deviceArray(),
                weights.woLayered[l].deviceArray(),
                w.queryBias()[l].asFloatArray(),
                w.keyBias()[l].asFloatArray(),
                w.valueBias()[l].asFloatArray(),
                w.outputBias()[l].asFloatArray(),
                w.sinks()[l].asFloatArray(),
                weights.rms_ffn_weightLayered[l].asFloatArray(),
                w.router()[l].asFloatArray(),
                w.routerBias()[l].asFloatArray(),
                experts(w.gateExperts()[l]),
                w.gateExpertsBias()[l].asFloatArray(),
                experts(w.upExperts()[l]),
                w.upExpertsBias()[l].asFloatArray(),
                experts(w.downExperts()[l]),
                w.downExpertsBias()[l].asFloatArray());
        if (l == firstLayer) {
            UnifiedMemoryWeights.upload(
                    layer,
                    state.workspace,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        } else {
            layer.consumeFromDevice(
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        }
    }

    protected Object[] scratch() {
        var ws = gptState.workspace;
        return new Object[] {
            context,
            ws.wrapXb,
            ws.wrapXb2,
            ws.wrapQ,
            ws.wrapK,
            ws.wrapV,
            keyCache(),
            valueCache(),
            ws.wrapRouterLogits,
            ws.wrapSelectedExperts,
            ws.wrapRoutingWeights,
            ws.wrapExpertGate,
            ws.wrapExpertDown,
            ws.wrapAttSplit,
            ws.wrapXbQuants,
            ws.wrapXbScales,
            ws.wrapXbSums
        };
    }

    protected void transfers(TaskGraph layer, int l) {
        var ws = gptState.workspace;
        if (l == firstLayer) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.positionHolder);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch());
        } else {
            layer.consumeFromDevice(scratch());
            layer.consumeFromDevice(ws.positionHolder);
        }
        // Every execution: acquiring or releasing a lease rewrites the block table.
        layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.wrapBlockTable);
    }

    private void attention(TaskGraph layer, int l) {
        var ws = gptState.workspace;
        GptOssLayerWeights<TornadoTensor> w = weights.layers();
        int dim = config.dim();
        layer.task(
                "attn_norm",
                GptOssKernels::normQuantize,
                context,
                ws.wrapX,
                weights.rms_att_weightLayered[l].asFloatArray(),
                ws.wrapXbQuants,
                ws.wrapXbScales,
                dim,
                config.rmsNormEps());
        layer.task(
                "qkv",
                TransformerComputeKernelsQ8_0DP4A::fusedQKVMatmulQ8_0DP4A,
                context,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapQ,
                ws.wrapK,
                ws.wrapV,
                weights.wqLayered[l].asByteArray(),
                weights.wkLayered[l].asByteArray(),
                weights.wvLayered[l].asByteArray(),
                dim,
                config.queryDim(),
                config.kvDim(),
                PROJECTION_LOCAL);
        int window = config.isSlidingWindowLayer(l) ? config.slidingWindow() : 0;
        if (state.usesFp16KeyValueCache()) {
            layer.task(
                    "rope_kv",
                    GptOssKernels::ropeAndCacheFP16,
                    context,
                    ws.positionHolder,
                    ws.wrapQ,
                    ws.wrapK,
                    ws.wrapV,
                    w.queryBias()[l].asFloatArray(),
                    w.keyBias()[l].asFloatArray(),
                    w.valueBias()[l].asFloatArray(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCacheFP16,
                    ws.wrapValueCacheFP16,
                    config.numberOfHeads(),
                    config.numberOfKeyValueHeads(),
                    config.headSize(),
                    config.kvDim(),
                    keyValueLayer(l),
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            if (groupedAttention()) {
                layer.task(
                        "attention",
                        GptOssKernels::attentionGroupSplitFP16,
                        context,
                        ws.positionHolder,
                        ws.wrapQ,
                        ws.wrapKeyCacheFP16,
                        ws.wrapValueCacheFP16,
                        ws.wrapAttSplit,
                        config.numberOfKeyValueHeads(),
                        config.kvDim(),
                        window,
                        keyValueLayer(l),
                        ws.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride);
            } else {
                layer.task(
                        "attention",
                        GptOssKernels::attentionSplitFP16,
                        context,
                        ws.positionHolder,
                        ws.wrapQ,
                        ws.wrapKeyCacheFP16,
                        ws.wrapValueCacheFP16,
                        ws.wrapAttSplit,
                        config.numberOfHeads(),
                        config.kvDim(),
                        config.kvMul(),
                        window,
                        keyValueLayer(l),
                        ws.wrapBlockTable,
                        state.kvBlockCfg,
                        state.kvBlockStride);
            }
        } else {
            layer.task(
                    "rope_kv",
                    GptOssKernels::ropeAndCache,
                    context,
                    ws.positionHolder,
                    ws.wrapQ,
                    ws.wrapK,
                    ws.wrapV,
                    w.queryBias()[l].asFloatArray(),
                    w.keyBias()[l].asFloatArray(),
                    w.valueBias()[l].asFloatArray(),
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    config.numberOfHeads(),
                    config.numberOfKeyValueHeads(),
                    config.headSize(),
                    config.kvDim(),
                    keyValueLayer(l),
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
            layer.task(
                    "attention",
                    GptOssKernels::attentionSplit,
                    context,
                    ws.positionHolder,
                    ws.wrapQ,
                    ws.wrapKeyCache,
                    ws.wrapValueCache,
                    ws.wrapAttSplit,
                    config.numberOfHeads(),
                    config.kvDim(),
                    config.kvMul(),
                    window,
                    keyValueLayer(l),
                    ws.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }
        layer.task(
                "attention_combine",
                GptOssKernels::combineSplitsQuantize,
                context,
                ws.positionHolder,
                ws.wrapAttSplit,
                w.sinks()[l].asFloatArray(),
                ws.wrapXbQuants,
                ws.wrapXbScales,
                config.numberOfHeads(),
                window);
        layer.task(
                "attn_output",
                TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericWithResidualQ8_0DP4A,
                context,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapX,
                weights.woLayered[l].asByteArray(),
                config.queryDim(),
                dim,
                PROJECTION_LOCAL);
    }

    private void experts(TaskGraph layer, int l) {
        var ws = gptState.workspace;
        GptOssLayerWeights<TornadoTensor> w = weights.layers();
        int dim = config.dim();
        layer.task(
                "ffn_norm",
                GptOssKernels::normFusedBias,
                context,
                ws.wrapXb2,
                ws.wrapX,
                w.outputBias()[l].asFloatArray(),
                weights.rms_ffn_weightLayered[l].asFloatArray(),
                dim,
                config.rmsNormEps());
        layer.task(
                "router",
                GptOssKernels::routerRows,
                context,
                ws.wrapXb2,
                w.router()[l].asFloatArray(),
                w.routerBias()[l].asFloatArray(),
                ws.wrapRouterLogits,
                config.numberOfExperts(),
                dim);
        layer.task(
                "topk",
                GptOssKernels::topKSoftmax,
                context,
                ws.wrapRouterLogits,
                ws.wrapSelectedExperts,
                ws.wrapRoutingWeights,
                config.numberOfExperts(),
                config.numberOfExpertsUsed());
        layer.task(
                "experts_gate_up",
                GptOssKernels::expertsGateUpPacked,
                context,
                ws.wrapXb2,
                ws.wrapSelectedExperts,
                experts(w.gateExperts()[l]),
                experts(w.upExperts()[l]),
                w.gateExpertsBias()[l].asFloatArray(),
                w.upExpertsBias()[l].asFloatArray(),
                ws.wrapExpertGate,
                dim,
                config.expertHiddenDim(),
                config.numberOfExpertsUsed());
        layer.task(
                "experts_down",
                GptOssKernels::expertsDownPacked,
                context,
                ws.wrapExpertGate,
                ws.wrapSelectedExperts,
                experts(w.downExperts()[l]),
                w.downExpertsBias()[l].asFloatArray(),
                ws.wrapExpertDown,
                dim,
                config.expertHiddenDim(),
                config.numberOfExpertsUsed());
        layer.task(
                "experts_combine",
                GptOssKernels::expertsCombineBias,
                context,
                ws.wrapExpertDown,
                ws.wrapRoutingWeights,
                w.outputBias()[l].asFloatArray(),
                ws.wrapX,
                dim,
                config.numberOfExpertsUsed());
    }

    /** The key cache in the representation the state allocated. */
    protected Object keyCache() {
        return state.usesFp16KeyValueCache()
                ? gptState.workspace.wrapKeyCacheFP16
                : gptState.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state allocated. */
    protected Object valueCache() {
        return state.usesFp16KeyValueCache()
                ? gptState.workspace.wrapValueCacheFP16
                : gptState.workspace.wrapValueCache;
    }

    /**
     * Whether the decode attention serves a grouped-query group per work-group: an FP16 cache and
     * the group width the kernel is written for.
     */
    protected boolean groupedAttention() {
        return state.usesFp16KeyValueCache() && config.kvMul() == GptOssKernels.GROUP_HEADS;
    }

    /** A stacked MXFP4 expert tensor as the ints the kernels read. */
    static uk.ac.manchester.tornado.api.types.arrays.IntArray experts(TornadoTensor experts) {
        return ((MXFP4TornadoTensor) experts).asIntArray();
    }
}
