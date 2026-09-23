package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/** Batched-prefill Transformer-layer TaskGraphs for Qwen2-MoE Q8_0. */
public final class Qwen2MoEQ8_0LayersBatchPrefill
        implements BatchPrefillTransformerLayerTaskGraphs {

    private static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final Qwen2MoEState state;
    private final Qwen2MoETornadoWeights weights;
    private final Qwen2MoEConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int dim;
    private final int kvDim;
    private final int topK;
    private final int numberOfAssignments;
    private final List<ImmutableTaskGraph> layerTaskGraphs;
    private String lastLayerTaskGraphID;

    public Qwen2MoEQ8_0LayersBatchPrefill(
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.dim = config.dim();
        this.kvDim = config.kvDim();
        this.topK = config.numberOfExpertsUsed();
        this.numberOfAssignments = batchSize * topK;
        this.layerTaskGraphs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    /** Creates the complete batch-prefill TaskGraph for one Transformer layer. */
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }

        TaskGraph layer = new TaskGraph(graphName);
        configureDataTransfers(layer, layerIndex);
        configureAttention(layer, layerIndex);
        configureMoE(layer, layerIndex);
        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.routerGateLayered[layerIndex].asFloatArray(),
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
        return layer;
    }

    /** Declares layer weights and the batch buffers that remain on the GPU. */
    private void configureDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.batchStartPosHolder,
                    state.workspace.activeBatchSizeHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapRouterLogitsBatch,
                    state.workspace.wrapSelectedExpertsBatch,
                    state.workspace.wrapRoutingWeightsBatch,
                    state.workspace.wrapGroupedAssignmentIds,
                    state.workspace.wrapGroupedPositionByAssignment,
                    state.workspace.wrapGroupedExpertHidden,
                    state.workspace.wrapGroupedExpertDown,
                    state.workspace.wrapSharedHiddenBatch,
                    state.workspace.wrapSharedWeightBatch);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String predecessor = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.batchStartPosHolder,
                    state.workspace.activeBatchSizeHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapRouterLogitsBatch,
                    state.workspace.wrapSelectedExpertsBatch,
                    state.workspace.wrapRoutingWeightsBatch,
                    state.workspace.wrapGroupedAssignmentIds,
                    state.workspace.wrapGroupedPositionByAssignment,
                    state.workspace.wrapGroupedExpertHidden,
                    state.workspace.wrapGroupedExpertDown,
                    state.workspace.wrapSharedHiddenBatch,
                    state.workspace.wrapSharedWeightBatch);
            layer.consumeFromDevice(predecessor, state.workspace.wrapBlockTable);
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.routerGateLayered[layerIndex].asFloatArray(),
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
    }

    /** Adds Qwen2 attention tasks for every token in the prefill batch. */
    private void configureAttention(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        layer.task(
                "batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim,
                kvDim,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_qkv_bias",
                Qwen2MoEBatchKernels::batchedQKVBias,
                context,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                state.workspace.activeBatchSizeHolder,
                dim,
                kvDim);

        layer.task(
                "batch_rope_kv",
                Qwen2PagedKvKernels::batchedRopeWithKVCacheQwen2Paged,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.activeBatchSizeHolder,
                state.workspace.wrapQBatch,
                state.workspace.wrapKBatch,
                state.workspace.wrapVBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                kvDim,
                config.headSize(),
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                dim,
                config.ropeTheta());

        layer.task(
                "batch_attention",
                TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionPaged,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapQBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                state.workspace.wrapXbBatch,
                config.numberOfHeads(),
                config.headSize(),
                kvDim,
                config.kvMul(),
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride,
                dim);

        layer.task(
                "batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim,
                dim,
                LOCAL_WORK_GROUP_SIZE);
    }

    /** Adds batched routing, routed experts, and the shared expert. */
    private void configureMoE(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        layer.task(
                "batch_router",
                Qwen2MoEBatchKernels::batchedRouterProjection,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapRouterLogitsBatch,
                weights.routerGateLayered[layerIndex].asFloatArray(),
                state.workspace.activeBatchSizeHolder,
                dim,
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_topk",
                Qwen2MoEBatchKernels::batchedSoftmaxAndTopK,
                context,
                state.workspace.wrapRouterLogitsBatch,
                state.workspace.wrapSelectedExpertsBatch,
                state.workspace.wrapRoutingWeightsBatch,
                state.workspace.activeBatchSizeHolder,
                config.numberOfExperts(),
                topK);

        layer.task(
                "batch_group_assignments",
                Qwen2MoEBatchKernels::groupAssignmentsByExpert,
                context,
                state.workspace.wrapSelectedExpertsBatch,
                state.workspace.wrapGroupedAssignmentIds,
                state.workspace.wrapGroupedPositionByAssignment,
                state.workspace.activeBatchSizeHolder,
                config.numberOfExperts(),
                topK);

        layer.task(
                "batch_routed_gate_up",
                Qwen2MoEBatchKernels::groupedRoutedExpertsGateUpSwiGLUQ8_0,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapSelectedExpertsBatch,
                state.workspace.wrapGroupedAssignmentIds,
                state.workspace.activeBatchSizeHolder,
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                state.workspace.wrapGroupedExpertHidden,
                dim,
                config.moeHiddenDim(),
                config.numberOfExperts(),
                topK,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_routed_down",
                Qwen2MoEBatchKernels::groupedRoutedExpertsDownQ8_0,
                context,
                state.workspace.wrapGroupedExpertHidden,
                state.workspace.wrapSelectedExpertsBatch,
                state.workspace.wrapGroupedAssignmentIds,
                state.workspace.activeBatchSizeHolder,
                weights.downExpertsLayered[layerIndex].asByteArray(),
                state.workspace.wrapGroupedExpertDown,
                dim,
                config.moeHiddenDim(),
                config.numberOfExperts(),
                topK,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_routed_accumulate",
                Qwen2MoEBatchKernels::accumulateGroupedRoutedExperts,
                context,
                state.workspace.wrapGroupedExpertDown,
                state.workspace.wrapGroupedPositionByAssignment,
                state.workspace.wrapRoutingWeightsBatch,
                state.workspace.wrapXBatch,
                state.workspace.activeBatchSizeHolder,
                dim,
                topK);

        layer.task(
                "batch_shared_gate_up",
                Qwen2MoEBatchKernels::batchedSharedExpertGateUpSwiGLUQ8_0,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.activeBatchSizeHolder,
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                state.workspace.wrapSharedHiddenBatch,
                dim,
                config.sharedExpertHiddenDim(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_shared_weight",
                Qwen2MoEBatchKernels::batchedSharedExpertGateWeight,
                context,
                state.workspace.wrapXbBatch,
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                state.workspace.wrapSharedWeightBatch,
                state.workspace.activeBatchSizeHolder,
                dim,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_shared_down",
                Qwen2MoEBatchKernels::batchedSharedExpertDownAndAccumulateQ8_0,
                context,
                state.workspace.wrapSharedHiddenBatch,
                state.workspace.wrapSharedWeightBatch,
                state.workspace.activeBatchSizeHolder,
                weights.sharedDownLayered[layerIndex].asByteArray(),
                state.workspace.wrapXBatch,
                dim,
                config.sharedExpertHiddenDim(),
                LOCAL_WORK_GROUP_SIZE);
    }

    /** Configures the fixed WorkerGrid used by every task in every layer. */
    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsWorker = groupedRowsWorker(batchSize);
        WorkerGrid batchScalarWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid batchElementWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker = groupedRowsWorker(batchSize * (dim + 2 * kvDim));
        int qkvBiasGlobalWork = batchSize * (dim + 2 * kvDim);
        WorkerGrid qkvBiasWorker =
                WorkerGridFactory.genericWorker(
                        qkvBiasGlobalWork, validLocalSize(qkvBiasGlobalWork, 256));

        int ropeGlobalWork = batchSize * (dim / 2);
        int ropeLocalWork = validLocalSize(ropeGlobalWork, 512);
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobalWork, ropeLocalWork);

        int attentionLocalWork = validLocalSize(config.headSize(), 64);
        WorkerGrid attentionWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * attentionLocalWork,
                        attentionLocalWork);

        WorkerGrid batchDimRowsWorker = groupedRowsWorker(batchSize * dim);
        WorkerGrid routerWorker = groupedRowsWorker(batchSize * config.numberOfExperts());
        WorkerGrid groupingWorker = WorkerGridFactory.createSingleWorker();
        WorkerGrid routedHiddenWorker =
                groupedRowsWorker(numberOfAssignments * config.moeHiddenDim());
        WorkerGrid routedDownWorker = groupedRowsWorker(numberOfAssignments * dim);
        WorkerGrid sharedHiddenWorker =
                groupedRowsWorker(batchSize * config.sharedExpertHiddenDim());
        WorkerGrid sharedWeightWorker = groupedRowsWorker(batchSize);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "batchPrefillLayer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_rms_apply", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv_bias", qkvBiasWorker);
            scheduler.addWorkerGrid(prefix + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(prefix + "batch_attention", attentionWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_out", batchDimRowsWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_router", routerWorker);
            scheduler.addWorkerGrid(prefix + "batch_topk", batchScalarWorker);
            scheduler.addWorkerGrid(prefix + "batch_group_assignments", groupingWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_gate_up", routedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_down", routedDownWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_accumulate", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_gate_up", sharedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_weight", sharedWeightWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_down", batchDimRowsWorker);
        }
    }

    /** Creates a 32-thread work-group for every logical output row. */
    private static WorkerGrid groupedRowsWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
    }

    /** Finds a legal local size that divides the requested global dimension. */
    private static int validLocalSize(int size, int maximum) {
        int localSize = Math.min(size, maximum);
        while (localSize > 1 && size % localSize != 0) {
            localSize--;
        }
        return localSize;
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerTaskGraphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
