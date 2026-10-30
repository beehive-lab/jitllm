package org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.QuantizedBatchProjections;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Batched-prefill Transformer-layer TaskGraphs for Qwen2-MoE Q8_0.
 *
 * <p>A layer that holds any weight other than Q8_0 projects one weight at a time, each by its own
 * type ({@link #nativeLayer}): Q, K, V and the attention output through {@link
 * QuantizedBatchProjections}, every grouped assignment's gate, up and down through {@code
 * Qwen2MoEBatchKernels.groupedRoutedExpertsMatvec*}, and the shared expert's projections through
 * {@link QuantizedBatchProjections} too (tiled; see {@link #batchProjections}), its sigmoid gate
 * folded into the hidden activation so that its down projection adds the gated output to the
 * residual stream.
 */
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
        String graphName = GraphNames.batchPrefillLayer(layerIndex);
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
            layer.consumeFromDevice(GraphNames.PREFILL_ACTIVATION, state.workspace.wrapXBatch);
            if (anyNativeLayer()) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, nativeScratch());
            }
        } else {
            String predecessor = GraphNames.batchPrefillLayer(layerIndex - 1);
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
            if (anyNativeLayer()) {
                layer.consumeFromDevice(predecessor, nativeScratch());
            }
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

        if (nativeLayer(layerIndex) && !q8_0Qkv(layerIndex)) {
            // One projection each: wq, wk and wv can be different representations.
            QuantizedBatchProjections projections = batchProjections();
            projections.projection(
                    layer,
                    "batch_q",
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapQBatch,
                    weights.wqLayered[layerIndex],
                    dim,
                    dim,
                    false);
            projections.projection(
                    layer,
                    "batch_k",
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapKBatch,
                    weights.wkLayered[layerIndex],
                    dim,
                    kvDim,
                    false);
            projections.projection(
                    layer,
                    "batch_v",
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapVBatch,
                    weights.wvLayered[layerIndex],
                    dim,
                    kvDim,
                    false);
        } else {
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
        }

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

        if (nativeLayer(layerIndex)) {
            batchProjections()
                    .projection(
                            layer,
                            "batch_attn_out",
                            state.workspace.wrapXbBatch,
                            state.workspace.wrapXBatch,
                            weights.woLayered[layerIndex],
                            dim,
                            dim,
                            true);
        } else {
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

        if (nativeLayer(layerIndex)) {
            configureNativeExperts(layer, layerIndex);
            return;
        }

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

        configureSharedExpert(layer, layerIndex);
    }

    /** The shared expert over Q8_0 weights: gate/up with SwiGLU, its gate, down and the add. */
    private void configureSharedExpert(TaskGraph layer, int layerIndex) {
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

        configureSharedWeight(layer, layerIndex);

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

    /** The sigmoid gate that scales each token's shared-expert output. */
    private void configureSharedWeight(TaskGraph layer, int layerIndex) {
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
            String prefix = GraphNames.batchPrefillLayer(layer) + ".";
            if (nativeLayer(layer)) {
                addNativeWorkerGrids(scheduler, prefix, layer, qkvWorker);
            }
            scheduler.addWorkerGrid(prefix + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_rms_apply", batchElementWorker);
            if (nativeLayer(layer)) {
                // The rest of the attention's grids and the experts' are registered above; the
                // router's, the top-K and the grouping are shared with an all-Q8_0 layer.
                scheduler.addWorkerGrid(prefix + "batch_qkv_bias", qkvBiasWorker);
                scheduler.addWorkerGrid(prefix + "batch_rope_kv", ropeWorker);
                scheduler.addWorkerGrid(prefix + "batch_attention", attentionWorker);
                scheduler.addWorkerGrid(prefix + "batch_ffn_rms", rmsWorker);
                scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", batchElementWorker);
                scheduler.addWorkerGrid(prefix + "batch_router", routerWorker);
                scheduler.addWorkerGrid(prefix + "batch_topk", batchScalarWorker);
                scheduler.addWorkerGrid(prefix + "batch_group_assignments", groupingWorker);
                scheduler.addWorkerGrid(prefix + "batch_routed_accumulate", batchElementWorker);
                scheduler.addWorkerGrid(prefix + "batch_shared_weight", sharedWeightWorker);
                continue;
            }
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

    // ── Layers that project each weight by its own type ──────────────────────────────────────

    /** Whether a layer holds any projection in a representation other than Q8_0. */
    private boolean nativeLayer(int layerIndex) {
        for (TornadoTensor w :
                new TornadoTensor[] {
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.gateExpertsLayered[layerIndex],
                    weights.upExpertsLayered[layerIndex],
                    weights.downExpertsLayered[layerIndex],
                    weights.sharedGateLayered[layerIndex],
                    weights.sharedUpLayered[layerIndex],
                    weights.sharedDownLayered[layerIndex]
                }) {
            if (w.dataType() != DataType.Q8_0) {
                return true;
            }
        }
        return false;
    }

    private boolean anyNativeLayer() {
        for (int i = 0; i < config.numberOfLayers(); i++) {
            if (nativeLayer(i)) {
                return true;
            }
        }
        return false;
    }

    /** The scratch the per-type layers add: the grouped up outputs and the shared up outputs. */
    private Object[] nativeScratch() {
        return new Object[] {
            state.workspace.wrapGroupedExpertUp, state.workspace.wrapSharedUpBatch
        };
    }

    private boolean q8_0Qkv(int layerIndex) {
        return weights.wqLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.wkLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.wvLayered[layerIndex].dataType() == DataType.Q8_0;
    }

    /** Whether the shared expert's three projections are Q8_0, which keep their batched kernels. */
    private boolean q8_0SharedExpert(int layerIndex) {
        return weights.sharedGateLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.sharedUpLayered[layerIndex].dataType() == DataType.Q8_0
                && weights.sharedDownLayered[layerIndex].dataType() == DataType.Q8_0;
    }

    // @formatter:off
    /**
     * The per-weight batched projections, on the tiled kernels on every backend.
     *
     * <p>Not the SIMD-group matrices on Metal: with them this layer misses the batched parity
     * bounds (Qwen1.5-MoE Q4_0, relL2 2.5e-3 against 9.3e-6 tiled, argmax 0/63), deterministically,
     * from the first decoded row. The kernels themselves match the host at this model's shapes,
     * with NaN rows past the chunk too, and either half of the layer on them alone (Q, K, V and the
     * shared gate and up; or the output and shared down projections) passes. The cause is open.
     */
    // @formatter:on
    private QuantizedBatchProjections batchProjections() {
        return new QuantizedBatchProjections(
                context, state.workspace, batchSize, false, false, LOCAL_WORK_GROUP_SIZE);
    }

    /**
     * Every grouped assignment's gate, up and down by the experts' types, and the shared expert's,
     * after the routing, top-K and grouping an all-Q8_0 layer runs too.
     */
    private void configureNativeExperts(TaskGraph layer, int layerIndex) {
        int hidden = config.moeHiddenDim();
        int shared = config.sharedExpertHiddenDim();
        grouped(
                layer,
                "batch_routed_gate",
                state.workspace.wrapXbBatch,
                1,
                weights.gateExpertsLayered[layerIndex],
                state.workspace.wrapGroupedExpertHidden,
                dim,
                hidden);
        grouped(
                layer,
                "batch_routed_up",
                state.workspace.wrapXbBatch,
                1,
                weights.upExpertsLayered[layerIndex],
                state.workspace.wrapGroupedExpertUp,
                dim,
                hidden);
        layer.task(
                "batch_routed_gate_up",
                TransformerComputeKernelsQ6_K::siluAndMultiply,
                context,
                state.workspace.wrapGroupedExpertHidden,
                state.workspace.wrapGroupedExpertUp,
                numberOfAssignments * hidden);
        grouped(
                layer,
                "batch_routed_down",
                state.workspace.wrapGroupedExpertHidden,
                0,
                weights.downExpertsLayered[layerIndex],
                state.workspace.wrapGroupedExpertDown,
                hidden,
                dim);
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

        if (q8_0SharedExpert(layerIndex)) {
            configureSharedExpert(layer, layerIndex);
            return;
        }
        QuantizedBatchProjections projections = batchProjections();
        projections.projection(
                layer,
                "batch_shared_gate",
                state.workspace.wrapXbBatch,
                state.workspace.wrapSharedHiddenBatch,
                weights.sharedGateLayered[layerIndex],
                dim,
                shared,
                false);
        projections.projection(
                layer,
                "batch_shared_up",
                state.workspace.wrapXbBatch,
                state.workspace.wrapSharedUpBatch,
                weights.sharedUpLayered[layerIndex],
                dim,
                shared,
                false);
        configureSharedWeight(layer, layerIndex);
        layer.task(
                "batch_shared_gate_up",
                Qwen2MoEBatchKernels::batchedSharedExpertSwiGLUGated,
                context,
                state.workspace.wrapSharedHiddenBatch,
                state.workspace.wrapSharedUpBatch,
                state.workspace.wrapSharedWeightBatch,
                state.workspace.activeBatchSizeHolder,
                shared);
        projections.projection(
                layer,
                "batch_shared_down",
                state.workspace.wrapSharedHiddenBatch,
                state.workspace.wrapXBatch,
                weights.sharedDownLayered[layerIndex],
                shared,
                dim,
                true);
    }

    /**
     * {@code out[p * rows + r] = w_e[r] · x[row(p) ...]} for every grouped assignment, through the
     * kernel for {@code w}'s type; {@code byToken} 1 reads each assignment's token, 0 its own row.
     */
    private void grouped(
            TaskGraph layer,
            String name,
            FloatArray x,
            int byToken,
            TornadoTensor w,
            FloatArray out,
            int n,
            int rows) {
        layer.task(
                name,
                switch (w.dataType()) {
                    case Q8_0 -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ8_0;
                    case Q4_0 -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_0;
                    case Q4_1 -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_1;
                    case Q4_K -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ4_K;
                    case Q5_K -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ5_K;
                    case Q6_K -> Qwen2MoEBatchKernels::groupedRoutedExpertsMatvecQ6_K;
                    default ->
                            throw new IllegalStateException(
                                    "no batched routed-expert kernel for a "
                                            + w.dataType()
                                            + " weight");
                },
                context,
                x,
                byToken,
                state.workspace.wrapSelectedExpertsBatch,
                state.workspace.wrapGroupedAssignmentIds,
                state.workspace.activeBatchSizeHolder,
                w.asByteArray(),
                out,
                n,
                rows,
                config.numberOfExperts(),
                topK,
                LOCAL_WORK_GROUP_SIZE);
    }

    /** The grids of a per-type layer's projections and expert tasks. */
    private void addNativeWorkerGrids(
            GridScheduler scheduler, String prefix, int layerIndex, WorkerGrid qkvWorker) {
        int hidden = config.moeHiddenDim();
        int shared = config.sharedExpertHiddenDim();
        QuantizedBatchProjections projections = batchProjections();
        if (q8_0Qkv(layerIndex)) {
            scheduler.addWorkerGrid(prefix + "batch_qkv", qkvWorker);
        } else {
            scheduler.addWorkerGrid(
                    prefix + "batch_q", projections.worker(weights.wqLayered[layerIndex], dim));
            scheduler.addWorkerGrid(
                    prefix + "batch_k", projections.worker(weights.wkLayered[layerIndex], kvDim));
            scheduler.addWorkerGrid(
                    prefix + "batch_v", projections.worker(weights.wvLayered[layerIndex], kvDim));
        }
        scheduler.addWorkerGrid(
                prefix + "batch_attn_out", projections.worker(weights.woLayered[layerIndex], dim));
        scheduler.addWorkerGrid(
                prefix + "batch_routed_gate", groupedRowsWorker(numberOfAssignments * hidden));
        scheduler.addWorkerGrid(
                prefix + "batch_routed_up", groupedRowsWorker(numberOfAssignments * hidden));
        scheduler.addWorkerGrid(
                prefix + "batch_routed_gate_up", elementwise(numberOfAssignments * hidden));
        scheduler.addWorkerGrid(
                prefix + "batch_routed_down", groupedRowsWorker(numberOfAssignments * dim));
        if (q8_0SharedExpert(layerIndex)) {
            scheduler.addWorkerGrid(
                    prefix + "batch_shared_gate_up", groupedRowsWorker(batchSize * shared));
            scheduler.addWorkerGrid(
                    prefix + "batch_shared_down", groupedRowsWorker(batchSize * dim));
            return;
        }
        scheduler.addWorkerGrid(
                prefix + "batch_shared_gate",
                projections.worker(weights.sharedGateLayered[layerIndex], shared));
        scheduler.addWorkerGrid(
                prefix + "batch_shared_up",
                projections.worker(weights.sharedUpLayered[layerIndex], shared));
        scheduler.addWorkerGrid(prefix + "batch_shared_gate_up", elementwise(batchSize * shared));
        scheduler.addWorkerGrid(
                prefix + "batch_shared_down",
                projections.worker(weights.sharedDownLayered[layerIndex], dim));
    }

    /** One lane per element, in workgroups of 256. */
    private static WorkerGrid elementwise(int elements) {
        return WorkerGridFactory.genericWorker((elements + 255) / 256 * 256, 256);
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
