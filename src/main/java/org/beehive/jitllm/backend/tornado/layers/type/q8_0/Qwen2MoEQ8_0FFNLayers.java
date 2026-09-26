package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Single-token Q8_0 TaskGraphs for Qwen2-MoE / Qwen1.5-MoE.
 *
 * <p>The attention block follows Qwen2. Its dense FFN is replaced by the routed-expert pipeline:
 * normalize, route, choose top-K experts, execute each selected expert, and accumulate its weighted
 * output into {@code wrapX}.
 */
public class Qwen2MoEQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen2MoETornadoWeights, Qwen2MoEConfiguration> {

    protected final Qwen2MoEState moeState;

    public Qwen2MoEQ8_0FFNLayers(
            String taskGraphName,
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.moeState = state;
        setupFFNLayers();
    }

    /** Sets the GPU worker grid for each task in each Transformer layer. */
    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsNormWorker =
                WorkerGridFactory.createRmsNormWorker(moeState.localSize, moeState.localSize);

        WorkerGrid qkvWorker = workerForRows(config.dim() + 2 * config.kvDim());
        WorkerGrid qkvBiasWorker = new WorkerGrid1D(config.dim());
        qkvBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qkvBiasWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid2D(config.numberOfHeads(), config.headSize() / 2);
        ropeWorker.setGlobalWork(config.numberOfHeads(), config.headSize() / 2, 1);
        ropeWorker.setLocalWork(1, 1, 1);

        int attentionLocalSize = Math.min(config.headSize(), 64);
        WorkerGrid attentionWorker = new WorkerGrid1D(config.numberOfHeads() * attentionLocalSize);
        attentionWorker.setLocalWork(attentionLocalSize, 1, 1);

        WorkerGrid dimElementWorker = new WorkerGrid1D(config.dim());
        dimElementWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        WorkerGrid dimWorker = workerForRows(config.dim());
        WorkerGrid routerWorker = workerForRows(config.numberOfExperts());
        WorkerGrid topKWorker = new WorkerGrid1D(LOCAL_WORK_GROUP_SIZE_ALLOC);
        topKWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        // Fused routed-expert launch: the slot index is folded into the work-group id.
        WorkerGrid allExpertsHiddenWorker =
                workerForRows(config.moeHiddenDim() * config.numberOfExpertsUsed());
        WorkerGrid sharedHiddenWorker = workerForRows(config.sharedExpertHiddenDim());

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "layer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsNormWorker);
            scheduler.addWorkerGrid(prefix + "attn_rms_qkv_projection", qkvWorker);
            scheduler.addWorkerGrid(prefix + "fused_qkv_bias", qkvBiasWorker);
            scheduler.addWorkerGrid(prefix + "rope_and_kv_cache", ropeWorker);
            scheduler.addWorkerGrid(prefix + "attention", attentionWorker);
            scheduler.addWorkerGrid(prefix + "attn_output_proj", dimWorker);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsNormWorker);
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", dimElementWorker);
            scheduler.addWorkerGrid(prefix + "router_projection", routerWorker);
            scheduler.addWorkerGrid(prefix + "router_softmax_topk", topKWorker);
            scheduler.addWorkerGrid(prefix + "routed_experts_gate_up", allExpertsHiddenWorker);
            scheduler.addWorkerGrid(prefix + "routed_experts_down", dimWorker);
            scheduler.addWorkerGrid(prefix + "shared_expert_gate_up", sharedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "shared_expert_down", dimWorker);
            scheduler.addWorkerGrid(prefix + "shared_expert_gate_and_accumulate", topKWorker);
        }
        return scheduler;
    }

    /** Creates a grid where one GPU work-group computes one output row. */
    private WorkerGrid workerForRows(int rows) {
        WorkerGrid worker = new WorkerGrid1D(rows * LOCAL_WORK_GROUP_SIZE_ALLOC);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        return worker;
    }

    /**
     * Creates the complete GPU TaskGraph for one Transformer layer. {@code layerIndex} selects that
     * layer's weights.
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph("layer_" + layerIndex);
        // Reuse wrapX produced by the previous TaskGraph on the GPU.
        String predecessor = predecessorGraphName(layerIndex);
        if (predecessor == null) {
            layer.consumeFromDevice(moeState.workspace.wrapX);
        } else {
            layer.consumeFromDevice(predecessor, moeState.workspace.wrapX);
        }
        layer = configureLayerWeights(layer, layerIndex);
        layer = configureLayerDataTransfers(layer, layerIndex);

        configureAttention(layer, layerIndex);
        configureRoutedExperts(layer, layerIndex);
        layer.persistOnDevice(
                moeState.workspace.wrapX,
                moeState.workspace.wrapKeyCache,
                moeState.workspace.wrapValueCache);
        return layer;
    }

    /** Returns an explicit predecessor for plans that connect multiple graph chains. */
    protected String predecessorGraphName(int layerIndex) {
        return null;
    }

    /** Uploads this layer's read-only weights on its first execution. */
    protected TaskGraph configureLayerWeights(TaskGraph layer, int layerIndex) {
        return layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
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
                weights.gateExpertsLayered[layerIndex].asRepackedByteArray(),
                weights.upExpertsLayered[layerIndex].asRepackedByteArray(),
                weights.downExpertsLayered[layerIndex].asRepackedByteArray(),
                weights.sharedGateLayered[layerIndex].asRepackedByteArray(),
                weights.sharedUpLayered[layerIndex].asRepackedByteArray(),
                weights.sharedDownLayered[layerIndex].asRepackedByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
    }

    /** Adds the normal Qwen2 attention tasks to this layer's TaskGraph. */
    private void configureAttention(TaskGraph layer, int layerIndex) {
        layer.task(
                "attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayerSingleGroup,
                context,
                moeState.workspace.temp,
                moeState.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                moeState.localSize);

        layer.task(
                "attn_rms_qkv_projection",
                Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                context,
                moeState.workspace.wrapX,
                moeState.workspace.wrapQ,
                moeState.workspace.wrapK,
                moeState.workspace.wrapV,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                moeState.workspace.temp,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                config.dim(),
                config.dim(),
                config.kvDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "fused_qkv_bias",
                TransformerComputeKernelsLayered::fusedQKvBiasAddition,
                context,
                moeState.workspace.wrapQ,
                moeState.workspace.wrapK,
                weights.q_biasLayered[layerIndex].asFloatArray(),
                moeState.workspace.wrapV,
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                config.dim(),
                config.kvDim());

        layer.task(
                "rope_and_kv_cache",
                Qwen3PagedKvKernels::ropeRotationWithCacheCopyPaged,
                context,
                moeState.workspace.positionHolder,
                moeState.workspace.wrapQ,
                moeState.workspace.wrapK,
                moeState.workspace.wrapV,
                moeState.workspace.wrapKeyCache,
                moeState.workspace.wrapValueCache,
                config.ropeTheta(),
                config.numberOfKeyValueHeads(),
                config.headSize(),
                config.kvDim(),
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride);

        layer.task(
                "attention",
                Qwen2PagedKvKernels::processHeadsFlashAttentionPaged,
                context,
                moeState.workspace.wrapQ,
                moeState.workspace.wrapKeyCache,
                moeState.workspace.wrapValueCache,
                moeState.workspace.wrapXb,
                config.numberOfHeads(),
                config.headSize(),
                config.kvDim(),
                config.kvMul(),
                moeState.workspace.positionHolder,
                layerIndex,
                state.workspace.wrapBlockTable,
                state.kvBlockCfg,
                state.kvBlockStride);

        layer.task(
                "attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                moeState.workspace.wrapXb,
                moeState.workspace.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);
    }

    /**
     * Adds router, top-K, and selected-expert FFN tasks to this layer's TaskGraph. Their weighted
     * outputs are added to the residual vector.
     */
    private void configureRoutedExperts(TaskGraph layer, int layerIndex) {
        layer.task(
                "ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayerSingleGroup,
                context,
                moeState.workspace.tempFFN,
                moeState.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                moeState.localSize);

        layer.task(
                "ffn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                moeState.workspace.wrapXb,
                moeState.workspace.wrapX,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                moeState.workspace.tempFFN);

        layer.task(
                "router_projection",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                moeState.workspace.wrapXb,
                moeState.workspace.wrapRouterLogits,
                weights.routerGateLayered[layerIndex].asFloatArray(),
                config.dim(),
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "router_softmax_topk",
                Qwen2MoEKernels::softmaxAndTopK,
                context,
                moeState.workspace.wrapRouterLogits,
                moeState.workspace.wrapSelectedExperts,
                moeState.workspace.wrapRoutingWeights,
                config.numberOfExperts(),
                config.numberOfExpertsUsed());

        // All routed slots in two launches instead of two per slot: at top-4 this is 2 kernel
        // launches per layer rather than 8, and the residual is accumulated once instead of
        // four times.
        layer.task(
                "routed_experts_gate_up",
                Qwen2MoEKernels::fusedRoutedExpertsGateUpSwiGLUQ8_0,
                context,
                moeState.workspace.wrapXb,
                moeState.workspace.wrapSelectedExperts,
                config.numberOfExpertsUsed(),
                weights.gateExpertsLayered[layerIndex].asRepackedByteArray(),
                weights.upExpertsLayered[layerIndex].asRepackedByteArray(),
                moeState.workspace.wrapExpertGate,
                config.dim(),
                config.moeHiddenDim(),
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "routed_experts_down",
                Qwen2MoEKernels::routedExpertsDownProjectAndAccumulateQ8_0,
                context,
                moeState.workspace.wrapExpertGate,
                moeState.workspace.wrapX,
                moeState.workspace.wrapSelectedExperts,
                moeState.workspace.wrapRoutingWeights,
                config.numberOfExpertsUsed(),
                weights.downExpertsLayered[layerIndex].asRepackedByteArray(),
                config.dim(),
                config.moeHiddenDim(),
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // The shared expert always runs; it does not depend on router top-K selection.
        layer.task(
                "shared_expert_gate_up",
                Qwen2MoEKernels::sharedExpertGateUpSwiGLUQ8_0,
                context,
                moeState.workspace.wrapXb,
                weights.sharedGateLayered[layerIndex].asRepackedByteArray(),
                weights.sharedUpLayered[layerIndex].asRepackedByteArray(),
                moeState.workspace.wrapSharedGate,
                config.dim(),
                config.sharedExpertHiddenDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "shared_expert_down",
                Qwen2MoEKernels::sharedExpertDownProjectQ8_0,
                context,
                moeState.workspace.wrapSharedGate,
                weights.sharedDownLayered[layerIndex].asRepackedByteArray(),
                moeState.workspace.wrapSharedOutput,
                config.dim(),
                config.sharedExpertHiddenDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "shared_expert_gate_and_accumulate",
                Qwen2MoEKernels::sharedExpertGateAndAccumulate,
                context,
                moeState.workspace.wrapXb,
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                moeState.workspace.wrapSharedOutput,
                moeState.workspace.wrapX,
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);
    }

    /** Configures which TaskGraph data is uploaded from the CPU or reused on the GPU. */
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    moeState.workspace.positionHolder,
                    moeState.workspace.temp,
                    moeState.workspace.tempFFN);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    moeState.workspace.wrapXb,
                    moeState.workspace.wrapXb2,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    moeState.workspace.wrapKeyCache,
                    moeState.workspace.wrapValueCache,
                    moeState.workspace.wrapAtt,
                    moeState.workspace.wrapRouterLogits,
                    moeState.workspace.wrapSelectedExperts,
                    moeState.workspace.wrapRoutingWeights,
                    moeState.workspace.wrapExpertGate,
                    moeState.workspace.wrapSharedGate,
                    moeState.workspace.wrapSharedOutput);
            // EVERY_EXECUTION, not once: acquiring or releasing a lease rewrites the table,
            // and a stale block index is still a valid index, so a table uploaded once leaves
            // the kernels reading a mapping that no longer exists, silently.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            layer.consumeFromDevice(
                    context,
                    moeState.workspace.wrapXb,
                    moeState.workspace.wrapXb2,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    moeState.workspace.wrapKeyCache,
                    moeState.workspace.wrapValueCache,
                    moeState.workspace.wrapAtt,
                    moeState.workspace.wrapRouterLogits,
                    moeState.workspace.wrapSelectedExperts,
                    moeState.workspace.wrapRoutingWeights,
                    moeState.workspace.wrapExpertGate,
                    moeState.workspace.wrapSharedGate,
                    moeState.workspace.wrapSharedOutput,
                    moeState.workspace.positionHolder);
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        }
        return layer;
    }
}
