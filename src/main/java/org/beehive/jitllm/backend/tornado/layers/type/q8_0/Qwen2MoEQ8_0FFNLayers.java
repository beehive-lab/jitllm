package org.beehive.jitllm.backend.tornado.layers.type.q8_0;

import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen3PagedKvKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.QuantizedProjections;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Single-token Q8_0 TaskGraphs for Qwen2-MoE / Qwen1.5-MoE.
 *
 * <p>The attention block follows Qwen2. Its dense FFN is replaced by the routed-expert pipeline:
 * normalize, route, choose top-K experts, execute each selected expert, and accumulate its weighted
 * output into {@code wrapX}.
 *
 * <p>A layer that holds any weight other than Q8_0, as a Q4_0 or Q6_K file's do, projects one
 * weight at a time, each by its own type ({@link #nativeLayer}): the norm applied once, Q, K and V
 * apart, every routed slot's gate, up and down through {@code
 * Qwen2MoEKernels.routedExpertsMatvec*}, the shared expert through {@link QuantizedProjections},
 * and one task that adds the routed and shared outputs to the residual. All in floating point:
 * these graphs bind no quantized activation.
 */
public class Qwen2MoEQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen2MoETornadoWeights, Qwen2MoEConfiguration> {

    protected final Qwen2MoEState moeState;

    public Qwen2MoEQ8_0FFNLayers(
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
        this.moeState = state;
        this.projections =
                new QuantizedProjections(context, state.workspace, LOCAL_WORK_GROUP_SIZE_ALLOC);
        setupFFNLayers();
    }

    /** The per-weight projections of a layer that is not entirely Q8_0. */
    private final QuantizedProjections projections;

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
        WorkerGrid attentionWorker =
                WorkerGridFactory.genericWorker(
                        config.numberOfHeads() * attentionLocalSize, attentionLocalSize);

        WorkerGrid dimElementWorker =
                WorkerGridFactory.genericWorker(config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid dimWorker = workerForRows(config.dim());
        WorkerGrid routerWorker = workerForRows(config.numberOfExperts());
        WorkerGrid topKWorker =
                WorkerGridFactory.genericWorker(
                        LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        // Fused routed-expert launch: the slot index is folded into the work-group id.
        WorkerGrid allExpertsHiddenWorker =
                workerForRows(config.moeHiddenDim() * config.numberOfExpertsUsed());
        WorkerGrid sharedHiddenWorker = workerForRows(config.sharedExpertHiddenDim());

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = GraphNames.layer(layer) + ".";
            if (nativeLayer(layer)) {
                addNativeWorkerGrids(scheduler, prefix, rmsNormWorker, ropeWorker, attentionWorker);
                continue;
            }
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
        WorkerGrid worker =
                WorkerGridFactory.genericWorker(
                        rows * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        return worker;
    }

    /**
     * Creates the complete GPU TaskGraph for one Transformer layer. {@code layerIndex} selects that
     * layer's weights.
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph(GraphNames.layer(layerIndex));
        // Reuse wrapX produced by the previous TaskGraph on the GPU.
        String predecessor = predecessorGraphName(layerIndex);
        if (predecessor == null) {
            layer.consumeFromDevice(moeState.workspace.wrapX);
        } else {
            layer.consumeFromDevice(predecessor, moeState.workspace.wrapX);
        }
        layer = configureLayerWeights(layer, layerIndex);
        layer = configureLayerDataTransfers(layer, layerIndex);

        if (nativeLayer(layerIndex)) {
            configureNativeAttention(layer, layerIndex);
            configureNativeExperts(layer, layerIndex);
        } else {
            configureAttention(layer, layerIndex);
            configureRoutedExperts(layer, layerIndex);
        }
        layer.persistOnDevice(moeState.workspace.wrapX, keyCache(), valueCache());
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
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
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

        configureRopeAndAttention(layer, layerIndex);

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

    /** RoPE with the key/value cache write, and the attention, in the cache's representation. */
    private void configureRopeAndAttention(TaskGraph layer, int layerIndex) {
        if (useFp16KVCache()) {
            layer.task(
                    "rope_and_kv_cache",
                    Qwen3PagedKvKernels::ropeRotationWithCacheCopyFP16Paged,
                    context,
                    moeState.workspace.positionHolder,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    moeState.workspace.wrapKeyCacheFP16,
                    moeState.workspace.wrapValueCacheFP16,
                    config.ropeTheta(),
                    config.numberOfKeyValueHeads(),
                    config.headSize(),
                    config.kvDim(),
                    layerIndex,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
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
        }

        if (useFp16KVCache()) {
            layer.task(
                    "attention",
                    Qwen2PagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapKeyCacheFP16,
                    moeState.workspace.wrapValueCacheFP16,
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
        } else {
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
        }
    }

    /** The key cache in the representation the state holds: FP16 where it allocated one. */
    protected Object keyCache() {
        return useFp16KVCache()
                ? moeState.workspace.wrapKeyCacheFP16
                : moeState.workspace.wrapKeyCache;
    }

    /** The value cache in the representation the state holds. */
    protected Object valueCache() {
        return useFp16KVCache()
                ? moeState.workspace.wrapValueCacheFP16
                : moeState.workspace.wrapValueCache;
    }

    /**
     * Adds router, top-K, and selected-expert FFN tasks to this layer's TaskGraph. Their weighted
     * outputs are added to the residual vector.
     */
    private void configureRoutedExperts(TaskGraph layer, int layerIndex) {
        configureRouter(layer, layerIndex);

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
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
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
                weights.downExpertsLayered[layerIndex].asByteArray(),
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
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                moeState.workspace.wrapSharedGate,
                config.dim(),
                config.sharedExpertHiddenDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task(
                "shared_expert_down",
                Qwen2MoEKernels::sharedExpertDownProjectQ8_0,
                context,
                moeState.workspace.wrapSharedGate,
                weights.sharedDownLayered[layerIndex].asByteArray(),
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

    /** The FFN norm into {@code wrapXb}, the router scores, and the softmax with top-K. */
    private void configureRouter(TaskGraph layer, int layerIndex) {
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
                    keyCache(),
                    valueCache(),
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
            bindNativeScratch(layer, layerIndex, null);
        } else {
            layer.consumeFromDevice(
                    context,
                    moeState.workspace.wrapXb,
                    moeState.workspace.wrapXb2,
                    moeState.workspace.wrapQ,
                    moeState.workspace.wrapK,
                    moeState.workspace.wrapV,
                    keyCache(),
                    valueCache(),
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
            bindNativeScratch(layer, layerIndex, null);
        }
        return layer;
    }

    // ── Layers that project each weight by its own type ──────────────────────────────────────

    /**
     * Whether a layer holds any projection in a representation other than Q8_0. Such a layer
     * projects one weight at a time, each through the kernel for its own type; an all-Q8_0 layer
     * keeps its fused kernels, task names and grids.
     */
    protected boolean nativeLayer(int layerIndex) {
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

    /**
     * Binds the scratch the per-type layers write, where any layer is one: the routed up and
     * per-slot down outputs and the shared up output, uploaded by layer 0 and passed along the
     * chain ({@code predecessor} names the graph before, or {@code null} for the previous one).
     */
    protected void bindNativeScratch(TaskGraph layer, int layerIndex, String predecessor) {
        boolean any = false;
        for (int i = 0; i < config.numberOfLayers(); i++) {
            any |= nativeLayer(i);
        }
        if (!any) {
            return;
        }
        Object[] scratch = {
            moeState.workspace.wrapExpertUp,
            moeState.workspace.wrapExpertDown,
            moeState.workspace.wrapSharedUp
        };
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch);
        } else if (predecessor == null) {
            layer.consumeFromDevice(scratch);
        } else {
            layer.consumeFromDevice(predecessor, scratch);
        }
    }

    /** The attention block, the norm applied once and Q, K, V and the output each by its type. */
    private void configureNativeAttention(TaskGraph layer, int layerIndex) {
        layer.task(
                "attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayerSingleGroup,
                context,
                moeState.workspace.temp,
                moeState.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                moeState.localSize);
        projections.rmsApply(
                layer,
                "attn_rms_apply",
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                moeState.workspace.temp);
        projections.matvec(
                layer,
                "q_proj",
                moeState.workspace.wrapQ,
                weights.wqLayered[layerIndex],
                config.dim(),
                config.dim());
        projections.matvec(
                layer,
                "k_proj",
                moeState.workspace.wrapK,
                weights.wkLayered[layerIndex],
                config.dim(),
                config.kvDim());
        projections.matvec(
                layer,
                "v_proj",
                moeState.workspace.wrapV,
                weights.wvLayered[layerIndex],
                config.dim(),
                config.kvDim());
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
        configureRopeAndAttention(layer, layerIndex);
        projections.residualMatvec(
                layer,
                "attn_out_quantize",
                "attn_output_proj",
                moeState.workspace.wrapXb,
                weights.woLayered[layerIndex],
                config.dim(),
                config.dim());
    }

    /**
     * The mixture of experts: the norm and router as before; every routed slot's gate, up and down,
     * and the shared expert's, each by its own type; then one task that adds the weighted slots and
     * the gated shared expert to the residual.
     */
    private void configureNativeExperts(TaskGraph layer, int layerIndex) {
        int dim = config.dim();
        int hidden = config.moeHiddenDim();
        int used = config.numberOfExpertsUsed();
        int shared = config.sharedExpertHiddenDim();
        configureRouter(layer, layerIndex);
        routed(
                layer,
                "routed_experts_gate",
                moeState.workspace.wrapXb,
                0,
                weights.gateExpertsLayered[layerIndex],
                moeState.workspace.wrapExpertGate,
                dim,
                hidden);
        routed(
                layer,
                "routed_experts_up",
                moeState.workspace.wrapXb,
                0,
                weights.upExpertsLayered[layerIndex],
                moeState.workspace.wrapExpertUp,
                dim,
                hidden);
        layer.task(
                "routed_experts_gate_up",
                TransformerComputeKernelsQ6_K::siluAndMultiply,
                context,
                moeState.workspace.wrapExpertGate,
                moeState.workspace.wrapExpertUp,
                used * hidden);
        routed(
                layer,
                "routed_experts_down",
                moeState.workspace.wrapExpertGate,
                hidden,
                weights.downExpertsLayered[layerIndex],
                moeState.workspace.wrapExpertDown,
                hidden,
                dim);

        projections.matvec(
                layer,
                "shared_expert_gate",
                moeState.workspace.wrapSharedGate,
                weights.sharedGateLayered[layerIndex],
                dim,
                shared);
        projections.matvec(
                layer,
                "shared_expert_up",
                moeState.workspace.wrapSharedUp,
                weights.sharedUpLayered[layerIndex],
                dim,
                shared);
        layer.task(
                "shared_expert_gate_up",
                TransformerComputeKernelsQ6_K::siluAndMultiply,
                context,
                moeState.workspace.wrapSharedGate,
                moeState.workspace.wrapSharedUp,
                shared);
        projections.floatMatvec(
                layer,
                "shared_expert_down",
                moeState.workspace.wrapSharedGate,
                moeState.workspace.wrapSharedOutput,
                weights.sharedDownLayered[layerIndex],
                shared,
                dim);

        layer.task(
                "shared_expert_gate_and_accumulate",
                Qwen2MoEKernels::routedAndSharedExpertsAccumulate,
                context,
                moeState.workspace.wrapXb,
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                moeState.workspace.wrapSharedOutput,
                moeState.workspace.wrapExpertDown,
                moeState.workspace.wrapSelectedExperts,
                moeState.workspace.wrapRoutingWeights,
                moeState.workspace.wrapX,
                used,
                config.numberOfExperts(),
                dim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);
    }

    /**
     * {@code out[slot * rows + r] = w_e[r] · x[slot * xStride ...]} for every routed slot, through
     * the kernel for {@code w}'s type.
     */
    private void routed(
            TaskGraph layer,
            String name,
            FloatArray x,
            int xStride,
            TornadoTensor w,
            FloatArray out,
            int n,
            int rows) {
        layer.task(
                name,
                switch (w.dataType()) {
                    case Q8_0 -> Qwen2MoEKernels::routedExpertsMatvecQ8_0;
                    case Q4_0 -> Qwen2MoEKernels::routedExpertsMatvecQ4_0;
                    case Q4_1 -> Qwen2MoEKernels::routedExpertsMatvecQ4_1;
                    case Q4_K -> Qwen2MoEKernels::routedExpertsMatvecQ4_K;
                    case Q5_K -> Qwen2MoEKernels::routedExpertsMatvecQ5_K;
                    case Q6_K -> Qwen2MoEKernels::routedExpertsMatvecQ6_K;
                    default ->
                            throw new IllegalStateException(
                                    "no routed-expert kernel for a " + w.dataType() + " weight");
                },
                context,
                x,
                xStride,
                moeState.workspace.wrapSelectedExperts,
                config.numberOfExpertsUsed(),
                w.asByteArray(),
                out,
                n,
                rows,
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);
    }

    /** The grids of a layer that projects one weight at a time. */
    private void addNativeWorkerGrids(
            GridScheduler scheduler,
            String prefix,
            WorkerGrid rmsNormWorker,
            WorkerGrid ropeWorker,
            WorkerGrid attentionWorker) {
        int dim = config.dim();
        int used = config.numberOfExpertsUsed();
        int hidden = config.moeHiddenDim();
        int shared = config.sharedExpertHiddenDim();
        WorkerGrid qkvBiasWorker = new WorkerGrid1D(dim);
        qkvBiasWorker.setGlobalWork(dim, 1, 1);
        qkvBiasWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsNormWorker);
        scheduler.addWorkerGrid(prefix + "attn_rms_apply", projections.rmsApplyWorker(dim));
        scheduler.addWorkerGrid(prefix + "q_proj", projections.worker(dim));
        scheduler.addWorkerGrid(prefix + "k_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(prefix + "v_proj", projections.worker(config.kvDim()));
        scheduler.addWorkerGrid(prefix + "fused_qkv_bias", qkvBiasWorker);
        scheduler.addWorkerGrid(prefix + "rope_and_kv_cache", ropeWorker);
        scheduler.addWorkerGrid(prefix + "attention", attentionWorker);
        scheduler.addWorkerGrid(prefix + "attn_output_proj", projections.worker(dim));
        scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsNormWorker);
        scheduler.addWorkerGrid(
                prefix + "ffn_rms_apply",
                WorkerGridFactory.genericWorker(dim, LOCAL_WORK_GROUP_SIZE_ALLOC));
        scheduler.addWorkerGrid(
                prefix + "router_projection", workerForRows(config.numberOfExperts()));
        scheduler.addWorkerGrid(
                prefix + "router_softmax_topk",
                WorkerGridFactory.genericWorker(
                        LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC));
        scheduler.addWorkerGrid(prefix + "routed_experts_gate", workerForRows(used * hidden));
        scheduler.addWorkerGrid(prefix + "routed_experts_up", workerForRows(used * hidden));
        scheduler.addWorkerGrid(prefix + "routed_experts_gate_up", elementwise(used * hidden));
        scheduler.addWorkerGrid(prefix + "routed_experts_down", workerForRows(used * dim));
        scheduler.addWorkerGrid(prefix + "shared_expert_gate", projections.worker(shared));
        scheduler.addWorkerGrid(prefix + "shared_expert_up", projections.worker(shared));
        scheduler.addWorkerGrid(prefix + "shared_expert_gate_up", elementwise(shared));
        scheduler.addWorkerGrid(prefix + "shared_expert_down", projections.worker(dim));
        scheduler.addWorkerGrid(
                prefix + "shared_expert_gate_and_accumulate",
                WorkerGridFactory.genericWorker(
                        LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC));
    }

    /** One lane per element, in workgroups of 256. */
    private static WorkerGrid elementwise(int elements) {
        return WorkerGridFactory.genericWorker((elements + 255) / 256 * 256, 256);
    }
}
