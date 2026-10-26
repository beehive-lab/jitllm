package org.beehive.jitllm.backend.tornado.layers.type.q4_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0Packed;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.plan.PackedRepack;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Llama transformer layers reading Q4_0 weights <b>in the file's own representation</b> rather than
 * a Q8_0 materialization of them.
 *
 * <p>Structurally the Q8_0 sibling it extends, with the four weight-reading tasks swapped for their
 * Q4_0 counterparts. Everything that does not touch a quantized weight — the RMS reductions, RoPE
 * with the precomputed frequencies, attention, the transfer and persistence structure, the paged
 * key/value walk — is inherited unchanged, because none of it depends on how the weights are
 * stored.
 *
 * <p><b>One deliberate difference in the feed-forward block.</b> The Q8_0 path uses {@code
 * fullyFusedRmsNormFFNGateUpQ8}, which folds the FFN's RMS normalization into the gate/up
 * projection. Here the norm is applied by the existing {@code reductionOneBlock2WithLayer} task —
 * the same one the attention block already uses — and the Q4_0 gate/up reads the normalized
 * activation. That reuses a verified kernel rather than restating the normalization inside a new
 * one, at the cost of one extra task per layer. It is the trade the Q4_K path already makes.
 *
 * <p><b>Why retain Q4_0 at all.</b> Materializing it as Q8_0 takes 4.5 bits per weight to 8.5, so a
 * model occupies roughly twice its file size on the device. That is the difference between a model
 * fitting and not fitting, and it is the whole reason this class exists rather than the Q8_0 one
 * being reused with converted weights.
 */
public class LlamaQ4_0FFNLayers extends LlamaQ8_0FFNLayers {

    public LlamaQ4_0FFNLayers(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(state, weights, config, schedulerType);
    }

    /** The layers {@code [firstLayer, endLayer)} only; see the Q8_0 constructor. */
    public LlamaQ4_0FFNLayers(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType,
            int firstLayer,
            int endLayer) {
        super(state, weights, config, schedulerType, firstLayer, endLayer);
    }

    /**
     * Each layer names the graph it consumes from: the packed projections are consumed by name from
     * their repack graph, and a graph that consumes anything by name takes nothing through an
     * unnamed consume. The first layer takes its input from the activation graph, which is also the
     * name of a later pipeline stage's receiving graph.
     */
    @Override
    protected String predecessorGraphName(int layerIndex) {
        return layerIndex > firstLayer ? layerGraphName(layerIndex - 1) : "activationUpdate";
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        return appendFFNLayer(null, layerIndex);
    }

    // @formatter:off
    /**
     * The Q8_0 layer's flow with four kernels replaced and one task added.
     *
     * <pre>
     *   attn_rms_reduce      (shared)
     *   attn_rms_finalize    (shared, non-NVIDIA only)
     *   attn_rms_apply       (shared)
     *   qkv_projection       Q4_0
     *   rope_and_kv_cache    (shared)
     *   attention            (shared)
     *   attn_output_proj     Q4_0, with residual
     *   ffn_rms_reduce       (shared)
     *   ffn_rms_finalize     (shared, non-NVIDIA only)
     *   ffn_rms_apply        ADDED — the norm the Q8_0 path folds into its fused gate/up
     *   ffn_gate_up          Q4_0, gate and up in one pass, with SiLU and the GLU product
     *   ffn_down_proj        Q4_0, with residual
     * </pre>
     */
    // @formatter:on
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
        // The projections as tensors: a packed one reaches the device through its repack graph,
        // which turns the uploaded blocks into tiles before any kernel reads them.
        Object[] layerWeights = {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex],
            weights.wkLayered[layerIndex],
            weights.wvLayered[layerIndex],
            weights.woLayered[layerIndex],
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex],
            weights.w2Layered[layerIndex],
            weights.w3Layered[layerIndex]
        };
        String weightSrc = weightSourceGraphName(layerIndex);
        Object[][] binding = weightBinding(weightSrc, layerIndex, layerWeights);
        if (binding[0].length > 0) {
            unifiedLayer.consumeFromDevice(weightSrc, PackedRepack.deviceArrays(binding[0]));
        }
        if (binding[1].length > 0) {
            state.workspace.packedRepack.upload(unifiedLayer, binding[1]);
        }
        if (firstInGraph) {
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            bindAttentionScratch(unifiedLayer, layerIndex);
            if (dp4a()) {
                Object[] packed = PackedActivationTasks.buffers(state.workspace);
                if (layerIndex == firstLayer) {
                    unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, packed);
                } else {
                    consume(unifiedLayer, predecessorGraphName(layerIndex), packed);
                }
            }
        }

        // === Attention Block ===
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

        rmsApply(
                unifiedLayer,
                tp + "attn_rms_apply",
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.temp);

        if (dp4a()) {
            // Three packed projections reading the one activation the apply quantized.
            packedProjection(
                    unifiedLayer,
                    tp + "q_proj",
                    state.workspace.wrapQ,
                    weights.wqLayered[layerIndex],
                    config.dim());
            packedProjection(
                    unifiedLayer,
                    tp + "k_proj",
                    state.workspace.wrapK,
                    weights.wkLayered[layerIndex],
                    config.kvDim());
            packedProjection(
                    unifiedLayer,
                    tp + "v_proj",
                    state.workspace.wrapV,
                    weights.wvLayered[layerIndex],
                    config.kvDim());
        } else {
            if (weights.wqLayered[layerIndex].isPackedQ4()) {
                // Packed weights have only the integer-dot kernels.
                throw PackedWeights.noPackedKernel(
                        "llama decode without packed integer dot products");
            }
            // Llama's query width is dim; the kernel takes it explicitly so the same code serves a
            // family whose head dimension is stated independently.
            unifiedLayer.task(
                    tp + "qkv_projection",
                    TransformerComputeKernelsQ4_0::fusedQKVMatmulQ4_0,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapQ,
                    state.workspace.wrapK,
                    state.workspace.wrapV,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    config.dim(),
                    config.dim(),
                    config.kvDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        configureAttention(unifiedLayer, layerIndex);

        if (dp4a()) {
            // The attention output is a new activation: quantized fresh for the output projection.
            quantize(unifiedLayer, tp + "attn_out_quantize", state.workspace.wrapXb);
            packedResidualProjection(
                    unifiedLayer,
                    tp + "attn_output_proj",
                    TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0DP4A,
                    weights.woLayered[layerIndex],
                    config.dim(),
                    config.dim());
        } else {
            unifiedLayer.task(
                    tp + "attn_output_proj",
                    TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
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

        // The task the Q8_0 path does not have: its gate/up kernel folds this in.
        rmsApply(
                unifiedLayer,
                tp + "ffn_rms_apply",
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.tempFFN);

        if (dp4a() && weights.w1Layered[layerIndex].isPackedQ4()) {
            unifiedLayer.task(
                    tp + "ffn_gate_up",
                    TransformerComputeKernelsQ4_0Packed::fusedFFNGateUpSiLUQ4_0Packed,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim());
        } else if (dp4a()) {
            unifiedLayer.task(
                    tp + "ffn_gate_up",
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapXbSums,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    PACKED_LOCAL);
        } else {
            unifiedLayer.task(
                    tp + "ffn_gate_up",
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapHb,
                    weights.w1Layered[layerIndex].asByteArray(),
                    weights.w3Layered[layerIndex].asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        boolean q4_1Down = weights.w2Layered[layerIndex].dataType() == DataType.Q4_1;
        if (dp4a()) {
            // SwiGLU's output, quantized fresh for the down projection.
            quantize(unifiedLayer, tp + "ffn_down_quantize", state.workspace.wrapHb);
            packedResidualProjection(
                    unifiedLayer,
                    tp + "ffn_down_proj",
                    q4_1Down
                            ? TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1DP4A
                            : TransformerComputeKernelsQ4_0
                                    ::matrixVectorGenericWithResidualQ4_0DP4A,
                    weights.w2Layered[layerIndex],
                    config.hiddenDim(),
                    config.dim());
        } else {
            // Q4_1 for the layers llama-quantize gave more bits; same arguments and grid.
            unifiedLayer.task(
                    tp + "ffn_down_proj",
                    q4_1Down
                            ? TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1
                            : TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
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

    /**
     * Lanes per workgroup for a packed projection: a multiple of 32, as its warp reduction needs.
     */
    private static final int PACKED_LOCAL = 128;

    // @formatter:off
    /**
     * Whether the projections read an activation quantized to eight bits and run as packed integer
     * dot products ({@code dp4a}) rather than in floating point. The path the Qwen3.5 layers take
     * on CUDA, granted where the device lowers {@code dp4a}.
     */
    // @formatter:on
    private static boolean dp4a() {
        return PackedActivationTasks.supported();
    }

    /** The norm's apply; on the packed path it also quantizes its output for the projections. */
    private void rmsApply(TaskGraph graph, String name, FloatArray rmsWeights, FloatArray temp) {
        if (dp4a()) {
            PackedActivationTasks.rmsApplyAndQuantize(
                    graph, name, context, state.workspace, rmsWeights, temp);
        } else {
            graph.task(
                    name,
                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                    context,
                    state.workspace.wrapXb,
                    state.workspace.wrapX,
                    rmsWeights,
                    temp);
        }
    }

    private void quantize(TaskGraph graph, String name, FloatArray activation) {
        PackedActivationTasks.quantize(graph, name, context, state.workspace, activation);
    }

    private void packedProjection(
            TaskGraph graph, String name, FloatArray out, TornadoTensor w, int rows) {
        if (w.isPackedQ4()) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_0Packed::matrixVectorQ4_0Packed,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    out,
                    w.asByteArray(),
                    config.dim(),
                    rows,
                    0);
            return;
        }
        graph.task(
                name,
                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                context,
                state.workspace.wrapXbQuants,
                state.workspace.wrapXbScales,
                state.workspace.wrapXbSums,
                out,
                w.asByteArray(),
                config.dim(),
                rows,
                PACKED_LOCAL);
    }

    private void packedResidualProjection(
            TaskGraph graph,
            String name,
            TornadoFunctions.Task9<
                            KernelContext,
                            IntArray,
                            FloatArray,
                            IntArray,
                            FloatArray,
                            ByteArray,
                            Integer,
                            Integer,
                            Integer>
                    kernel,
            TornadoTensor w,
            int n,
            int rows) {
        if (w.isPackedQ4()) {
            graph.task(
                    name,
                    TransformerComputeKernelsQ4_0Packed::matrixVectorQ4_0Packed,
                    context,
                    state.workspace.wrapXbQuants,
                    state.workspace.wrapXbScales,
                    state.workspace.wrapX,
                    w.asByteArray(),
                    n,
                    rows,
                    1);
            return;
        }
        graph.task(
                name,
                kernel,
                context,
                state.workspace.wrapXbQuants,
                state.workspace.wrapXbScales,
                state.workspace.wrapXbSums,
                state.workspace.wrapX,
                w.asByteArray(),
                n,
                rows,
                PACKED_LOCAL);
    }

    private static WorkerGrid packedWorker(int rows) {
        return PackedActivationTasks.projectionWorker(rows, PACKED_LOCAL);
    }

    /** The grid for a projection over {@code w}: a 16-row group per workgroup when it is packed. */
    private static WorkerGrid packedWorker(int rows, TornadoTensor w, boolean fused) {
        if (w.isPackedQ4()) {
            int local =
                    fused
                            ? TransformerComputeKernelsQ4_0Packed.FUSED_LOCAL
                            : TransformerComputeKernelsQ4_0Packed.LOCAL;
            return WorkerGridFactory.genericWorker(rows / 16 * local, local);
        }
        return packedWorker(rows);
    }

    /**
     * The Q8_0 schedule plus a grid for {@code ffn_rms_apply}, and {@code ffn_gate_up} in place of
     * {@code rms_ffn_gate_up}.
     *
     * <p>Written out rather than delegating and patching: a grid registered for a task name that no
     * longer exists is not an error TornadoVM reports, and a task with no grid is. Stating the
     * whole schedule keeps the two lists in one place where they can be read against the graph
     * above.
     */
    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
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

        WorkerGrid parallelAttentionWorker = attentionWorker();

        for (int i = firstLayer; i < endLayer(config.numberOfLayers()); i++) {
            if (dp4a()) {
                String layer = layerGraphName(i) + "." + layerTaskPrefix(i);
                tornadoForwardScheduler.addWorkerGrid(layer + "attn_rms_reduce", rmsReduceWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "attn_rms_apply",
                        PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "q_proj", packedWorker(config.dim(), weights.wqLayered[i], false));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "k_proj",
                        packedWorker(config.kvDim(), weights.wkLayered[i], false));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "v_proj",
                        packedWorker(config.kvDim(), weights.wvLayered[i], false));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "rope_and_kv_cache", ropeWithCacheWorker);
                tornadoForwardScheduler.addWorkerGrid(layer + "attention", parallelAttentionWorker);
                addAttentionCombineGrid(tornadoForwardScheduler, i);
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "attn_out_quantize",
                        PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "attn_output_proj",
                        packedWorker(config.dim(), weights.woLayered[i], false));
                tornadoForwardScheduler.addWorkerGrid(layer + "ffn_rms_reduce", rmsReduceWorker);
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "ffn_rms_apply",
                        PackedActivationTasks.quantizeWorker(config.dim()));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "ffn_gate_up",
                        packedWorker(config.hiddenDim(), weights.w1Layered[i], true));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "ffn_down_quantize",
                        PackedActivationTasks.quantizeWorker(config.hiddenDim()));
                tornadoForwardScheduler.addWorkerGrid(
                        layer + "ffn_down_proj",
                        packedWorker(config.dim(), weights.w2Layered[i], false));
                continue;
            }
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "attn_rms_reduce",
                    rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "qkv_projection",
                    fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "rope_and_kv_cache",
                    ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "attention",
                    parallelAttentionWorker);
            addAttentionCombineGrid(tornadoForwardScheduler, i);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "attn_output_proj",
                    configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "ffn_rms_reduce",
                    rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "ffn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "ffn_gate_up",
                    configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    layerGraphName(i) + "." + layerTaskPrefix(i) + "ffn_down_proj",
                    configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }
}
