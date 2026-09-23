package org.beehive.jitllm.backend.tornado.layers.type.q4_0;

import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

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
            String taskGraphName,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
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
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.workspace.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.workspace.wrapX);
        }
        Object[] layerWeights = {
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
        String weightSrc = weightSourceGraphName(layerIndex);
        if (weightSrc != null) {
            unifiedLayer.consumeFromDevice(weightSrc, layerWeights);
        } else {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, layerWeights);
        }
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

        unifiedLayer.task(
                "attn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.temp);

        // Llama's query width is dim; the kernel takes it explicitly so the same code serves a
        // family whose head dimension is stated independently.
        unifiedLayer.task(
                "qkv_projection",
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

        ropeAndKeyValueCache(unifiedLayer, layerIndex);

        configureAttention(unifiedLayer, layerIndex);

        unifiedLayer.task(
                "attn_output_proj",
                TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

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

        // The task the Q8_0 path does not have: its gate/up kernel folds this in.
        unifiedLayer.task(
                "ffn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapX,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.tempFFN);

        unifiedLayer.task(
                "ffn_gate_up",
                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapHb,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                config.dim(),
                config.hiddenDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task(
                "ffn_down_proj",
                TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
                context,
                state.workspace.wrapHb,
                state.workspace.wrapX,
                weights.w2Layered[layerIndex].asByteArray(),
                config.hiddenDim(),
                config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.workspace.wrapX);

        return unifiedLayer;
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

        WorkerGrid parallelAttentionWorker =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_rms_reduce", rmsReduceWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid(
                    "layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }
}
