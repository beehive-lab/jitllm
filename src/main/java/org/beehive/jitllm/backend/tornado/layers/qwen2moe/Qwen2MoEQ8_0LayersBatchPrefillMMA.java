package org.beehive.jitllm.backend.tornado.layers.qwen2moe;

import org.beehive.jitllm.backend.tornado.NativePrefillSupport;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.layers.Qwen35MoeBatchFeedForward;
import org.beehive.jitllm.backend.tornado.layers.qwen2.Qwen2Q8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Qwen2MoEState;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * Qwen2-MoE's tensor-core batched prefill: Qwen2's attention on the tensor cores ({@link
 * Qwen2Q8_0LayersBatchPrefillMMA}), and a mixture-of-experts feed-forward in place of the dense
 * one.
 *
 * <p>The feed-forward adds the attention output to the stream, normalizes and quantizes the chunk,
 * and routes it with Qwen2-MoE's own router and top-k: a softmax over every expert, the chosen
 * experts keeping their share of it unrenormalized, for any expert count. From the routed ids and
 * weights on it runs {@link Qwen35MoeBatchFeedForward}'s expert tasks: the assignments grouped by
 * expert, the grouped int8 tensor-core GEMMs over the stacked Q8_0 experts, the shared expert
 * behind its sigmoid gate, and the weighted combine into the residual stream.
 *
 * <p>A shared expert the int8 GEMMs do not read (any weight other than Q8_0 or Q4_0) runs here
 * instead, as the dense layers run their feed-forward ({@link FP16Projections}): gate and up
 * decoded to FP16 and multiplied in one GEMM, SwiGLU, then down. The routed experts' grouped int8
 * GEMMs, run as one expert over the chunk, took about five times as long as the dense int8 GEMMs
 * over the same work, and on Q6_K half again as long as these.
 */
// @formatter:on
public class Qwen2MoEQ8_0LayersBatchPrefillMMA extends Qwen2Q8_0LayersBatchPrefillMMA {

    /** Threads per router row, as the single-graph Qwen2-MoE batched layers use. */
    private static final int ROUTER_LOCAL = 32;

    // Created on first use: the base builds the graphs in its constructor, before this class's
    // fields are assigned, so these fields take no initializer.
    private Qwen35MoeBatchFeedForward moe;
    private FP16Projections sharedProjections;
    private FloatArray sharedGateUpBatch;
    private HalfFloatArray sharedHiddenFP16;

    public Qwen2MoEQ8_0LayersBatchPrefillMMA(
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            int batchSize) {
        super(state, weights, config, batchSize);
    }

    private Qwen2MoETornadoWeights moeWeights() {
        return (Qwen2MoETornadoWeights) weights;
    }

    private Qwen2MoEConfiguration moeConfig() {
        return (Qwen2MoEConfiguration) config;
    }

    private Qwen35MoeBatchFeedForward moe() {
        if (moe == null) {
            Qwen2MoEConfiguration c = moeConfig();
            Qwen2MoETornadoWeights w = moeWeights();
            moe =
                    new Qwen35MoeBatchFeedForward(
                            state.workspace,
                            new Qwen35Experts(
                                    c.numberOfExperts(),
                                    c.numberOfExpertsUsed(),
                                    c.moeHiddenDim(),
                                    c.sharedExpertHiddenDim()),
                            new Qwen35ExpertWeights<TornadoTensor>(
                                    w.routerGateLayered,
                                    w.gateExpertsLayered,
                                    w.upExpertsLayered,
                                    w.downExpertsLayered,
                                    w.sharedGateLayered,
                                    w.sharedUpLayered,
                                    w.sharedDownLayered,
                                    w.sharedGateInputLayered),
                            c.dim(),
                            batchSize);
        }
        return moe;
    }

    /** Whether any layer's shared expert runs here, on {@link #sharedProjections}. */
    private boolean anyFP16Shared() {
        for (int i = 0; i < moeConfig().numberOfLayers(); i++) {
            if (!moe().int8Shared(i)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The FP16 projections of the shared experts that run here, decoded per chunk, and their packed
     * gate/up output and FP16 hidden activation; created with the first.
     */
    private FP16Projections sharedProjections() {
        if (sharedProjections == null) {
            Qwen2MoEConfiguration c = moeConfig();
            int shared = c.sharedExpertHiddenDim();
            sharedProjections =
                    new FP16Projections(
                            context,
                            state.workspace.prepacking,
                            2L * shared * c.dim(),
                            paddedBatch,
                            Math.max(2 * shared, c.dim()),
                            false,
                            2 * c.numberOfLayers(),
                            NativePrefillSupport.nativeProjections(state.executionPolicy()));
            sharedGateUpBatch = new FloatArray(paddedBatch * 2 * shared);
            sharedHiddenFP16 = new HalfFloatArray(paddedBatch * shared);
        }
        return sharedProjections;
    }

    /** Attention's weights and the experts', in place of a dense gate, up and down. */
    @Override
    protected Object[] layerWeights(int layerIndex) {
        Qwen2MoETornadoWeights w = moeWeights();
        return new Object[] {
            w.rms_att_weightLayered[layerIndex].asFloatArray(),
            w.wqLayered[layerIndex].asByteArray(),
            w.wkLayered[layerIndex].asByteArray(),
            w.wvLayered[layerIndex].asByteArray(),
            w.woLayered[layerIndex].asByteArray(),
            w.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            w.routerGateLayered[layerIndex].asFloatArray(),
            w.sharedGateInputLayered[layerIndex].asFloatArray(),
            w.gateExpertsLayered[layerIndex].asByteArray(),
            w.upExpertsLayered[layerIndex].asByteArray(),
            w.downExpertsLayered[layerIndex].asByteArray(),
            w.sharedGateLayered[layerIndex].asByteArray(),
            w.sharedUpLayered[layerIndex].asByteArray(),
            w.sharedDownLayered[layerIndex].asByteArray()
        };
    }

    /**
     * The chunk's routing and expert scratch, and the FP16 shared expert's where a layer runs one.
     */
    private Object[] moeScratch() {
        Object[] experts = moe().scratch();
        Object[] shared =
                anyFP16Shared()
                        ? java.util.stream.Stream.concat(
                                        java.util.Arrays.stream(sharedProjections().buffers()),
                                        java.util.stream.Stream.of(
                                                sharedGateUpBatch, sharedHiddenFP16))
                                .toArray()
                        : new Object[0];
        Object[] scratch = new Object[experts.length + 4 + shared.length];
        scratch[0] = state.workspace.wrapRouterLogitsBatch;
        scratch[1] = state.workspace.wrapNormedBatch;
        scratch[2] = state.workspace.wrapQ8ActBatch;
        scratch[3] = state.workspace.wrapQ8ActScales;
        System.arraycopy(experts, 0, scratch, 4, experts.length);
        System.arraycopy(shared, 0, scratch, 4 + experts.length, shared.length);
        return scratch;
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.activeBatchSizeHolder);
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, moeScratch());
        } else {
            String pred = GraphNames.batchPrefillLayer(layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(pred, state.workspace.activeBatchSizeHolder);
            batchPrefillLayer.consumeFromDevice(pred, moeScratch());
        }
    }

    @Override
    protected void feedForward(TaskGraph batchPrefillLayer, int layerIndex) {
        Qwen2MoEConfiguration c = moeConfig();
        int dim = c.dim();
        // x += woOut, fused into the RMS reduction, as the dense layers do.
        batchPrefillLayer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                c.rmsNormEps(),
                RMS_LOCAL_SIZE);
        batchPrefillLayer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapXBatch,
                moeWeights().rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        batchPrefillLayer.task(
                "moe_act_q8",
                Int8GemmKernels::quantizeActivationsQ8Warp,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapQ8ActBatch,
                state.workspace.wrapQ8ActScales,
                dim);
        batchPrefillLayer.task(
                "batch_router",
                Qwen2MoEBatchKernels::batchedRouterProjection,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapRouterLogitsBatch,
                moeWeights().routerGateLayered[layerIndex].asFloatArray(),
                state.workspace.activeBatchSizeHolder,
                dim,
                c.numberOfExperts(),
                ROUTER_LOCAL);
        batchPrefillLayer.task(
                "batch_topk",
                Qwen2MoEBatchKernels::batchedSoftmaxAndTopK,
                context,
                state.workspace.wrapRouterLogitsBatch,
                state.workspace.wrapMoeIdsBatch,
                state.workspace.wrapMoeWeightsBatch,
                state.workspace.activeBatchSizeHolder,
                c.numberOfExperts(),
                c.numberOfExpertsUsed());
        moe().appendSharedGateInput(batchPrefillLayer, context, layerIndex);
        boolean fp16Shared = !moe().int8Shared(layerIndex);
        if (fp16Shared) {
            appendSharedExpertFP16(batchPrefillLayer, layerIndex);
        }
        moe().appendExperts(batchPrefillLayer, context, layerIndex, !fp16Shared);
    }

    /**
     * The shared expert over the chunk into {@code wrapMoeSharedOut}, which the combine gates and
     * adds: the normed chunk in FP16, gate and up in one GEMM, SwiGLU, down.
     */
    private void appendSharedExpertFP16(TaskGraph layer, int layerIndex) {
        Qwen2MoEConfiguration c = moeConfig();
        Qwen2MoETornadoWeights w = moeWeights();
        int dim = c.dim();
        int shared = c.sharedExpertHiddenDim();
        FP16Projections projections = sharedProjections();
        layer.task(
                "shared_rms_apply16",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                w.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        projections.project(
                layer,
                "sharedGateUpProj",
                state.workspace.normedXFFNFP16,
                sharedGateUpBatch,
                paddedBatch,
                dim,
                w.sharedGateLayered[layerIndex],
                w.sharedUpLayered[layerIndex]);
        layer.task(
                "shared_swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                sharedHiddenFP16,
                sharedGateUpBatch,
                shared);
        projections.project(
                layer,
                "sharedDownProj",
                sharedHiddenFP16,
                state.workspace.wrapMoeSharedOut,
                batchSize,
                shared,
                w.sharedDownLayered[layerIndex]);
    }

    @Override
    protected void addFeedForwardGrids(GridScheduler scheduler, String p, int layerIndex) {
        int dim = moeConfig().dim();
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms",
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p + "batch_ffn_rms_apply", WorkerGridFactory.genericWorker(batchSize * dim, 256));
        scheduler.addWorkerGrid(
                p + "moe_act_q8", WorkerGridFactory.genericWorker(batchSize * dim, 128));
        scheduler.addWorkerGrid(
                p + "batch_router",
                WorkerGridFactory.genericWorker(
                        batchSize * moeConfig().numberOfExperts() * ROUTER_LOCAL, ROUTER_LOCAL));
        scheduler.addWorkerGrid(p + "batch_topk", WorkerGridFactory.genericWorker(batchSize, 1));
        moe().addSharedGateInputGrid(scheduler, p);
        boolean fp16Shared = !moe().int8Shared(layerIndex);
        if (fp16Shared) {
            int shared = moeConfig().sharedExpertHiddenDim();
            scheduler.addWorkerGrid(
                    p + "shared_rms_apply16",
                    WorkerGridFactory.genericWorker(batchSize * dim, 256));
            scheduler.addWorkerGrid(p + "sharedGateUpProj", mmaGrid(paddedBatch, 2 * shared));
            scheduler.addWorkerGrid(p + "shared_swiglu", elementwiseGrid(batchSize * shared));
            scheduler.addWorkerGrid(p + "sharedDownProj", mmaGrid(paddedBatch, dim));
            // After the GEMMs' grids, which a split projection's replace.
            sharedProjections().addGrids(scheduler);
        }
        moe().addExpertGrids(scheduler, p, layerIndex, !fp16Shared);
    }
}
