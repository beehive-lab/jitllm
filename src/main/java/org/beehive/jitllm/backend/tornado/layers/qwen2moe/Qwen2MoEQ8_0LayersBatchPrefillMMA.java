package org.beehive.jitllm.backend.tornado.layers.qwen2moe;

import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.Qwen2MoEBatchKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
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
 */
// @formatter:on
public class Qwen2MoEQ8_0LayersBatchPrefillMMA extends Qwen2Q8_0LayersBatchPrefillMMA {

    /** Threads per router row, as the single-graph Qwen2-MoE batched layers use. */
    private static final int ROUTER_LOCAL = 32;

    // Created on first use: the base builds the graphs in its constructor, before this class's
    // fields are assigned, so this field takes no initializer.
    private Qwen35MoeBatchFeedForward moe;

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

    /** The chunk's routing and expert scratch, and the host's per-chunk token count. */
    private Object[] moeScratch() {
        Object[] experts = moe().scratch();
        Object[] scratch = new Object[experts.length + 4];
        scratch[0] = state.workspace.wrapRouterLogitsBatch;
        scratch[1] = state.workspace.wrapNormedBatch;
        scratch[2] = state.workspace.wrapQ8ActBatch;
        scratch[3] = state.workspace.wrapQ8ActScales;
        System.arraycopy(experts, 0, scratch, 4, experts.length);
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
        moe().appendExperts(batchPrefillLayer, context, layerIndex);
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
        moe().addExpertGrids(scheduler, p, layerIndex);
    }
}
