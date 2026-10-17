package org.beehive.jitllm.backend.tornado.layers.granite;

import org.beehive.jitllm.backend.tornado.kernels.GraniteBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.GraniteState;
import org.beehive.jitllm.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Granite's tensor-core batched prefill: {@link LlamaFP16LayersBatchPrefillMMA} with Granite's
 * multipliers. The first layer scales the batch by the embedding multiplier; Q is scaled by {@code
 * attentionScale * sqrt(headSize)} after the QKV GEMM, which the attention kernel's {@code 1 /
 * sqrt(headSize)} turns into the attention multiplier; and both projections added to the stream are
 * scaled by the residual multiplier, the feed-forward's before its FP16 rounding (see {@link
 * #swiglu}). RoPE reads the precomputed tables, built from Granite's own base.
 */
public class GraniteFP16LayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    public GraniteFP16LayersBatchPrefillMMA(
            GraniteState state,
            GraniteTornadoWeights weights,
            GraniteConfiguration config,
            int batchSize) {
        super(state, weights, config, batchSize);
    }

    /** The power of two the SwiGLU product is divided by on top of the residual multiplier. */
    private static final float SWIGLU_HEADROOM = 4f;

    private GraniteConfiguration granite() {
        return (GraniteConfiguration) config;
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.task(
                    "batch_embedding_scale",
                    GraniteBatchPrefillKernels::batchedScale,
                    context,
                    state.workspace.wrapXBatch,
                    granite().embeddingScale(),
                    batchSize * config.dim());
        }
    }

    @Override
    protected void afterQkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_q_scale",
                GraniteBatchPrefillKernels::batchedPackedScaleQ,
                context,
                state.workspace.qkvResultBatch,
                granite().attentionScale() * (float) Math.sqrt(config.headSize()),
                config.dim(),
                config.kvDim(),
                batchSize);
    }

    @Override
    protected void afterAttentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_wo_scale",
                GraniteBatchPrefillKernels::batchedScale,
                context,
                state.workspace.woOut,
                granite().residualScale(),
                batchSize * config.dim());
    }

    /**
     * The residual multiplier is applied to the SwiGLU product before it is rounded to FP16 rather
     * than to the down projection's output: the projection is linear, so the result is the same,
     * and the product stays in range. {@link #SWIGLU_HEADROOM} more keeps it there with room to
     * spare; the down projection's output is multiplied back by it, which is exact for a power of
     * two.
     */
    @Override
    protected void swiglu(TaskGraph batchPrefillLayer, int layerIndex) {
        int n = batchSize * config.hiddenDim();
        batchPrefillLayer.task(
                "swiglu",
                GraniteBatchPrefillKernels::batchedFFNSwiGLUFP16PackedScaled,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                config.hiddenDim(),
                granite().residualScale() / SWIGLU_HEADROOM,
                n);
    }

    @Override
    protected void afterDownProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        batchPrefillLayer.task(
                "batch_w2_scale",
                GraniteBatchPrefillKernels::batchedScale,
                context,
                state.workspace.w2Out,
                SWIGLU_HEADROOM,
                batchSize * config.dim());
    }

    @Override
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {
        var dimWorker = WorkerGridFactory.genericWorker(roundUp(batchSize * config.dim()), 256);
        if (layerIndex == 0) {
            scheduler.addWorkerGrid(p + "batch_embedding_scale", dimWorker);
        }
        scheduler.addWorkerGrid(p + "batch_q_scale", dimWorker);
        scheduler.addWorkerGrid(p + "batch_wo_scale", dimWorker);
        scheduler.addWorkerGrid(p + "batch_w2_scale", dimWorker);
    }

    private static int roundUp(int n) {
        return ((n + 255) / 256) * 256;
    }
}
