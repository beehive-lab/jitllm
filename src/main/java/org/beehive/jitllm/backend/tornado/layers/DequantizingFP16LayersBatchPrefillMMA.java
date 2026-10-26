package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * A batched prefill on the tensor cores for quantized weights that have no tensor-core GEMM of
 * their own: {@link LlamaFP16LayersBatchPrefillMMA}, each projection's weight decoded into one FP16
 * scratch before its FP16 GEMM.
 *
 * <p>Decoding a projection at a time keeps the weights quantized between chunks, at the cost of one
 * scratch the size of the widest projection pair (gate and up) and a decode pass per projection per
 * chunk. Q, K and V are decoded end to end, and gate and up likewise, so one plain GEMM writes the
 * packed rows the layers read. Subclasses say how a weight is decoded.
 */
// @formatter:on
public abstract class DequantizingFP16LayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    // Created on first use: the base builds the graphs in its constructor, before this class's
    // fields are assigned, so this field takes no initializer.
    private HalfFloatArray scratch;

    protected DequantizingFP16LayersBatchPrefillMMA(
            State state, TornadoWeights weights, Configuration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    private int qDim() {
        return config.numberOfHeads() * config.headSize();
    }

    protected HalfFloatArray scratch() {
        if (scratch == null) {
            int dim = config.dim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim();
            long widest =
                    Math.max(
                            Math.max((long) (qDim() + 2 * kvDim) * dim, 2L * hidden * dim),
                            Math.max((long) dim * qDim(), (long) dim * hidden));
            scratch = new HalfFloatArray(Math.toIntExact(widest));
        }
        return scratch;
    }

    @Override
    protected Object[] layerWeights(int layerIndex) {
        return new Object[] {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            weights.wqLayered[layerIndex].asByteArray(),
            weights.wkLayered[layerIndex].asByteArray(),
            weights.wvLayered[layerIndex].asByteArray(),
            weights.woLayered[layerIndex].asByteArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            weights.w1Layered[layerIndex].asByteArray(),
            weights.w2Layered[layerIndex].asByteArray(),
            weights.w3Layered[layerIndex].asByteArray(),
            weights.freq_cis_realFlat.asFloatArray(),
            weights.freq_cis_imagFlat.asFloatArray()
        };
    }

    @Override
    protected void atLayerStart(TaskGraph batchPrefillLayer, int layerIndex) {
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch());
        } else {
            batchPrefillLayer.consumeFromDevice(
                    GraphNames.batchPrefillLayer(layerIndex - 1), scratch());
        }
    }

    /**
     * Adds the task {@code task} that decodes {@code n} elements of {@code w} into the scratch at
     * {@code destOffset}.
     */
    protected abstract void decode(
            TaskGraph layer, String task, TornadoTensor w, int destOffset, int n);

    @Override
    protected void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        decode(batchPrefillLayer, "qDecode", weights.wqLayered[layerIndex], 0, qDim() * dim);
        decode(
                batchPrefillLayer,
                "kDecode",
                weights.wkLayered[layerIndex],
                qDim() * dim,
                kvDim * dim);
        decode(
                batchPrefillLayer,
                "vDecode",
                weights.wvLayered[layerIndex],
                (qDim() + kvDim) * dim,
                kvDim * dim);
        projectFP16(
                batchPrefillLayer,
                "qkvProj",
                state.workspace.wrapXbFP16Batch,
                scratch(),
                state.workspace.qkvResultBatch,
                paddedBatch,
                qDim() + 2 * kvDim,
                dim);
    }

    @Override
    protected void attentionOutputProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        decode(batchPrefillLayer, "woDecode", weights.woLayered[layerIndex], 0, dim * qDim());
        projectFP16(
                batchPrefillLayer,
                "woProj",
                state.workspace.attnOutFP16,
                scratch(),
                state.workspace.woOut,
                paddedBatch,
                dim,
                qDim());
    }

    @Override
    protected void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidden = config.hiddenDim();
        decode(batchPrefillLayer, "gateDecode", weights.w1Layered[layerIndex], 0, hidden * dim);
        decode(
                batchPrefillLayer,
                "upDecode",
                weights.w3Layered[layerIndex],
                hidden * dim,
                hidden * dim);
        projectFP16(
                batchPrefillLayer,
                "gateUpProj",
                state.workspace.normedXFFNFP16,
                scratch(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                2 * hidden,
                dim);
    }

    @Override
    protected void downProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        int dim = config.dim();
        int hidden = config.hiddenDim();
        decode(batchPrefillLayer, "downDecode", weights.w2Layered[layerIndex], 0, dim * hidden);
        projectFP16(
                batchPrefillLayer,
                "w2Proj",
                state.workspace.wrapHbFP16Batch,
                scratch(),
                state.workspace.w2Out,
                batchSize,
                dim,
                hidden);
    }

    @Override
    protected void addFamilyTaskGrids(GridScheduler scheduler, String p, int layerIndex) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidden = config.hiddenDim();
        scheduler.addWorkerGrid(p + "qDecode", elementwise(qDim() * dim));
        scheduler.addWorkerGrid(p + "kDecode", elementwise(kvDim * dim));
        scheduler.addWorkerGrid(p + "vDecode", elementwise(kvDim * dim));
        scheduler.addWorkerGrid(p + "woDecode", elementwise(dim * qDim()));
        scheduler.addWorkerGrid(p + "gateDecode", elementwise(hidden * dim));
        scheduler.addWorkerGrid(p + "upDecode", elementwise(hidden * dim));
        scheduler.addWorkerGrid(p + "downDecode", elementwise(dim * hidden));
    }

    private static uk.ac.manchester.tornado.api.WorkerGrid elementwise(int n) {
        return WorkerGridFactory.genericWorker(((n + 255) / 256) * 256, 256);
    }
}
