package org.beehive.jitllm.backend.tornado.layers.phi3;

import org.beehive.jitllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Phi-3's tensor-core batched prefill: {@link LlamaFP16LayersBatchPrefillMMA}, reading the fused
 * {@code wqkv} and {@code wUp} weights in one GEMM each, and Phi-3's split-half RoPE. A fused
 * weight's rows are the packed rows the layers read: {@code [q|k|v]}, and {@code [gate|up]} with
 * the gate rows first.
 */
public class Phi3FP16LayersBatchPrefillMMA extends LlamaFP16LayersBatchPrefillMMA {

    public Phi3FP16LayersBatchPrefillMMA(
            Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config, int batchSize) {
        super(state, weights, config, batchSize);
    }

    // The base builds its graphs in its constructor, before this class's fields would be set,
    // so the hooks read the inherited fields.
    private Phi3TornadoWeights phi3Weights() {
        return (Phi3TornadoWeights) weights;
    }

    @Override
    protected Object[] layerWeights(int layerIndex) {
        return new Object[] {
            weights.rms_att_weightLayered[layerIndex].asFloatArray(),
            phi3Weights().wqkvLayered[layerIndex].asHalfFloatArray(),
            weights.woLayered[layerIndex].asHalfFloatArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            phi3Weights().wUpLayered[layerIndex].asHalfFloatArray(),
            phi3Weights().wDownLayered[layerIndex].asHalfFloatArray()
        };
    }

    @Override
    protected void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        projectFP16(
                batchPrefillLayer,
                "qkvProj",
                state.workspace.wrapXbFP16Batch,
                phi3Weights().wqkvLayered[layerIndex].asHalfFloatArray(),
                state.workspace.qkvResultBatch,
                paddedBatch,
                config.dim() + 2 * config.kvDim(),
                config.dim());
    }

    @Override
    protected void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        projectFP16(
                batchPrefillLayer,
                "gateUpProj",
                state.workspace.normedXFFNFP16,
                phi3Weights().wUpLayered[layerIndex].asHalfFloatArray(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                2 * config.hiddenDim(),
                config.dim());
    }

    @Override
    protected TornadoTensor downWeight(int layerIndex) {
        return phi3Weights().wDownLayered[layerIndex];
    }

    @Override
    protected boolean splitHalfRope() {
        return true;
    }

    @Override
    protected float ropeTheta() {
        return config.ropeTheta();
    }
}
