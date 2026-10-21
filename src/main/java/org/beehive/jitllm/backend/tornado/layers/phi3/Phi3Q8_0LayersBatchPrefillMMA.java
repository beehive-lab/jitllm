package org.beehive.jitllm.backend.tornado.layers.phi3;

import org.beehive.jitllm.backend.tornado.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.Phi3State;
import org.beehive.jitllm.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.jitllm.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Phi-3's tensor-core batched prefill: {@link LlamaQ8_0LayersBatchPrefillMMA}, reading the fused
 * {@code wqkv} and {@code wUp} weights in one GEMM each, and Phi-3's split-half RoPE. A fused
 * weight's rows are the packed rows the layers read: {@code [q|k|v]}, and {@code [gate|up]} with
 * the gate rows first.
 */
public class Phi3Q8_0LayersBatchPrefillMMA extends LlamaQ8_0LayersBatchPrefillMMA {

    public Phi3Q8_0LayersBatchPrefillMMA(
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
            phi3Weights().wqkvLayered[layerIndex].asByteArray(),
            weights.woLayered[layerIndex].asByteArray(),
            weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
            phi3Weights().wUpLayered[layerIndex].asByteArray(),
            phi3Weights().wDownLayered[layerIndex].asByteArray()
        };
    }

    @Override
    protected void qkvProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        projectQ8(
                batchPrefillLayer,
                "qkvProj",
                state.workspace.wrapXbFP16Batch,
                state.workspace.qkvResultBatch,
                paddedBatch,
                config.dim(),
                phi3Weights().wqkvLayered[layerIndex]);
    }

    @Override
    protected void gateUpProjection(TaskGraph batchPrefillLayer, int layerIndex) {
        projectQ8(
                batchPrefillLayer,
                "gateUpProj",
                state.workspace.normedXFFNFP16,
                state.workspace.gateUpResultBatch,
                paddedBatch,
                config.dim(),
                phi3Weights().wUpLayered[layerIndex]);
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
