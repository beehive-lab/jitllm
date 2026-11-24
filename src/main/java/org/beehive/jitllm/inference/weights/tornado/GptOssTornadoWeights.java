package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * {@code gpt-oss} weights for the device, each tensor in the file's representation: Q8_0 attention
 * and vocabulary, MXFP4 experts, F32 norms, biases, sinks and router.
 */
public final class GptOssTornadoWeights extends TornadoWeights {

    private final GptOssLayerWeights<TornadoTensor> layers;

    public GptOssTornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] attentionNorm,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] postAttentionNorm,
            GptOssLayerWeights<TornadoTensor> layers,
            TornadoTensor finalNorm,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor wcls,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                attentionNorm,
                wq,
                wk,
                wv,
                wo,
                postAttentionNorm,
                null,
                null,
                null,
                finalNorm,
                freqCisReal,
                freqCisImag,
                wcls,
                weightType);
        this.layers = layers;
    }

    public GptOssLayerWeights<TornadoTensor> layers() {
        return layers;
    }

    /** The attention projections and the experts: what the prepacking and memory plans walk. */
    @Override
    public java.util.List<TornadoTensor[]> projections() {
        return java.util.Arrays.asList(
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                layers.gateExperts(),
                layers.upExperts(),
                layers.downExperts());
    }
}
