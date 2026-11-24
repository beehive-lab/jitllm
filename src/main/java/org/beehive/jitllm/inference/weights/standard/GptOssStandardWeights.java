package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.GptOssLayerWeights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/** {@code gpt-oss} weights for the host. The FFN slots are empty: every layer is experts. */
public final class GptOssStandardWeights extends StandardWeights {

    private final GptOssLayerWeights<FloatTensor> layers;

    public GptOssStandardWeights(
            FloatTensor tokenEmbeddingTable,
            FloatTensor[] attentionNorm,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] postAttentionNorm,
            GptOssLayerWeights<FloatTensor> layers,
            FloatTensor finalNorm,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            FloatTensor wcls,
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

    public GptOssLayerWeights<FloatTensor> layers() {
        return layers;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
