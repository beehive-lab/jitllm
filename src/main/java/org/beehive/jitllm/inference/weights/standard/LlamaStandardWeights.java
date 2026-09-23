package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A model-specific implementation of {@link StandardWeights} for the Llama model. This class
 * encapsulates the weights required for performing inference using the Llama model in the standard
 * CPU-based format.
 *
 * <p><b>Note:</b> This weight format is also used for the Mistral model.
 */
public class LlamaStandardWeights extends StandardWeights {

    // @formatter:off
    /**
     * Constructor for LlamaStandardWeights.
     *
     * @param token_embedding_table The token embedding table tensor.
     * @param rms_att_weight Array of RMS attention weights tensors.
     * @param wq Array of query weight tensors.
     * @param wk Array of key weight tensors.
     * @param wv Array of value weight tensors.
     * @param wo Array of output weight tensors.
     * @param rms_ffn_weight Array of RMS feed-forward network weights.
     * @param w1 Array of first feed-forward layer weights.
     * @param w2 Array of second feed-forward layer weights.
     * @param w3 Array of third feed-forward layer weights.
     * @param rms_final_weight Final RMS weight tensor.
     * @param freq_cis_real Real part of frequency cis tensor.
     * @param freq_cis_imag Imaginary part of frequency cis tensor.
     * @param wcls Class token weight tensor.
     * @param weightType The GGML weight type.
     */
    public LlamaStandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            DataType weightType) {
        // call to StandardWeights constructor
        super(
                token_embedding_table,
                rms_att_weight,
                wq,
                wk,
                wv,
                wo,
                rms_ffn_weight,
                w1,
                w2,
                w3,
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);
    }

    // @formatter:on

    @Override
    public DataType dataType() {
        return weightType;
    }
}
