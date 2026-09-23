package org.beehive.jllm.inference.weights.standard;

import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * A model-specific implementation of {@link StandardWeights} for the Granite model. This class
 * encapsulates the weights required for performing inference using the Granite model in the
 * standard CPU-based format.
 *
 * <p><b>Note:</b> Granite uses the same weight structure as Llama, with differences only in the
 * scaling factors applied during inference.
 */
public class GraniteStandardWeights extends StandardWeights {

    // @formatter:off
    /**
     * Constructor for GraniteStandardWeights.
     *
     * @param token_embedding_table The token embedding table tensor.
     * @param rms_att_weight Array of RMS attention weights tensors.
     * @param wq Array of query weight tensors.
     * @param wk Array of key weight tensors.
     * @param wv Array of value weight tensors.
     * @param wo Array of output weight tensors.
     * @param rms_ffn_weight Array of RMS feed-forward network weights.
     * @param w1 Array of first feed-forward layer weights (gate).
     * @param w2 Array of second feed-forward layer weights (down).
     * @param w3 Array of third feed-forward layer weights (up).
     * @param rms_final_weight Final RMS weight tensor.
     * @param freq_cis_real Real part of frequency cis tensor (RoPE).
     * @param freq_cis_imag Imaginary part of frequency cis tensor (RoPE).
     * @param wcls Output/classification weight tensor (or shared embedding).
     * @param weightType The GGML weight type (FP16 or Q8_0).
     */
    public GraniteStandardWeights(
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
