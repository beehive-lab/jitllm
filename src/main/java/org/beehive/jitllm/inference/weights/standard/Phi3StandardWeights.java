package org.beehive.jllm.inference.weights.standard;

import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.FloatTensor;

public class Phi3StandardWeights extends StandardWeights {

    // Phi3-specific weight fields that don't exist in the base StandardWeights
    public final FloatTensor[] wqkv; // Combined query, key, value matrices (Phi3 format)
    public final FloatTensor[] wDown; // FFN down projection weight matrices
    public final FloatTensor[] wGateUp; // FFN gate and up projection weight matrices (combined)

    /**
     * Constructor for Phi3 standard (non-TornadoVM) mode Maps Phi3-specific weights to the standard
     * format and stores Phi3-specific weights
     *
     * @param token_embedding_table Token embeddings matrix
     * @param rms_att_weight RMSNorm weights for attention layers
     * @param wqkv Combined query, key, value weight matrices (Phi3 format)
     * @param wo Output projection matrices
     * @param rms_ffn_weight RMSNorm weights for FFN layers
     * @param wDown FFN down projection weight matrices
     * @param wGateUp FFN gate and up projection weight matrices (combined)
     * @param rms_final_weight Final layer normalization weights
     * @param freq_cis_real RoPE cosine components
     * @param freq_cis_imag RoPE sine components
     * @param wcls Classifier weights for output logits
     * @param weightType Weight type specification
     */
    public Phi3StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wqkv,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] wDown,
            FloatTensor[] wGateUp,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            DataType weightType) {

        // Call parent constructor with standard format (using nulls for unsupported weights)
        super(
                token_embedding_table,
                rms_att_weight,
                null,
                null,
                null, // wq, wk, wv - not used in Phi3
                wo,
                rms_ffn_weight,
                null,
                null,
                null, // w1, w2, w3 - not used in Phi3
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);

        // Store Phi3-specific weights
        this.wqkv = wqkv;
        this.wDown = wDown;
        this.wGateUp = wGateUp;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
