package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Base class that represents the standard weight format used for Java-based CPU inference. This
 * abstract class provides the foundation for defining model-specific weights in the StandardWeights
 * format.
 */
public abstract class StandardWeights implements Weights {
    // token embedding table
    public final FloatTensor token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public final FloatTensor[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls
    public final FloatTensor[] wq; // (layer, n_heads * head_size)
    public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
    public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
    public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
    public final FloatTensor[] rms_ffn_weight; // (layer, dim)

    // weights for ffn
    public final FloatTensor[] w1; // (layer, hidden_dim, dim)
    public final FloatTensor[] w2; // (layer, dim, hidden_dim)
    public final FloatTensor[] w3; // (layer, hidden_dim, dim)
    //
    public final FloatTensor wcls; // (vocab_size, dim)
    // public final rmsnorm
    public final FloatTensor rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public final FloatTensor freq_cis_real; // (seq_len, head_size/2)
    public final FloatTensor freq_cis_imag; // (seq_len, head_size/2)

    // (optional) classifier weights for the logits, on the last layer
    protected final DataType weightType;

    // @formatter:off
    /**
     * Constructor for standard (non-TornadoVM) mode
     *
     * @param token_embedding_table Token embeddings matrix
     * @param rms_att_weight RMSNorm weights for attention layers
     * @param wq Query weight matrices
     * @param wk Key weight matrices
     * @param wv Value weight matrices
     * @param wo Output projection matrices
     * @param rms_ffn_weight RMSNorm weights for FFN layers
     * @param w1 First FFN weight matrices
     * @param w2 Second FFN weight matrices
     * @param w3 Third FFN weight matrices (gate)
     * @param rms_final_weight Final layer normalization weights
     * @param freq_cis_real RoPE cosine components
     * @param freq_cis_imag RoPE sine components
     * @param wcls Classifier weights for output logits
     */
    protected StandardWeights(
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

        // Standard format
        this.token_embedding_table = token_embedding_table;
        this.rms_att_weight = rms_att_weight;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;

        this.rms_ffn_weight = rms_ffn_weight;
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
        this.wcls = wcls;
        this.rms_final_weight = rms_final_weight;
        this.freq_cis_real = freq_cis_real;
        this.freq_cis_imag = freq_cis_imag;
        this.weightType = weightType;
    }
    // @formatter:on
}
