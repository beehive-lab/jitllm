package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Base class for TornadoVM-optimized weights. All weight fields are TornadoTensor types (parallel
 * to StandardWeights using FloatTensor).
 *
 * <p>Notes:
 *
 * <ul>
 *   {@link TornadoWeights#tokenEmbeddingTable} should always be loaded as F32 see {@link
 *   ModelLoader#loadTornadoTensorAsFP32}. {@link TornadoWeights#rms_ffn_weightLayered} should
 *   always be loaded as F32 see {@link ModelLoader#loadTornadoTensorAsFP32}. {@link
 *   TornadoWeights#rms_final_weight_as_floatArray} should always be loaded as F32 see {@link
 *   ModelLoader#loadTornadoTensorAsFP32}.
 * </ul>
 */
public abstract class TornadoWeights implements Weights {
    // Token embedding table
    public final TornadoTensor tokenEmbeddingTable; // (vocab_size, dim)

    // Weights for RMSNorms
    public final TornadoTensor[] rms_att_weightLayered; // (layer, dim) rmsnorm weights

    // Weights for attention
    public final TornadoTensor[] wqLayered; // (layer, n_heads * head_size)
    public final TornadoTensor[] wkLayered; // (layer, n_kv_heads, head_size)
    public final TornadoTensor[] wvLayered; // (layer, n_kv_heads * head_size)
    public final TornadoTensor[] woLayered; // (layer, n_heads * head_size, dim)
    public final TornadoTensor[] rms_ffn_weightLayered; // (layer, dim)

    // Weights for FFN
    public final TornadoTensor[] w1Layered; // (layer, hidden_dim, dim)
    public final TornadoTensor[] w2Layered; // (layer, dim, hidden_dim)
    public final TornadoTensor[] w3Layered; // (layer, hidden_dim, dim)

    // Final weights
    public final TornadoTensor rms_final_weight_as_floatArray; // (dim,)
    public final TornadoTensor wclsByteArray; // (vocab_size, dim)

    // RoPE frequencies (always F32)
    public final TornadoTensor freq_cis_realFlat; // (seq_len, head_size/2)
    public final TornadoTensor freq_cis_imagFlat; // (seq_len, head_size/2)

    protected final DataType weightType;

    protected TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rms_att_weightLayered,
            TornadoTensor[] wqLayered,
            TornadoTensor[] wkLayered,
            TornadoTensor[] wvLayered,
            TornadoTensor[] woLayered,
            TornadoTensor[] rms_ffn_weightLayered,
            TornadoTensor[] w1Layered,
            TornadoTensor[] w2Layered,
            TornadoTensor[] w3Layered,
            TornadoTensor rms_final_weight_as_floatArray,
            TornadoTensor freq_cis_realFlat,
            TornadoTensor freq_cis_imagFlat,
            TornadoTensor wclsByteArray,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.rms_att_weightLayered = rms_att_weightLayered;
        this.wqLayered = wqLayered;
        this.wkLayered = wkLayered;
        this.wvLayered = wvLayered;
        this.woLayered = woLayered;
        this.rms_ffn_weightLayered = rms_ffn_weightLayered;
        this.w1Layered = w1Layered;
        this.w2Layered = w2Layered;
        this.w3Layered = w3Layered;
        this.rms_final_weight_as_floatArray = rms_final_weight_as_floatArray;
        this.freq_cis_realFlat = freq_cis_realFlat;
        this.freq_cis_imagFlat = freq_cis_imagFlat;
        this.wclsByteArray = wclsByteArray;
        this.weightType = weightType;
    }

    public TornadoTensor getTokenEmbeddingTable() {
        return tokenEmbeddingTable;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
