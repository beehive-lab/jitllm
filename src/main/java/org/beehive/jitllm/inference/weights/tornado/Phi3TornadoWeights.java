package org.beehive.jllm.inference.weights.tornado;

import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.runtime.tensor.DataType;

public class Phi3TornadoWeights extends TornadoWeights {

    // Phi3-specific weight arrays
    public TornadoTensor[]
            wqkvLayered; // hf - Combined QKV weights: (layer, op_size, dim) where op_size = dim + 2
    // * (n_kv_heads * head_dim)
    public TornadoTensor[] wDownLayered; // hf - FFN down projection: (layer, dim, hidden_dim)
    public TornadoTensor[] wUpLayered; // hf - FFN up projection: (layer, hidden_dim, dim)

    // @formatter:off
    public Phi3TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rms_att_weightLayered,
            TornadoTensor[] wqkvLayered, // Combined QKV weights for Phi3
            TornadoTensor[] woLayered,
            TornadoTensor[] rms_ffn_weightLayered,
            TornadoTensor[] wDownLayered, // FFN down weights
            TornadoTensor[] wUpLayered, // FFN up weights
            TornadoTensor rms_final_weight_as_floatArray,
            TornadoTensor freq_cis_realFlat,
            TornadoTensor freq_cis_imagFlat,
            TornadoTensor wclsByteArray,
            DataType weightType) {

        // Call to BaseTornadoWeights constructor with null values for unused standard weights
        super(
                tokenEmbeddingTable,
                rms_att_weightLayered,
                null, // wqLayered - not used in Phi3, using combined wqkv instead
                null, // wkLayered - not used in Phi3, using combined wqkv instead
                null, // wvLayered - not used in Phi3, using combined wqkv instead
                woLayered,
                rms_ffn_weightLayered,
                null, // w1Layered - not used in Phi3, using wUp instead
                null, // w2Layered - not used in Phi3, using wDown instead
                null, // w3Layered - not used in Phi3, using wUp instead
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);

        // Initialize Phi3-specific fields
        this.wqkvLayered = wqkvLayered;
        this.wDownLayered = wDownLayered;
        this.wUpLayered = wUpLayered;
    }
    // @formatter:on
}
