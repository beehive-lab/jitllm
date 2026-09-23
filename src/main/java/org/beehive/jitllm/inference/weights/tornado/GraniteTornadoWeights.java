package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * A model-specific implementation of {@link TornadoWeights} for the Granite model. This class
 * encapsulates the weights required for performing inference using the Granite model in the
 * TornadoVM GPU-accelerated format.
 *
 * <p><b>Note:</b> Granite uses the same weight structure as Llama in TornadoVM format, with
 * differences only in the scaling factors applied during inference.
 */
public class GraniteTornadoWeights extends TornadoWeights {
    // @formatter:off
    /**
     * Constructor for GraniteTornadoWeights.
     *
     * @param tokenEmbeddingTable The token embedding table tensor.
     * @param rms_att_weightLayered Array of RMS attention weights tensors.
     * @param wqLayered Array of query weight tensors.
     * @param wkLayered Array of key weight tensors.
     * @param wvLayered Array of value weight tensors.
     * @param woLayered Array of output weight tensors.
     * @param rms_ffn_weightLayered Array of RMS feed-forward network weights.
     * @param w1Layered Array of first feed-forward layer weights (gate).
     * @param w2Layered Array of second feed-forward layer weights (down).
     * @param w3Layered Array of third feed-forward layer weights (up).
     * @param rms_final_weight_as_floatArray Final RMS weight tensor.
     * @param freq_cis_realFlat Real part of frequency cis tensor (RoPE).
     * @param freq_cis_imagFlat Imaginary part of frequency cis tensor (RoPE).
     * @param wclsByteArray Output/classification weight tensor (or shared embedding).
     * @param weightType The GGML weight type (FP16 or Q8_0).
     */
    public GraniteTornadoWeights(
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
        super(
                tokenEmbeddingTable,
                rms_att_weightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                rms_ffn_weightLayered,
                w1Layered,
                w2Layered,
                w3Layered,
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);
    }
    // @formatter:on
}
