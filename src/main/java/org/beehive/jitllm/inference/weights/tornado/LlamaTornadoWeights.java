package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;

public class LlamaTornadoWeights extends TornadoWeights {
    // @formatter:off
    public LlamaTornadoWeights(
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
