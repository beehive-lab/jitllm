package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

public class Qwen2MoEStandardWeights extends StandardWeights {
    // Qwen2-MoE-specific weights
    public final FloatTensor[] q_bias, k_bias, v_bias;
    public final FloatTensor[] routerGate;
    public final FloatTensor[] gateExps;
    public final FloatTensor[] upExps;
    public final FloatTensor[] downExps;
    public final FloatTensor[] sharedGate;
    public final FloatTensor[] sharedUp;
    public final FloatTensor[] sharedDown;
    public final FloatTensor[] sharedGateInp;

    public Qwen2MoEStandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] q_bias,
            FloatTensor[] k_bias,
            FloatTensor[] v_bias,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor[] routerGate,
            FloatTensor[] gateExps,
            FloatTensor[] upExps,
            FloatTensor[] downExps,
            FloatTensor[] sharedGate,
            FloatTensor[] sharedUp,
            FloatTensor[] sharedDown,
            FloatTensor[] sharedGateInp,
            FloatTensor rms_final_weight,
            ArrayFloatTensor freq_cis_real,
            ArrayFloatTensor freq_cis_imag,
            FloatTensor wcls,
            DataType weightType) {
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
        this.q_bias = q_bias;
        this.k_bias = k_bias;
        this.v_bias = v_bias;
        this.routerGate = routerGate;
        this.gateExps = gateExps;
        this.upExps = upExps;
        this.downExps = downExps;
        this.sharedGate = sharedGate;
        this.sharedUp = sharedUp;
        this.sharedDown = sharedDown;
        this.sharedGateInp = sharedGateInp;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
