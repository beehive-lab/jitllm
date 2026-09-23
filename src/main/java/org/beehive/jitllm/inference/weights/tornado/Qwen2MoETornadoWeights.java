package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * TornadoVM weight container for Qwen2-MoE / Qwen1.5-MoE models.
 *
 * <p>The inherited fields provide the embedding, attention, RMSNorm and final classifier weights.
 * MoE-specific tensors remain in their GGUF Q8_0 layout; GPU kernels will access their {@code
 * ByteArray} representation through {@link TornadoTensor#asByteArray()}.
 */
public final class Qwen2MoETornadoWeights extends Qwen2TornadoWeights {

    public final TornadoTensor[] routerGateLayered;
    public final TornadoTensor[] gateExpertsLayered;
    public final TornadoTensor[] upExpertsLayered;
    public final TornadoTensor[] downExpertsLayered;
    public final TornadoTensor[] sharedGateLayered;
    public final TornadoTensor[] sharedUpLayered;
    public final TornadoTensor[] sharedDownLayered;
    public final TornadoTensor[] sharedGateInputLayered;

    // @formatter:off
    public Qwen2MoETornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rmsAttWeightLayered,
            TornadoTensor[] wqLayered,
            TornadoTensor[] wkLayered,
            TornadoTensor[] wvLayered,
            TornadoTensor[] qBiasLayered,
            TornadoTensor[] kBiasLayered,
            TornadoTensor[] vBiasLayered,
            TornadoTensor[] woLayered,
            TornadoTensor[] rmsFfnWeightLayered,
            TornadoTensor[] routerGateLayered,
            TornadoTensor[] gateExpertsLayered,
            TornadoTensor[] upExpertsLayered,
            TornadoTensor[] downExpertsLayered,
            TornadoTensor[] sharedGateLayered,
            TornadoTensor[] sharedUpLayered,
            TornadoTensor[] sharedDownLayered,
            TornadoTensor[] sharedGateInputLayered,
            TornadoTensor rmsFinalWeight,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor wCls,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                rmsAttWeightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                qBiasLayered,
                kBiasLayered,
                vBiasLayered,
                woLayered,
                rmsFfnWeightLayered,
                null,
                null,
                null,
                rmsFinalWeight,
                freqCisReal,
                freqCisImag,
                wCls,
                weightType);
        this.routerGateLayered = routerGateLayered;
        this.gateExpertsLayered = gateExpertsLayered;
        this.upExpertsLayered = upExpertsLayered;
        this.downExpertsLayered = downExpertsLayered;
        this.sharedGateLayered = sharedGateLayered;
        this.sharedUpLayered = sharedUpLayered;
        this.sharedDownLayered = sharedDownLayered;
        this.sharedGateInputLayered = sharedGateInputLayered;
    }
    // @formatter:on
}
