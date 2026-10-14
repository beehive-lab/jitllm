package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Device weights for the {@code lfm2moe} architecture.
 *
 * <p>Extends {@link TornadoWeights} for the reason {@code Qwen35TornadoWeights} does: the shared
 * activation and logits graphs are written against it. The base class's per-layer arrays are
 * indexed by <b>absolute layer</b> and are {@code null} where a layer is of the other kind:
 *
 * <ul>
 *   <li>{@code wqLayered} … {@code woLayered} — attention layers only;
 *   <li>{@code w1Layered}, {@code w2Layered}, {@code w3Layered} — the dense gate, down and up of
 *       the leading layers only;
 *   <li>{@code rms_att_weightLayered}, {@code rms_ffn_weightLayered} — every layer.
 * </ul>
 *
 * <p>Every tensor is retained in the file's own representation; the layer graphs pick a kernel per
 * tensor. {@code weightType} is the representative type the model reports, not a claim about every
 * tensor.
 */
public final class Lfm2TornadoWeights extends TornadoWeights {

    public final TornadoTensor[] attnQNorm;
    public final TornadoTensor[] attnKNorm;

    /** Router, {@code experts × dim}, F32. MoE layers only. */
    public final TornadoTensor[] ffnGateInp;

    /** Selection bias, F32. MoE layers only. */
    public final TornadoTensor[] expProbsBias;

    /** Stacked expert projections. MoE layers only. */
    public final TornadoTensor[] ffnGateExps;

    public final TornadoTensor[] ffnUpExps;
    public final TornadoTensor[] ffnDownExps;

    /** Short-convolution weights. Convolution layers only. */
    public final TornadoTensor[] convInProj;

    public final TornadoTensor[] convKernel;
    public final TornadoTensor[] convOutProj;

    // @formatter:off
    public Lfm2TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] attnNorm,
            TornadoTensor[] ffnNorm,
            TornadoTensor[] ffnGate,
            TornadoTensor[] ffnDown,
            TornadoTensor[] ffnUp,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] attnQNorm,
            TornadoTensor[] attnKNorm,
            TornadoTensor[] ffnGateInp,
            TornadoTensor[] expProbsBias,
            TornadoTensor[] ffnGateExps,
            TornadoTensor[] ffnUpExps,
            TornadoTensor[] ffnDownExps,
            TornadoTensor[] convInProj,
            TornadoTensor[] convKernel,
            TornadoTensor[] convOutProj,
            TornadoTensor outputNorm,
            TornadoTensor output,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                attnNorm,
                wq,
                wk,
                wv,
                wo,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                outputNorm,
                freqCisReal,
                freqCisImag,
                output,
                weightType);
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.ffnGateInp = ffnGateInp;
        this.expProbsBias = expProbsBias;
        this.ffnGateExps = ffnGateExps;
        this.ffnUpExps = ffnUpExps;
        this.ffnDownExps = ffnDownExps;
        this.convInProj = convInProj;
        this.convKernel = convKernel;
        this.convOutProj = convOutProj;
    }
    // @formatter:on
}
