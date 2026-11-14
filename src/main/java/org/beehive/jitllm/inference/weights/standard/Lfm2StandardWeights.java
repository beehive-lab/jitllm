package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Host weights for the {@code lfm2moe} architecture.
 *
 * <p>Implements {@link Weights} directly rather than extending {@link StandardWeights}, for the
 * reason {@code Qwen35StandardWeights} does: three layers in four carry no query, key or value
 * projection, and the feed-forward is dense in some layers and routed in the rest.
 *
 * <p>Every per-layer array is indexed by <b>absolute layer</b> and holds {@code null} where the
 * layer is of the other kind, so the forward pass reads {@code wq[l]} with the same {@code l} it
 * uses everywhere else.
 *
 * <p>The expert tensors are stacked, one tensor per layer holding every expert: {@code
 * ffn_gate_exps} and {@code ffn_up_exps} are {@code experts × expertHidden × dim}, {@code
 * ffn_down_exps} is {@code experts × dim × expertHidden}, row-major, so expert {@code e} starts
 * {@code e × rows × columns} elements in.
 */
public final class Lfm2StandardWeights implements Weights {

    public final FloatTensor tokenEmbeddingTable;

    /** Input norm of the mixer, every layer (llama.cpp's {@code operator_norm}). */
    public final FloatTensor[] attnNorm;

    /** Input norm of the feed-forward, every layer. */
    public final FloatTensor[] ffnNorm;

    // ---- dense feed-forward (the leading layers) ---------------------------

    public final FloatTensor[] ffnGate;
    public final FloatTensor[] ffnUp;
    public final FloatTensor[] ffnDown;

    // ---- routed feed-forward (every later layer) ---------------------------

    /** Router, {@code experts × dim}, F32. */
    public final FloatTensor[] ffnGateInp;

    /** Selection bias added to the router's probabilities, {@code experts}, F32. */
    public final FloatTensor[] expProbsBias;

    public final FloatTensor[] ffnGateExps;
    public final FloatTensor[] ffnUpExps;
    public final FloatTensor[] ffnDownExps;

    // ---- attention layers --------------------------------------------------

    public final FloatTensor[] wq;
    public final FloatTensor[] wk;
    public final FloatTensor[] wv;
    public final FloatTensor[] wo;
    public final FloatTensor[] attnQNorm;
    public final FloatTensor[] attnKNorm;

    // ---- short-convolution layers ------------------------------------------

    /** {@code dim → 3·dim}: the {@code B}, {@code C} and {@code x} chunks, in that order. */
    public final FloatTensor[] convInProj;

    /** Depthwise taps, {@code dim × kernel}, channel-major. */
    public final FloatTensor[] convKernel;

    public final FloatTensor[] convOutProj;

    // ---- head ----------------------------------------------------------------

    /** The final norm, stored as {@code token_embd_norm} in the file. */
    public final FloatTensor outputNorm;

    /** The vocabulary projection; the token embedding itself when the file ties them. */
    public final FloatTensor output;

    public final FloatTensor freqCisReal;
    public final FloatTensor freqCisImag;

    private final DataType weightType;

    // @formatter:off
    public Lfm2StandardWeights(
            FloatTensor tokenEmbeddingTable,
            FloatTensor[] attnNorm,
            FloatTensor[] ffnNorm,
            FloatTensor[] ffnGate,
            FloatTensor[] ffnUp,
            FloatTensor[] ffnDown,
            FloatTensor[] ffnGateInp,
            FloatTensor[] expProbsBias,
            FloatTensor[] ffnGateExps,
            FloatTensor[] ffnUpExps,
            FloatTensor[] ffnDownExps,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] attnQNorm,
            FloatTensor[] attnKNorm,
            FloatTensor[] convInProj,
            FloatTensor[] convKernel,
            FloatTensor[] convOutProj,
            FloatTensor outputNorm,
            FloatTensor output,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.ffnGate = ffnGate;
        this.ffnUp = ffnUp;
        this.ffnDown = ffnDown;
        this.ffnGateInp = ffnGateInp;
        this.expProbsBias = expProbsBias;
        this.ffnGateExps = ffnGateExps;
        this.ffnUpExps = ffnUpExps;
        this.ffnDownExps = ffnDownExps;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.convInProj = convInProj;
        this.convKernel = convKernel;
        this.convOutProj = convOutProj;
        this.outputNorm = outputNorm;
        this.output = output;
        this.freqCisReal = freqCisReal;
        this.freqCisImag = freqCisImag;
        this.weightType = weightType;
    }

    // @formatter:on

    @Override
    public DataType dataType() {
        return weightType;
    }
}
