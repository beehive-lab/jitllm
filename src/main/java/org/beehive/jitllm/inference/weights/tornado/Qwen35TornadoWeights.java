package org.beehive.jllm.inference.weights.tornado;

import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Device weights for the {@code qwen35} architecture.
 *
 * <p>Extends {@link TornadoWeights} rather than implementing {@link Weights} directly, which is the
 * opposite of what the host {@code Qwen35StandardWeights} does. The reason is reach rather than
 * symmetry: {@code Activation}, {@code AbstractLogitsTaskGraph} and {@code TornadoForwardPass} are
 * written against {@code TornadoWeights}, and a family outside that type would need each of them
 * changed to accommodate it. Fitting the existing shape costs a documented convention; not fitting
 * it costs edits to shared classes for one family's benefit.
 *
 * <p>The base class's per-layer arrays are indexed by <b>absolute block index</b>, and are {@code
 * null} where the block is of the other kind:
 *
 * <ul>
 *   <li>{@code wqLayered}, {@code wkLayered}, {@code wvLayered}, {@code woLayered} — the attending
 *       blocks only, {@code null} at the 48 recurrent layers, whose mixer has no query, key or
 *       value projection at all. {@code wqLayered} is twice a query projection's width here,
 *       carrying an interleaved output gate.
 *   <li>{@code rms_att_weightLayered}, {@code rms_ffn_weightLayered}, {@code w1/w2/w3Layered} —
 *       every block, of either kind, because both mixers are followed by the same dense
 *       feed-forward. {@code rms_ffn_weightLayered} is the file's {@code post_attention_norm}.
 * </ul>
 *
 * <p>The recurrent layers' own weights are the fields below, indexed the same way and {@code null}
 * at the attending blocks. A second, compacted numbering would be the kind of off-by-one that
 * produces fluent, wrong text; the dense indices that do exist ({@code keyValueLayerIndex} and
 * {@code recurrentLayerIndex}) address <i>state</i>, not weights.
 *
 * <p>{@code weightType} is the representation the <b>projections</b> are retained in. It is not a
 * claim that every tensor here is that type — this model is mixed by construction, with Q5_K
 * recurrent outputs and a Q6_K vocabulary projection — and a layer graph dispatches per tensor.
 */
public final class Qwen35TornadoWeights extends TornadoWeights {

    /** Trunk layers plus MTP blocks; the length of every per-layer array here. */
    public final int blockCount;

    /** Per-head query norm, attending blocks only. */
    public final TornadoTensor[] attnQNorm;

    /** Per-head key norm, attending blocks only. */
    public final TornadoTensor[] attnKNorm;

    /** Fused {@code q ‖ k ‖ v} projection feeding the depthwise convolution. */
    public final TornadoTensor[] ssmQkv;

    /** The {@code z} gate the delta-net output is normalized against. */
    public final TornadoTensor[] ssmGate;

    /** Depthwise causal convolution kernel, {@code conv_kernel} taps per channel. */
    public final TornadoTensor[] ssmConv1d;

    /** Per-value-head decay projection, before the bias, softplus and {@link #ssmA}. */
    public final TornadoTensor[] ssmAlpha;

    /** Per-value-head write-strength projection, before the logistic. */
    public final TornadoTensor[] ssmBeta;

    /** Bias added to the decay projection before the softplus. */
    public final TornadoTensor[] ssmDtBias;

    /** {@code -exp(A_log)}: multiplies the softplus to give the log decay. */
    public final TornadoTensor[] ssmA;

    /** Gated RMS norm scale over one value head's width. */
    public final TornadoTensor[] ssmNorm;

    /** Output projection of the recurrent branch. Q5_K in the 27B. */
    public final TornadoTensor[] ssmOut;

    // @formatter:off
    public Qwen35TornadoWeights(
            int blockCount,
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] attnNorm,
            TornadoTensor[] ffnNorm,
            TornadoTensor[] ffnGate,
            TornadoTensor[] ffnDown,
            TornadoTensor[] ffnUp,
            TornadoTensor outputNorm,
            TornadoTensor output,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] attnQNorm,
            TornadoTensor[] attnKNorm,
            TornadoTensor[] ssmQkv,
            TornadoTensor[] ssmGate,
            TornadoTensor[] ssmConv1d,
            TornadoTensor[] ssmAlpha,
            TornadoTensor[] ssmBeta,
            TornadoTensor[] ssmDtBias,
            TornadoTensor[] ssmA,
            TornadoTensor[] ssmNorm,
            TornadoTensor[] ssmOut,
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
        this.blockCount = blockCount;
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.ssmQkv = ssmQkv;
        this.ssmGate = ssmGate;
        this.ssmConv1d = ssmConv1d;
        this.ssmAlpha = ssmAlpha;
        this.ssmBeta = ssmBeta;
        this.ssmDtBias = ssmDtBias;
        this.ssmA = ssmA;
        this.ssmNorm = ssmNorm;
        this.ssmOut = ssmOut;
    }
    // @formatter:on
}
