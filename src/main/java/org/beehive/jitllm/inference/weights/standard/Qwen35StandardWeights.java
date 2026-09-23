package org.beehive.jllm.inference.weights.standard;

import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Host weights for the {@code qwen35} architecture.
 *
 * <p>Implements {@link Weights} directly rather than extending {@link StandardWeights}, because
 * that base class assumes every layer has a query, key and value projection. Here three layers in
 * four have none: they mix with a delta-net recurrence and carry an entirely different weight set.
 * Extending it would mean 48 nulls per array and a base class whose fields lie about the model.
 *
 * <p>Every per-layer array is nonetheless indexed by <b>absolute block index</b>, not by a per-kind
 * counter. An entry is {@code null} where the layer is of the other kind. That keeps the forward
 * pass reading {@code wq[l]} with the same {@code l} it uses everywhere else — a second, compacted
 * indexing is exactly the sort of off-by-one that produces fluent, wrong text.
 *
 * <p>The arrays are sized for {@link #blockCount} blocks: the trunk plus its MTP blocks, so an MTP
 * block's attention and feed-forward weights sit at their own index alongside the trunk's.
 */
public final class Qwen35StandardWeights implements Weights {

    /** Trunk layers plus MTP blocks; the length of every per-layer array here. */
    public final int blockCount;

    // ---- shared by both layer kinds ---------------------------------------

    public final FloatTensor tokenEmbeddingTable;

    /** Input norm of the mixer branch, every block. */
    public final FloatTensor[] attnNorm;

    /** Input norm of the feed-forward branch. Named {@code post_attention_norm} in the file. */
    public final FloatTensor[] ffnNorm;

    public final FloatTensor[] ffnGate;
    public final FloatTensor[] ffnDown;
    public final FloatTensor[] ffnUp;

    public final FloatTensor outputNorm;
    public final FloatTensor output;

    /** RoPE tables, precomputed over {@code rope.dimension_count} rather than the head width. */
    public final FloatTensor freqCisReal;

    public final FloatTensor freqCisImag;

    // ---- attention layers --------------------------------------------------

    /**
     * The fused query/gate projection: per head, a query slice then a gate slice, interleaved.
     * Twice a plain query projection's width.
     */
    public final FloatTensor[] wq;

    public final FloatTensor[] wk;
    public final FloatTensor[] wv;
    public final FloatTensor[] wo;
    public final FloatTensor[] attnQNorm;
    public final FloatTensor[] attnKNorm;

    // ---- recurrent (Gated Delta Net) layers --------------------------------

    /** Fused {@code q ‖ k ‖ v} projection feeding the depthwise convolution. */
    public final FloatTensor[] ssmQkv;

    /** The {@code z} gate the delta-net output is normalized against. */
    public final FloatTensor[] ssmGate;

    /** Depthwise causal convolution kernel, {@code conv_kernel} taps per channel. */
    public final FloatTensor[] ssmConv1d;

    /** Per-value-head decay projection, before the bias, softplus and {@code ssmA}. */
    public final FloatTensor[] ssmAlpha;

    /** Per-value-head write-strength projection, before the logistic. */
    public final FloatTensor[] ssmBeta;

    /** Bias added to the decay projection before the softplus. */
    public final FloatTensor[] ssmDtBias;

    /** {@code -exp(A_log)}: multiplies the softplus to give the log decay. */
    public final FloatTensor[] ssmA;

    /** Gated RMS norm scale over one value head's width. */
    public final FloatTensor[] ssmNorm;

    /** Output projection of the recurrent branch. */
    public final FloatTensor[] ssmOut;

    // ---- MTP / NextN blocks -------------------------------------------------

    /** Norm applied to the drafted token's embedding before it is concatenated. */
    public final FloatTensor[] nextnENorm;

    /** Norm applied to the trunk's final hidden state before it is concatenated. */
    public final FloatTensor[] nextnHNorm;

    /** Projects the concatenated pair back down to {@code dim}. */
    public final FloatTensor[] nextnEhProj;

    /** Final norm before the shared LM head. */
    public final FloatTensor[] nextnSharedHeadNorm;

    private final DataType weightType;

    // @formatter:off
    public Qwen35StandardWeights(
            int blockCount,
            FloatTensor tokenEmbeddingTable,
            FloatTensor[] attnNorm,
            FloatTensor[] ffnNorm,
            FloatTensor[] ffnGate,
            FloatTensor[] ffnDown,
            FloatTensor[] ffnUp,
            FloatTensor outputNorm,
            FloatTensor output,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] attnQNorm,
            FloatTensor[] attnKNorm,
            FloatTensor[] ssmQkv,
            FloatTensor[] ssmGate,
            FloatTensor[] ssmConv1d,
            FloatTensor[] ssmAlpha,
            FloatTensor[] ssmBeta,
            FloatTensor[] ssmDtBias,
            FloatTensor[] ssmA,
            FloatTensor[] ssmNorm,
            FloatTensor[] ssmOut,
            FloatTensor[] nextnENorm,
            FloatTensor[] nextnHNorm,
            FloatTensor[] nextnEhProj,
            FloatTensor[] nextnSharedHeadNorm,
            DataType weightType) {
        this.blockCount = blockCount;
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.ffnGate = ffnGate;
        this.ffnDown = ffnDown;
        this.ffnUp = ffnUp;
        this.outputNorm = outputNorm;
        this.output = output;
        this.freqCisReal = freqCisReal;
        this.freqCisImag = freqCisImag;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
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
        this.nextnENorm = nextnENorm;
        this.nextnHNorm = nextnHNorm;
        this.nextnEhProj = nextnEhProj;
        this.nextnSharedHeadNorm = nextnSharedHeadNorm;
        this.weightType = weightType;
    }

    // @formatter:on

    @Override
    public DataType dataType() {
        return weightType;
    }
}
