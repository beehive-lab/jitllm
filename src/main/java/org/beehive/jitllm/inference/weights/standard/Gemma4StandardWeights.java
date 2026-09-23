package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Weights for the Gemma 4 architecture in the standard (CPU) format.
 *
 * <p>Gemma 4's layer structure differs substantially from the "Llama-like" models that {@link
 * StandardWeights} models, so this class implements {@link Weights} directly rather than extending
 * it: every layer carries its own Q/K-norm, a "sandwich" of pre- and post-normalization around both
 * attention and FFN, a per-layer-embedding (PLE) gate/projection/ norm, and an optional learned
 * output scale. There are also two separate RoPE frequency tables (sliding-window vs. full/global
 * attention layers use different bases, head dimensions, and — for full-attention layers — a
 * per-dimension frequency scaling baked in from {@code rope_freqs}).
 */
public class Gemma4StandardWeights implements Weights {

    public final FloatTensor tokenEmbeddingTable;
    public final FloatTensor outputWeight;
    public final FloatTensor outputNorm;

    // per-layer attention
    public final FloatTensor[] attnNorm;
    public final FloatTensor[] wq;
    public final FloatTensor[] wk;
    public final FloatTensor[] wv;
    public final FloatTensor[] wo;
    public final FloatTensor[] attnQNorm;
    public final FloatTensor[] attnKNorm;
    public final FloatTensor[] attnPostNorm; // a.k.a. post_attention_norm

    // per-layer FFN
    public final FloatTensor[] ffnNorm;
    public final FloatTensor[] ffnGate;
    public final FloatTensor[] ffnUp;
    public final FloatTensor[] ffnDown;
    public final FloatTensor[] ffnPostNorm; // a.k.a. post_ffw_norm

    // per-layer embedding (PLE)
    public final FloatTensor[] perLayerInpGate;
    public final FloatTensor[] perLayerProj;
    public final FloatTensor[] perLayerPostNorm; // a.k.a. post_norm
    public final FloatTensor[] layerOutputScale; // optional, may contain nulls

    // shared per-layer-embedding tensors

    /**
     * The per-layer token embedding table ({@code [embeddingLengthPerLayer * numberOfLayers,
     * vocabularySize]}, ~2.35 billion elements for Gemma-4-E2B). It is kept as a raw {@link
     * LongIndexedTensor} rather than a {@link FloatTensor} -- whose int-indexed API would overflow
     * for a tensor this large -- and addressed one embedding row at a time via {@code
     * CpuOperations.embeddingLookupLongIndexed}.
     */
    public final LongIndexedTensor perLayerTokenEmbd;

    public final FloatTensor perLayerModelProj;
    public final FloatTensor perLayerProjNorm;

    // RoPE tables: sliding-window (local) layers and full (global) attention layers use different
    // bases/dimensions; full-attention layers additionally bake in the `rope_freqs` per-dimension
    // scaling.
    public final FloatTensor freqCisRealSwa;
    public final FloatTensor freqCisImagSwa;
    public final FloatTensor freqCisRealFull;
    public final FloatTensor freqCisImagFull;

    private final DataType weightType;

    // @formatter:off
    public Gemma4StandardWeights(
            FloatTensor tokenEmbeddingTable,
            FloatTensor outputWeight,
            FloatTensor outputNorm,
            FloatTensor[] attnNorm,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] attnQNorm,
            FloatTensor[] attnKNorm,
            FloatTensor[] attnPostNorm,
            FloatTensor[] ffnNorm,
            FloatTensor[] ffnGate,
            FloatTensor[] ffnUp,
            FloatTensor[] ffnDown,
            FloatTensor[] ffnPostNorm,
            FloatTensor[] perLayerInpGate,
            FloatTensor[] perLayerProj,
            FloatTensor[] perLayerPostNorm,
            FloatTensor[] layerOutputScale,
            LongIndexedTensor perLayerTokenEmbd,
            FloatTensor perLayerModelProj,
            FloatTensor perLayerProjNorm,
            FloatTensor freqCisRealSwa,
            FloatTensor freqCisImagSwa,
            FloatTensor freqCisRealFull,
            FloatTensor freqCisImagFull,
            DataType weightType) {
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.outputWeight = outputWeight;
        this.outputNorm = outputNorm;
        this.attnNorm = attnNorm;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.attnPostNorm = attnPostNorm;
        this.ffnNorm = ffnNorm;
        this.ffnGate = ffnGate;
        this.ffnUp = ffnUp;
        this.ffnDown = ffnDown;
        this.ffnPostNorm = ffnPostNorm;
        this.perLayerInpGate = perLayerInpGate;
        this.perLayerProj = perLayerProj;
        this.perLayerPostNorm = perLayerPostNorm;
        this.layerOutputScale = layerOutputScale;
        this.perLayerTokenEmbd = perLayerTokenEmbd;
        this.perLayerModelProj = perLayerModelProj;
        this.perLayerProjNorm = perLayerProjNorm;
        this.freqCisRealSwa = freqCisRealSwa;
        this.freqCisImagSwa = freqCisImagSwa;
        this.freqCisRealFull = freqCisRealFull;
        this.freqCisImagFull = freqCisImagFull;
        this.weightType = weightType;
    }

    // @formatter:on

    @Override
    public DataType dataType() {
        return weightType;
    }
}
