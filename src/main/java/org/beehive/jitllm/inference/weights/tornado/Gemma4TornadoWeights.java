package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.LongIndexedTensor;

/**
 * TornadoVM (GPU) weights for the Gemma 4 architecture.
 *
 * <p>Extends {@link TornadoWeights} (rather than implementing {@link
 * org.beehive.jitllm.inference.weights.Weights} directly, as its CPU counterpart {@link
 * org.beehive.jitllm.inference.weights.standard.Gemma4StandardWeights} does) because the shared
 * {@code AbstractLogitsLayer}/{@code LogitsFP16Layer} GPU infrastructure requires a {@link
 * TornadoWeights}. The base class's "Llama-like" fields are reused for the closest equivalents
 * (e.g. {@code rms_att_weightLayered} &rarr; {@code attnNorm}, {@code w1Layered}/{@code w3Layered}
 * &rarr; {@code ffnGate}/{@code ffnUp}); every other Gemma 4-specific tensor (sandwich norms,
 * Q/K-norm, per-layer-embedding (PLE) gate/proj/norm, dual RoPE tables, optional layer-output
 * scale) is added here.
 *
 * <p>Note: {@code per_layer_token_embd} is intentionally <b>not</b> present here -- at ~2.35
 * billion elements it is far too large to keep resident on the GPU. Its per-token row is instead
 * gathered on the host (see {@code CpuOperations.embeddingLookupLongIndexed}) and streamed to the
 * GPU each step via {@link
 * org.beehive.jitllm.inference.state.Gemma4State#wrapPerLayerTokenEmbedRow}.
 */
public class Gemma4TornadoWeights extends TornadoWeights {

    // Gemma4-specific per-layer attention tensors (sandwich norm + Q/K-norm)
    public final TornadoTensor[] attnQNorm;
    public final TornadoTensor[] attnKNorm;
    public final TornadoTensor[] attnPostNorm;

    // Gemma4-specific per-layer FFN tensor (sandwich norm)
    public final TornadoTensor[] ffnPostNorm;

    // per-layer embedding (PLE)
    public final TornadoTensor[] perLayerInpGate;
    public final TornadoTensor[] perLayerProj;
    public final TornadoTensor[] perLayerPostNorm;
    public final TornadoTensor[] layerOutputScale; // optional, may contain nulls

    // shared per-layer-embedding tensors

    /**
     * The per-layer token embedding table ({@code [embeddingLengthPerLayer * numberOfLayers,
     * vocabularySize]}, ~2.35 billion elements for Gemma-4-E2B). Far too large to keep resident on
     * the GPU, so it is kept as a raw {@link LongIndexedTensor} and addressed one row at a time on
     * the host -- via {@code CpuOperations.embeddingLookupLongIndexed} -- with the resulting row
     * streamed to the GPU each step (see {@link
     * org.beehive.jitllm.inference.state.Gemma4State#wrapPerLayerTokenEmbedRow}).
     */
    public final LongIndexedTensor perLayerTokenEmbd;

    public final TornadoTensor perLayerModelProj;
    public final TornadoTensor perLayerProjNorm;

    // RoPE tables: sliding-window (local) layers and full (global) attention layers use different
    // bases/dimensions.
    public final TornadoTensor freqCisRealSwa;
    public final TornadoTensor freqCisImagSwa;
    public final TornadoTensor freqCisRealFull;
    public final TornadoTensor freqCisImagFull;

    // @formatter:off
    public Gemma4TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] attnNorm,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] attnQNorm,
            TornadoTensor[] attnKNorm,
            TornadoTensor[] attnPostNorm,
            TornadoTensor[] ffnNorm,
            TornadoTensor[] ffnGate,
            TornadoTensor[] ffnUp,
            TornadoTensor[] ffnDown,
            TornadoTensor[] ffnPostNorm,
            TornadoTensor[] perLayerInpGate,
            TornadoTensor[] perLayerProj,
            TornadoTensor[] perLayerPostNorm,
            TornadoTensor[] layerOutputScale,
            LongIndexedTensor perLayerTokenEmbd,
            TornadoTensor perLayerModelProj,
            TornadoTensor perLayerProjNorm,
            TornadoTensor outputNorm,
            TornadoTensor freqCisRealSwa,
            TornadoTensor freqCisImagSwa,
            TornadoTensor freqCisRealFull,
            TornadoTensor freqCisImagFull,
            TornadoTensor outputWeight,
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
                freqCisRealFull,
                freqCisImagFull,
                outputWeight,
                weightType);
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.attnPostNorm = attnPostNorm;
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
    }
    // @formatter:on
}
