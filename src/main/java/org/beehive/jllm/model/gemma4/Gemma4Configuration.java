package org.beehive.jllm.model.gemma4;

import org.beehive.jllm.model.Configuration;

/**
 * Configuration for the Gemma 4 architecture (e.g. Gemma-4-E2B-It).
 *
 * <p>Gemma 4 alternates sliding-window and full (global) attention layers, each with their own head
 * dimensions, RoPE base/scaling, and a subset of layers reusing the KV cache produced by an earlier
 * layer ("shared KV layers"). It also augments every layer with a per-layer embedding (PLE)
 * mechanism and applies a final logit soft-cap.
 */
// @formatter:off
public record Gemma4Configuration(
        String quantization,
        int dim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int headDimSwa,
        int headDimFull,
        int[] feedForwardLength,
        boolean[] slidingWindowPattern,
        int slidingWindowSize,
        int sharedKvLayers,
        int embeddingLengthPerLayer,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        float rmsNormEps,
        float ropeTheta,
        float ropeThetaSwa,
        float finalLogitSoftcapping)
        implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int hiddenDim() {
        throw new UnsupportedOperationException(
                "Gemma4 has per-layer feed-forward dimensions; use feedForwardLength(layer).");
    }

    @Override
    public int numberOfHeadsKey() {
        throw new UnsupportedOperationException(
                "Gemma4 has per-layer head dimensions; use headDim(layer).");
    }

    @Override
    public int headSize() {
        throw new UnsupportedOperationException(
                "Gemma4 has per-layer head dimensions; use headDim(layer).");
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException(
                "Gemma4 has per-layer head dimensions; use headDim(layer) * numberOfKeyValueHeads().");
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }

    /** Returns the feed-forward (FFN hidden) dimension for the given layer. */
    public int feedForwardLength(int layer) {
        return feedForwardLength[layer];
    }

    /**
     * Whether the given layer uses sliding-window (local) attention as opposed to full (global)
     * attention.
     */
    public boolean isSwa(int layer) {
        return slidingWindowPattern[layer];
    }

    /**
     * Returns the attention head dimension for the given layer (depends on whether it is a
     * sliding-window or full layer).
     */
    public int headDim(int layer) {
        return isSwa(layer) ? headDimSwa : headDimFull;
    }

    /** The maximum head dimension across all layers; used to size shared scratch buffers. */
    public int maxHeadDim() {
        return Math.max(headDimSwa, headDimFull);
    }

    /**
     * The maximum feed-forward dimension across all layers; used to size shared scratch buffers.
     */
    public int maxFeedForwardLength() {
        int max = 0;
        for (int ff : feedForwardLength) {
            max = Math.max(max, ff);
        }
        return max;
    }

    /**
     * How many slices the decode attention window is cut into, or 1 for the single-pass kernel.
     *
     * <p>Lives on the configuration because two places need the same answer and must not drift: the
     * state sizes the split scratch from it, and the layer graph launches from it. A disagreement
     * is an out-of-bounds write into that scratch rather than an error.
     *
     * <p>Derived from this model's own attention shape. One workgroup per head is {@code
     * numberOfHeads()} of them — eight here, on a device with more than a hundred multiprocessors —
     * so the window is split until there is enough resident work, then capped so the slices do not
     * become too fine to amortise their own combine: at least 32 positions per slice at the
     * shortest window this model uses.
     *
     * <p><b>{@code TARGET_WORKGROUPS} is an approximation and is written as one.</b> It stands for
     * "enough workgroups to fill a modern GPU" and happens to be the multiprocessor count of the
     * device this was measured on. The honest form is a device capability; there is no compute-unit
     * count on the backend's {@code Device} contract today, and adding one is a change to a backend
     * interface rather than something a decode kernel should do quietly.
     */
    public int attentionSplits() {
        final int targetWorkgroups = 128;
        final int minPositionsPerSlice = 32;
        if (contextLength < 512) {
            return 1;
        }
        int bySpread = Math.max(1, targetWorkgroups / numberOfHeads);
        int byWindow =
                Math.max(1, Math.min(slidingWindowSize, contextLength) / minPositionsPerSlice);
        return Math.min(bySpread, byWindow);
    }

    /**
     * Number of (initial) layers that own and populate their own KV cache; later layers reuse one
     * of these.
     */
    public int nLayerKvFromStart() {
        return numberOfLayers - sharedKvLayers;
    }

    /**
     * Whether the given layer computes and stores its own K/V (as opposed to reusing an earlier
     * layer's KV cache).
     */
    public boolean hasOwnKv(int layer) {
        return layer < nLayerKvFromStart();
    }

    /**
     * Returns the index of the layer whose KV cache this layer reuses, or -1 if this layer owns its
     * KV cache.
     */
    public int kvReuseLayer(int layer) {
        if (hasOwnKv(layer)) {
            return -1;
        }
        return nLayerKvFromStart() - (isSwa(layer) ? 2 : 1);
    }
}
// @formatter:on
