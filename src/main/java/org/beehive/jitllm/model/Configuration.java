package org.beehive.jllm.model;

import org.beehive.jllm.runtime.tensor.DataType;

public interface Configuration {

    /**
     * @deprecated Use {@link #activationType()}. This string comes from GGUF's {@code
     *     general.file_type} and describes the <i>file</i>, which is the least reliable of the
     *     three notions of "the model's type" the code carries: a K-quant file reports {@code
     *     "Q8_0"} because that is what its activations end up as, not because that is what is in
     *     it. The per-tensor {@code DataType} on a descriptor is the truth.
     */
    @Deprecated
    String quantization();

    /**
     * The representation activations are held in.
     *
     * <p>Not the weights' type: an FP16 model keeps FP16 activations, and everything quantized
     * quantizes its activations to Q8_0 to match the kernels that consume them.
     */
    default DataType activationType() {
        return "FP16".equals(quantization()) ? DataType.F16 : DataType.Q8_0;
    }

    /** Transformer embedding dimension */
    int dim();

    /** Hidden dimension size for feed-forward network layers */
    int hiddenDim();

    /** Number of transformer layers in the model */
    int numberOfLayers();

    /** Number of attention heads for queries */
    int numberOfHeads();

    /** Number of key/value heads (can be fewer than query heads in multi-query attention) */
    int numberOfKeyValueHeads();

    int numberOfHeadsKey();

    // @formatter:off
    /**
     * How many layers hold key/value entries.
     *
     * <p>Every layer, for a stack that is attention throughout — which is every family but one.
     * {@code qwen35} attends in one layer of four and mixes the rest with a recurrence that retains
     * nothing per position, so its key/value store is sized by this rather than by the layer count,
     * and a memory prediction built from {@link #numberOfLayers()} over-predicts it fourfold.
     */
    // @formatter:on
    default int keyValueLayerCount() {
        return numberOfLayers();
    }

    // @formatter:off
    /**
     * How many of a layout's graph families actually <b>bind</b> the per-layer weights.
     *
     * <p>The layout says how many families of per-layer graphs a mode builds; the Tornado runtime
     * allocates a device buffer per graph that binds an array, so that count is the multiplier on
     * per-layer weight memory — <i>unless</i> a family consumes the copy another uploaded. A family
     * that consumes costs graphs, not gigabytes.
     *
     * <p>Defaults to the layout's own count, which is what a family whose graphs each upload their
     * own weights should report. Over-predicting here is not the safe direction it usually is: this
     * prediction can <b>refuse</b> a load, so a model claiming twice the weights it holds is
     * refused on a device it fits.
     */
    // @formatter:on
    default int weightBindingFamilies(int layerGraphFamilies) {
        return layerGraphFamilies;
    }

    // @formatter:off
    /**
     * Bytes of <b>stacked</b> projection weights a native batch-prefill family keeps per layer.
     *
     * <p>Zero unless a family rewrites its projections as single library GEMMs over operands that
     * have to be contiguous, which makes the stacked forms copies rather than views. The originals
     * are not freed — other graphs still read them — so this is additional to the weight footprint
     * and not a redistribution of it.
     */
    // @formatter:on
    default long nativeStackedProjectionBytesPerLayer() {
        return 0L;
    }

    // @formatter:off
    /**
     * Bytes of contiguous staging a native fused attention needs for one prefill chunk.
     *
     * <p>Zero unless a family bridges its own layout to a library's. Sized from the batch width and
     * allocated once for the execution plan, not once per layer graph.
     */
    // @formatter:on
    default long nativeAttentionStagingBytes(int batchSize) {
        return 0L;
    }

    // @formatter:off
    /**
     * Bytes of chunk-wide scratch a family allocates beyond what the generic batch staging covers.
     *
     * <p>Zero unless a family's batched graphs need buffers the generic ones cannot describe — a
     * projection twice a query's width, a convolved {@code q ‖ k ‖ v} of unequal parts. Sized from
     * the batch width, and zero when there is no batch.
     */
    // @formatter:on
    default long additionalBatchWorkspaceBytes(int batchSize) {
        return 0L;
    }

    // @formatter:off
    /**
     * Bytes of per-session state that is neither key/value cache nor scratch.
     *
     * <p>Zero for a stack that is attention throughout. A recurrent layer keeps its history in a
     * fixed-size state instead of in a growing cache — a convolution window and a delta-net matrix
     * per head — which persists across tokens, is updated in place, and is sized from the
     * configuration rather than from the context length. It is not scratch, so a workspace figure
     * derived from the transformer's dimensions does not include it.
     */
    // @formatter:on
    default long recurrentStateBytes() {
        return 0L;
    }

    /** Size of the vocabulary (token set) */
    int vocabularySize();

    /** Maximum sequence length the model can process */
    int contextLength();

    /** Max sequence length in model */
    int contextLengthModel();

    /** Epsilon value for RMSNorm layers (stabilizes normalization) */
    float rmsNormEps();

    /** Base value for RoPE (Rotary Position Embedding) calculations */
    float ropeTheta();

    int headSize();

    int kvDim();

    int kvMul();
}
