package org.beehive.jllm.model.qwen3;

import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.runtime.tensor.DataType;

// @formatter:off
public record Qwen3Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int numberOfHeadsKey,
        int numberOfHeadsValue,
        int vocabularySize,
        int contextLengthModel,
        int contextLength,
        boolean sharedWeights,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    /**
     * One: batched prefill lays out a prefill and a decode family of per-layer graphs, but the
     * decode graphs consume the weights the prefill graphs uploaded (their {@code
     * weightSourceGraphName}), so the plan holds the weights once. Measured: Llama-3.2-1B F16
     * batched prefill runs in a 2600 MB budget and Qwen3-8B F16 in 17 GB, where counting the
     * weights twice predicted 4.2 GB and 29.2 GB and refused the second. The native prefill path's
     * stacked projection copies are a separate component, not this count.
     */
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

    @Override
    public String quantization() {
        return quantization;
    }

    // @formatter:off
    /**
     * The per-head width, which this family carries in the file rather than deriving.
     *
     * <p>This used to throw. The derivation the interface documents — {@code dim / numberOfHeads} —
     * is wrong here, and throwing was the safe answer while nothing neutral asked. Something does
     * now: the memory model needs the attention scratch, and a throw made every prediction for this
     * architecture fail and be swallowed, so the preflight silently did nothing. The quantity is
     * not unsupported, it is simply read from a different field.
     */
    // @formatter:on
    @Override
    public int headSize() {
        return numberOfHeadsKey;
    }

    /** Key/value width per token per layer: the value head width across the key/value heads. */
    @Override
    public int kvDim() {
        return numberOfHeadsValue * numberOfKeyValueHeads;
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    // @formatter:off
    /**
     * Bytes of stacked projection weights a native batch-prefill family keeps per layer, or zero
     * where it keeps none.
     *
     * <p>cuBLAS takes one B operand and the binding has no operand offset, so a single GEMM over
     * {@code [q|k|v]} or {@code [gate|up]} needs those operands adjacent in memory — a copy, not a
     * view. The originals stay resident because the decode graphs and the generated-kernel path
     * still read them, so this is genuinely additional.
     *
     * <p>FP16 only, and the check is on the file's own representation rather than on a flag: the
     * native path passes dense half-precision matrices straight to the library, and a quantized
     * file would have to be dequantized into a staging copy this figure does not describe. The test
     * is {@link #activationType()} rather than the quantization string, because that is where the
     * interface already decides what half precision means.
     */
    // @formatter:on
    @Override
    public long nativeStackedProjectionBytesPerLayer() {
        if (activationType() != DataType.F16) {
            return 0L;
        }
        long qDim = (long) numberOfHeadsKey * numberOfHeads;
        long kvWidth = kvDim();
        return (long) dim * (qDim + 2 * kvWidth + 2 * hiddenDim) * Short.BYTES;
    }

    // @formatter:off
    /**
     * Bytes of contiguous staging the fused attention adapters need for one chunk, or zero.
     *
     * <p>Four tensors — query, the group-expanded key and value, and the output — each {@code
     * [head][token][headDim]} in half precision, allocated once for the whole execution plan rather
     * than once per layer graph.
     */
    // @formatter:on
    @Override
    public long nativeAttentionStagingBytes(int batchSize) {
        if (activationType() != DataType.F16) {
            return 0L;
        }
        long qDim = (long) numberOfHeadsKey * numberOfHeads;
        return 4L * qDim * batchSize * Short.BYTES;
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }
}
