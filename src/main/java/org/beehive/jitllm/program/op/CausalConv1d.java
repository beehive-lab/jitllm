package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * One step of a depthwise causal convolution over a channel vector, against a retained window.
 *
 * <p><b>Depthwise:</b> each channel is convolved with its own taps and never mixes with another, so
 * this is {@code channels} independent dot products of length {@code kernel} rather than a matrix
 * multiply. <b>Causal:</b> the taps cover this step and the {@code kernel - 1} before it, never
 * ahead.
 *
 * <p>The window is <b>state</b>, which is what separates this from an elementwise operation over
 * the same weights, and why it appears as an operand rather than being hidden: an implementation
 * owns advancing it, and a description that did not name it would allow a backend to advance it
 * twice or not at all.
 *
 * <p>The first convolution in this vocabulary, and not a Qwen3.5 invention — Mamba, Mamba2 and
 * Qwen3-Next all place one before their state update.
 *
 * @param input this step's value per channel
 * @param weight the taps, channel-major: {@code channels * kernel}, oldest tap first
 * @param window the retained inputs, {@code channels * (kernel - 1)}, updated in place
 * @param output the convolved result, one value per channel
 * @param channels how many independent channels
 * @param kernel how many taps, this step included
 * @param dataType the representation the convolution executes at
 */
public record CausalConv1d(
        OperandRef input,
        OperandRef.Weight weight,
        OperandRef window,
        OperandRef output,
        int channels,
        int kernel,
        DataType dataType)
        implements Operation {

    public CausalConv1d {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (channels < 1) {
            throw new IllegalArgumentException("channels must be at least 1: " + channels);
        }
        if (kernel < 1) {
            throw new IllegalArgumentException("kernel must be at least 1: " + kernel);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.CAUSAL_CONV_1D;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input, weight, window);
    }

    /** The window is written as well as read: it carries this step forward. */
    @Override
    public List<OperandRef> outputs() {
        return List.of(output, window);
    }
}
