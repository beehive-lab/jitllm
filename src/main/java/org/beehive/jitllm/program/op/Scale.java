package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Multiplying every element by one scalar from the model's configuration.
 *
 * <p>Granite is why this exists. It applies maximal-update-parameterization (µP) factors at four
 * points a Llama-shaped model has none: after the embedding lookup, on each residual branch before
 * it is added back, and on the logits. Gemma scales its embeddings the same way.
 *
 * <p><b>The factor is configuration, not an invocation value.</b> It comes from the model file and
 * is the same for every token, so it belongs in the description — unlike a sampling temperature,
 * which changes per request and would compile a program per value.
 *
 * <p>Deliberately not folded into the neighbouring operation. A "{@code MatVec} with an optional
 * output scale" would give every family a parameter only one uses, and an implementation that has
 * to check it. Naming the scale keeps the families that do not scale free of it, and makes
 * Granite's four scaling points visible in the program rather than hidden inside four other
 * operations.
 *
 * @param input the values to scale
 * @param output the scaled values; may be {@code input}
 * @param factor the scalar from the model configuration
 * @param dataType the representation the multiplication executes at
 */
public record Scale(OperandRef input, OperandRef output, float factor, DataType dataType)
        implements Operation {

    public Scale {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SCALE;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
