package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Stochastic selection of the next token from the logits.
 *
 * <p><b>Temperature, top-p and the seed are not here.</b> They change per request and are delivered
 * as invocation values — written into a persistent control array before the invocation, never by
 * rebinding one. Carrying them in the description would put them in the program signature, and
 * therefore in the compiled-program cache key, which would compile a separate program for every
 * temperature a caller ever asks for.
 *
 * <p>That is the whole difference from {@link ArgMax}: same operands, parameters supplied later.
 * Which of the two a program contains is an execution-policy choice resolved once at session
 * creation, not a per-token branch.
 *
 * @param logits the scores to sample from
 * @param output where the chosen identifier is written
 * @param dataType the representation the logits are held in
 */
public record Sample(OperandRef logits, OperandRef output, DataType dataType) implements Operation {

    public Sample {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SAMPLE;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(logits);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
