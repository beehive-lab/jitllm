package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Greedy selection: the identifier of the highest-scoring logit.
 *
 * <p>Sampling is an operation and may execute on the device (Rule 8b), which is what makes the
 * existing {@code jitllm.deviceSample} path expressible rather than a special case. Rule 14 is not in
 * tension with that: it forbids core abstractions from <i>requiring</i> a sampler, not from naming
 * one.
 *
 * <p>Greedy selection takes no parameters, which is exactly what separates it from {@link Sample}.
 *
 * @param logits the scores to choose from
 * @param output where the chosen identifier is written
 * @param dataType the representation the logits are held in
 */
public record ArgMax(OperandRef logits, OperandRef output, DataType dataType) implements Operation {

    public ArgMax {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.ARG_MAX;
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
