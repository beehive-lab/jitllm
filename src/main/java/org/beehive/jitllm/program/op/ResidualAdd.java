package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Elementwise addition of a residual branch back into the main activation.
 *
 * <p>Trivial arithmetic, named anyway: it is where the residual stream is, and a program that does
 * not say so is a program whose data flow has to be inferred from buffer names.
 *
 * @param left the main activation
 * @param right the branch result being added back
 * @param output the sum; may be {@code left}
 * @param dataType the representation the residual stream is held in
 */
public record ResidualAdd(OperandRef left, OperandRef right, OperandRef output, DataType dataType)
        implements Operation {

    public ResidualAdd {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.RESIDUAL_ADD;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(left, right);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
