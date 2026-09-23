package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Softmax over the last dimension, one row at a time.
 *
 * <p>Listed as its own operation because it is one on the CPU path, where attention scores are
 * normalized as a separate step. A backend that fuses it into {@link Attention} — as the
 * flash-attention style kernels do — simply has no separate implementation to reach, which is a
 * property of that backend's mapping and not a reason to leave the operation unnamed.
 *
 * @param input the scores
 * @param output the normalized scores; may be {@code input}
 * @param rows how many independent rows are normalized
 * @param columns the length of one row
 * @param dataType the representation the scores are held in
 */
public record Softmax(OperandRef input, OperandRef output, int rows, int columns, DataType dataType)
        implements Operation {

    public Softmax {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException(
                    "rows and columns must be positive: " + rows + "x" + columns);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SOFTMAX;
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
