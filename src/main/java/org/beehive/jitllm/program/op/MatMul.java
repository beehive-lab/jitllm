package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * A weight matrix times a batch of activation rows.
 *
 * <p>What prefill and batched decode do, and what the tensor-core paths exist for. The batch size
 * is part of the description rather than an invocation value because the buffers are sized for it
 * and the plan is captured for it: a step may have fewer active rows than the batch it was built
 * for, but it may never have more.
 *
 * @param weight the weight matrix
 * @param input the activation rows, {@code batchSize} of them
 * @param output the resulting rows
 * @param rows rows of the weight matrix
 * @param columns columns of the weight matrix, and the width of one activation row
 * @param batchSize the number of rows the buffers are sized for
 * @param dataType the representation the weight matrix was materialized in
 */
public record MatMul(
        OperandRef.Weight weight,
        OperandRef input,
        OperandRef output,
        int rows,
        int columns,
        int batchSize,
        DataType dataType)
        implements Operation {

    public MatMul {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException(
                    "rows and columns must be positive: " + rows + "x" + columns);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.MAT_MUL;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(weight, input);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
