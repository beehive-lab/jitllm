package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * A weight matrix times one activation vector.
 *
 * <p>The decode path's workhorse: every projection in a single-token step is one of these. It is
 * kept distinct from {@link MatMul} because the two are genuinely different work on a device — a
 * matrix-vector product is bandwidth-bound and a matrix-matrix product is not, and they reach
 * different kernels — not because the shapes could not be unified on paper.
 *
 * <p><b>Where dequantization went.</b> Nowhere: it stays inside whichever implementation needs it.
 * A backend whose weight representation is encoded decodes blocks in its own inner loop, as the CPU
 * does for the K-quants; a backend that cannot execute a representation was given a materialized
 * fallback at load, as the GPU is given {@code Q8_0}. There is no decode operation to place before
 * this one.
 *
 * @param weight the weight matrix
 * @param input the activation vector
 * @param output the resulting vector
 * @param rows rows of the weight matrix, and the length of {@code output}
 * @param columns columns of the weight matrix, and the length of {@code input}
 * @param dataType the representation the weight matrix was materialized in
 */
public record MatVec(
        OperandRef.Weight weight,
        OperandRef input,
        OperandRef output,
        int rows,
        int columns,
        DataType dataType)
        implements Operation {

    public MatVec {
        Objects.requireNonNull(weight, "weight");
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
        return OperationKind.MAT_VEC;
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
