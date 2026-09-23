package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Scaling to unit length: {@code output = input / ||input||₂}, per group.
 *
 * <p>Not {@link RmsNorm} with a weight of one, and the difference is not a detail. RMS
 * normalization divides by the root <i>mean</i> square, which is smaller by {@code √length}, and it
 * carries a learned scale. There is no weight here at all — the operand simply lands on the unit
 * sphere. Qwen3.5's delta-net layers normalize their convolved queries and keys this way before the
 * state update, once per head.
 *
 * <p>Epsilon is a <b>floor on the divisor</b> rather than a term added under the root, matching
 * what the reference implementation does. The two agree to rounding for a healthy operand and
 * disagree sharply for one whose norm approaches epsilon, which is exactly the case epsilon is
 * there for.
 *
 * @param input the activations to normalize
 * @param output where the normalized activations are written; may be {@code input}
 * @param groups how many independent groups are normalized; 1 for the whole operand
 * @param groupLength how many elements one group covers, or 0 when {@code groups} is 1
 * @param epsilon the floor on the divisor
 * @param dataType the representation the normalization executes at
 */
public record L2Norm(
        OperandRef input,
        OperandRef output,
        int groups,
        int groupLength,
        float epsilon,
        DataType dataType)
        implements Operation {

    public L2Norm {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (groups < 1) {
            throw new IllegalArgumentException("groups must be at least 1: " + groups);
        }
        if (groups > 1 && groupLength < 1) {
            throw new IllegalArgumentException(
                    "a norm over " + groups + " groups must state the group length");
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.L2_NORM;
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
