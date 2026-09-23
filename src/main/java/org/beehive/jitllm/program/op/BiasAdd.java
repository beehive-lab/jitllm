package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Adding a learned bias vector to a projection.
 *
 * <p>Arithmetically the same elementwise addition as {@link ResidualAdd}, and a separate operation
 * anyway, because the two are different work in every way that matters to a program: this one's
 * right-hand side is a <b>model weight</b> addressed by role, not an activation coming back from a
 * branch, and it sits immediately after a projection rather than at the end of a block. A program
 * that called both "add" would need the operand kinds inspected to tell a bias from a residual
 * connection.
 *
 * <p>Qwen2 and Qwen2-MoE are why it exists: they carry {@code q_bias}, {@code k_bias} and {@code
 * v_bias} where Llama carries none. Added under the rule {@link OperationKind} states — the work
 * genuinely differs, rather than a family arranging the same work differently.
 *
 * @param input the projection the bias is added to
 * @param bias the learned bias vector
 * @param output the result; may be {@code input}
 * @param dataType the representation the addition executes at
 */
public record BiasAdd(
        OperandRef input, OperandRef.Weight bias, OperandRef output, DataType dataType)
        implements Operation {

    public BiasAdd {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.BIAS_ADD;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(input, bias);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
