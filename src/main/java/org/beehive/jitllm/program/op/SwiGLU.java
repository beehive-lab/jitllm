package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * The SwiGLU feed-forward activation: {@code silu(gate) * up}, elementwise.
 *
 * <p>Named for what every supported family actually computes. A family using a different gate
 * activation would be a new kind rather than a parameter on this one — the set of activation
 * functions in use is small enough that a function selector here would be a dispatch key with one
 * value.
 *
 * @param gate the gate projection, to which SiLU is applied
 * @param up the up projection, multiplied in
 * @param output the activated hidden state; may be {@code gate}
 * @param dataType the representation the hidden state is held in
 */
public record SwiGLU(OperandRef gate, OperandRef up, OperandRef output, DataType dataType)
        implements Operation {

    public SwiGLU {
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SWIGLU;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(gate, up);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
