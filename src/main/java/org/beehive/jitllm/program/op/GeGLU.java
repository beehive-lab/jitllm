package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * The GeGLU feed-forward activation: {@code gelu(gate) * up}, elementwise.
 *
 * <p>Gemma4's gate. Structurally {@link SwiGLU} with a different activation on the gate half, and a
 * separate operation rather than a parameter on it: exactly two gated activations exist across all
 * eight families, and an enum over two values would turn {@code SwiGLU} from a <i>name</i> into a
 * <i>parameter value</i>. If a third ever arrives, generalizing then is a mechanical change over
 * two call sites with equivalence tests already in place.
 *
 * @param gate the gate projection, to which GeLU is applied
 * @param up the up projection, multiplied in
 * @param output the activated hidden state; may be {@code gate}
 * @param dataType the representation the activation executes at
 */
public record GeGLU(OperandRef gate, OperandRef up, OperandRef output, DataType dataType)
        implements Operation {

    public GeGLU {
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
    }

    @Override
    public OperationKind kind() {
        return OperationKind.GEGLU;
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
