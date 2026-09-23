package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Bounding the logits smoothly: {@code output[i] = cap * tanh(input[i] / cap)}.
 *
 * <p>Runs <b>after the vocabulary projection and before sampling</b>.
 *
 * <h2>Forward computation, not generation policy</h2>
 *
 * <p>It acts where sampling acts, which makes the classification worth stating rather than
 * assuming. It is part of the forward pass: the model's output <i>is</i> the capped value, and a
 * caller that read the logits without it would be reading something the model does not produce.
 * Treating it as a sampler setting would make the host and device sampling paths describe different
 * computations — so <b>its placement is identical whether sampling runs on the host or on the
 * device</b>.
 *
 * <h2>Absence, not a neutral value</h2>
 *
 * <p>A model without soft-capping <b>omits the component</b>. There is no cap value meaning "do
 * nothing": {@code cap == 0} is a division by zero, not an identity, which is why the constructor
 * refuses it rather than letting a mis-read configuration produce silent NaNs.
 *
 * @param input the logits
 * @param output the capped logits; may be {@code input}
 * @param cap the soft-cap value, strictly positive
 * @param dataType the representation the logits are held in
 */
public record LogitSoftCap(OperandRef input, OperandRef output, float cap, DataType dataType)
        implements Operation {

    public LogitSoftCap {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (!(cap > 0f) || !Float.isFinite(cap)) {
            throw new IllegalArgumentException(
                    "cap must be finite and positive; a model without"
                            + " soft-capping omits this operation rather than passing "
                            + cap);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.LOGIT_SOFT_CAP;
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
