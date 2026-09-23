package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Root-mean-square normalization with a learned per-channel scale.
 *
 * <p>{@code output = input / rms(input) * weight}, where the epsilon guards the reciprocal square
 * root. Epsilon comes from the model's configuration and is therefore part of the description, not
 * something supplied per invocation.
 *
 * <p><b>The weight is optional, and its absence is stated rather than faked.</b> Gemma4 normalizes
 * its values with no learned scale at all. An all-ones weight would be the obvious substitute and
 * is refused: besides being untrue, it adds one multiplication per element that the model does not
 * perform.
 *
 * <p>Families differ in where they apply it and in one detail this description deliberately does
 * not model: some add one to the stored weight before scaling. That is a property of how a family
 * stores its weights, and it belongs to the architecture that assembles the program or to the
 * loader, not to the meaning of "normalize".
 *
 * <h2>Groups</h2>
 *
 * <p>Some families normalize <b>per head</b> rather than over the whole operand: Qwen3 applies its
 * query and key norms once per head, at that head's offset and for that head's length. A group
 * count and a group length say so in one component. The default — one group over everything — is
 * what every other family does, so nothing else changes shape.
 *
 * @param input the activations to normalize
 * @param weight the learned scale, or empty for the unweighted form
 * @param output where the normalized activations are written; may be {@code input}
 * @param epsilon the variance epsilon from the model configuration
 * @param dataType the representation the normalization executes at
 * @param groups how many independent groups are normalized; 1 for the whole operand
 * @param groupLength how many elements one group covers, or 0 when {@code groups} is 1 and the
 *     length is the operand's own
 */
public record RmsNorm(
        OperandRef input,
        Optional<OperandRef.Weight> weight,
        OperandRef output,
        float epsilon,
        DataType dataType,
        int groups,
        int groupLength)
        implements Operation {

    public RmsNorm {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        if (groups < 1) {
            throw new IllegalArgumentException("groups must be at least 1: " + groups);
        }
        if (groups == 1 && groupLength < 0) {
            throw new IllegalArgumentException("groupLength must not be negative: " + groupLength);
        }
        if (groups > 1 && groupLength < 1) {
            // A grouped norm without a length would be N norms over an unstated span, which is not
            // a description of anything — the failure this operation exists to prevent.
            throw new IllegalArgumentException(
                    "a norm over " + groups + " groups must state the group length");
        }
    }

    /** The ungrouped form: one normalization over the whole operand. */
    public RmsNorm(
            OperandRef input,
            Optional<OperandRef.Weight> weight,
            OperandRef output,
            float epsilon,
            DataType dataType) {
        this(input, weight, output, epsilon, dataType, 1, 0);
    }

    /** Whether this normalizes per group rather than over the whole operand. */
    public boolean isGrouped() {
        return groups > 1;
    }

    @Override
    public OperationKind kind() {
        return OperationKind.RMS_NORM;
    }

    @Override
    public List<OperandRef> inputs() {
        return weight.<List<OperandRef>>map(w -> List.of(input, w)).orElseGet(() -> List.of(input));
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
