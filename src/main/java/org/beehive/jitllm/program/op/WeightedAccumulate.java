package org.beehive.jllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Accumulating a branch result into the residual stream with a scalar weight.
 *
 * <p>Distinct from {@link ResidualAdd}, which adds a branch back unweighted, and from {@link Scale}
 * followed by an add, which would scale a buffer the caller may still need. In a mixture of experts
 * this <b>is</b> the residual connection: there is no separate add afterwards, so the order in
 * which these run is the order the floating-point sum is formed.
 *
 * <p>The weight is read from an element of a buffer rather than carried as a parameter, because it
 * is computed per token — a routing probability, or a gate score — and a parameter would make it
 * part of the program's identity.
 *
 * @param stream the residual stream, accumulated into in place
 * @param branch the branch result being added
 * @param weightSource the buffer holding the scalar weight
 * @param weightIndex which element of it to read
 * @param gate what to apply to the weight before using it
 * @param length how many elements to accumulate
 * @param dataType the representation the accumulation executes at
 */
public record WeightedAccumulate(
        OperandRef stream,
        OperandRef branch,
        OperandRef weightSource,
        int weightIndex,
        GateActivation gate,
        int length,
        DataType dataType)
        implements Operation {

    /**
     * What is applied to the scalar weight before it multiplies the branch.
     *
     * <p>Two values, and deliberately <b>not</b> a general elementwise activation operation: a
     * routed expert's weight is already a probability, while the shared expert's is a raw gate
     * score that needs a logistic sigmoid. Introducing a general {@code Activation} to unify this
     * with a feed-forward's GeLU would couple two families' vocabularies for one shared value
     */
    public enum GateActivation {
        /** The weight is used as it is — a routing probability. */
        NONE,
        /** The weight is a gate score; the logistic sigmoid of it is used. */
        LOGISTIC
    }

    public WeightedAccumulate {
        Objects.requireNonNull(stream, "stream");
        Objects.requireNonNull(branch, "branch");
        Objects.requireNonNull(weightSource, "weightSource");
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(dataType, "dataType");
        if (weightIndex < 0) {
            throw new IllegalArgumentException("weightIndex must not be negative: " + weightIndex);
        }
        if (length <= 0) {
            throw new IllegalArgumentException("length must be positive: " + length);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.WEIGHTED_ACCUMULATE;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(branch, weightSource);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(stream);
    }
}
