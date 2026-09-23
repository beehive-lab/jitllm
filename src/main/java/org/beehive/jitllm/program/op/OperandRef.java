package org.beehive.jllm.program.op;

import java.util.Objects;
import org.beehive.jllm.runtime.tensor.TensorRole;

/**
 * A reference to a tensor an operation reads or writes, by name rather than by handle.
 *
 * <p>An operation is a description, so it cannot hold a buffer, a device address or a {@code
 * TensorDescriptor} bound to one. It names what it needs and leaves resolution to the program layer
 * which is also where the reference acquires a binding class: a {@link Weight} resolves to a
 * program-fixed device binding, and an {@link Activation} to a workspace or staging array that is
 * equally program-fixed but written per invocation.
 *
 * <p>The split is exactly that — weights are addressed by {@link TensorRole}, which the runtime
 * already owns as a closed set, while activations are addressed only by name because no role
 * vocabulary describes them and inventing one would be a second dispatch key.
 *
 * <p>Sealed: a third kind of operand would be a change to how programs bind, not a convenience.
 */
public sealed interface OperandRef {

    /** The name this operand is resolved by. Unique within one program. */
    String name();

    /**
     * A model weight, addressed by the role the runtime already assigns it.
     *
     * <p>The role is what makes an operation family-independent: an architecture assembles {@code
     * MatVec} over {@code ATTENTION_QUERY} without the operation knowing which family's query
     * projection it is.
     */
    record Weight(String name, TensorRole role) implements OperandRef {
        public Weight {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(role, "role");
        }
    }

    /**
     * An activation, a scratch buffer or a result — anything that is not a model weight.
     *
     * <p>Named rather than typed by role deliberately: activations have no closed role vocabulary,
     * and adding one would create a second thing architecture code could branch on.
     */
    record Activation(String name) implements OperandRef {
        public Activation {
            Objects.requireNonNull(name, "name");
        }
    }
}
