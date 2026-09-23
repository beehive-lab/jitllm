package org.beehive.jllm.program;

import java.util.Objects;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.Shape;

/**
 * One entry in a program's binding surface.
 *
 * <p>Three classes, and the split is the whole point. It separates the arrays whose identity is
 * fixed for the life of the compiled program from the values an invocation varies and the results a
 * caller reads — so that <b>an invocation moves values, never arrays</b>.
 *
 * <p>Capability <a href="./././././././docs/architecture/memory-and-concurrency.md">C1</a> is why:
 * a captured CUDA graph bakes device addresses, and re-pointing a captured buffer produces <b>wrong
 * output rather than an error</b>. Under this split there is no rebind operation to get wrong.
 */
public sealed interface BindingEntry {

    /**
     * Index of this entry within the signature's binding list. Its identity for cross-references.
     */
    int index();

    /**
     * A device array bound into the captured graph.
     *
     * <p>Its identity is <b>fixed for the life of the compiled program</b>. It is never rebound,
     * only written into.
     */
    record ProgramFixed(
            int index,
            BindingRole role,
            String name,
            DataType dataType,
            Shape shape,
            Direction direction)
            implements BindingEntry {
        public ProgramFixed {
            requireIndex(index);
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(direction, "direction");
        }
    }

    /**
     * A scalar the caller supplies per invocation, delivered by writing into a carrier.
     *
     * <p>{@code carrier} is the index of a {@link ProgramFixed} entry — a control or staging array
     * — and {@code elementOffset} is where in it this value goes. The entry carries a value and a
     * destination; it never carries an array, which is what makes rebinding unrepresentable.
     */
    record InvocationValue(
            int index,
            ValueId id,
            ValueType valueType,
            int carrier,
            long elementOffset,
            long elementCount)
            implements BindingEntry {
        public InvocationValue {
            requireIndex(index);
            Objects.requireNonNull(id, "id");
            Objects.requireNonNull(valueType, "valueType");
            requireIndex(carrier);
            if (elementOffset < 0) {
                throw new IllegalArgumentException(
                        "elementOffset must not be negative: " + elementOffset);
            }
            if (elementCount <= 0) {
                throw new IllegalArgumentException(
                        "elementCount must be positive: " + elementCount);
            }
        }
    }

    /** Something the caller reads once the invocation completes, from a declared carrier. */
    record HostVisibleResult(
            int index,
            ResultId id,
            ValueType scalarType,
            int carrier,
            long elementOffset,
            long elementCount)
            implements BindingEntry {
        public HostVisibleResult {
            requireIndex(index);
            Objects.requireNonNull(id, "id");
            requireIndex(carrier);
            if (elementOffset < 0) {
                throw new IllegalArgumentException(
                        "elementOffset must not be negative: " + elementOffset);
            }
            if (elementCount <= 0) {
                throw new IllegalArgumentException(
                        "elementCount must be positive: " + elementCount);
            }
        }

        /** Whether this result is a scalar rather than a tensor read out of its carrier. */
        public boolean isScalar() {
            return scalarType != null;
        }
    }

    private static void requireIndex(int index) {
        if (index < 0) {
            throw new IllegalArgumentException("binding indices must not be negative: " + index);
        }
    }
}
