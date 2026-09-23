package org.beehive.jllm.runtime.tensor;

import java.util.Objects;

/**
 * What a tensor is, without saying where it lives.
 *
 * <p>Immutable. A descriptor carries the data type, the logical shape, the role and the layout;
 * storage — a host memory segment, a device array — is materialized <i>from</i> a descriptor and is
 * owned by whoever allocates it. That split is the point: descriptors are backend-neutral and can
 * be reasoned about before anything is allocated, and the layers above the format can talk about
 * tensors without naming a file type or a device type.
 *
 * <p>The shape is what makes this more than bookkeeping. Today's tensors are flat and index
 * linearly, so nothing can detect a transposed or mis-bound weight; a descriptor can, because
 * {@code 4096 × 11008} is not {@code 11008 × 4096}.
 */
public final class TensorDescriptor {

    private final String name;
    private final DataType dataType;
    private final Shape shape;
    private final TensorRole role;
    private final TensorLayout layout;

    public TensorDescriptor(
            String name, DataType dataType, Shape shape, TensorRole role, TensorLayout layout) {
        this.name = Objects.requireNonNull(name, "name");
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.shape = Objects.requireNonNull(shape, "shape");
        this.role = Objects.requireNonNull(role, "role");
        this.layout = Objects.requireNonNull(layout, "layout");
    }

    /** The tensor's own name, kept so that a failure can say which tensor it was. */
    public String name() {
        return name;
    }

    public DataType dataType() {
        return dataType;
    }

    public Shape shape() {
        return shape;
    }

    public TensorRole role() {
        return role;
    }

    public TensorLayout layout() {
        return layout;
    }

    public long elementCount() {
        return shape.elementCount();
    }

    /** How many bytes storage for this descriptor occupies. */
    public long byteSize() {
        return layout.byteSize(shape.elementCount());
    }

    /**
     * Checks that this descriptor is what an operation expects to be given.
     *
     * <p>Where a mis-bound weight becomes an error instead of a wrong answer. Flat indexing cannot
     * tell a query projection from a key projection of the same size; a shape and a role can.
     *
     * @throws IllegalArgumentException naming both sides when they disagree
     */
    public void requireCompatibleWith(TensorDescriptor expected) {
        Objects.requireNonNull(expected, "expected");
        if (!shape.equals(expected.shape)) {
            throw new IllegalArgumentException(
                    "tensor "
                            + name
                            + " has shape "
                            + shape
                            + " where "
                            + expected.name
                            + " expects "
                            + expected.shape);
        }
        if (dataType != expected.dataType) {
            throw new IllegalArgumentException(
                    "tensor "
                            + name
                            + " is "
                            + dataType
                            + " where "
                            + expected.name
                            + " expects "
                            + expected.dataType);
        }
    }

    @Override
    public String toString() {
        return name + " [" + role + " " + dataType + " " + shape + "]";
    }
}
