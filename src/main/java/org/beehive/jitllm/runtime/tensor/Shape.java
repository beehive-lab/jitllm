package org.beehive.jitllm.runtime.tensor;

import java.util.Arrays;

/**
 * A tensor's logical dimensions.
 *
 * <p>Immutable. Dimensions are in <b>GGUF {@code ne} order, dimension 0 fastest-varying</b> — the
 * order the file states them in, kept rather than normalized so that nothing has to remember which
 * convention a given descriptor follows. That convention is the point: {@code 4096 × 11008} and
 * {@code 11008 × 4096} hold the same number of elements and mean different things, and an unstated
 * order brings that confusion straight back through the check meant to prevent it.
 */
public final class Shape {

    private final long[] dimensions;
    private final long elementCount;

    private Shape(long[] dimensions, long elementCount) {
        this.dimensions = dimensions;
        this.elementCount = elementCount;
    }

    /**
     * @param dimensions in GGUF {@code ne} order; copied, so the caller may reuse its array
     * @throws IllegalArgumentException if empty, non-positive, or the product overflows
     */
    public static Shape of(long... dimensions) {
        if (dimensions == null || dimensions.length == 0) {
            throw new IllegalArgumentException("a tensor has at least one dimension");
        }
        long count = 1;
        for (long dimension : dimensions) {
            if (dimension <= 0) {
                throw new IllegalArgumentException(
                        "dimensions must be positive: " + Arrays.toString(dimensions));
            }
            try {
                count = Math.multiplyExact(count, dimension);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException(
                        "element count overflows a long: " + Arrays.toString(dimensions), overflow);
            }
        }
        return new Shape(dimensions.clone(), count);
    }

    /** Convenience for the file's own {@code int[]} shape. */
    public static Shape of(int[] dimensions) {
        if (dimensions == null || dimensions.length == 0) {
            throw new IllegalArgumentException("a tensor has at least one dimension");
        }
        long[] widened = new long[dimensions.length];
        for (int i = 0; i < dimensions.length; i++) {
            widened[i] = dimensions[i];
        }
        return of(widened);
    }

    public int rank() {
        return dimensions.length;
    }

    /** Dimension {@code index}, in {@code ne} order. */
    public long dimension(int index) {
        return dimensions[index];
    }

    /** A copy; the shape stays immutable. */
    public long[] dimensions() {
        return dimensions.clone();
    }

    /** The validated product of the dimensions. */
    public long elementCount() {
        return elementCount;
    }

    /**
     * The element count as an {@code int}, for the storage APIs that index with one.
     *
     * @throws IllegalStateException naming the caller's context when the tensor is too large
     */
    public int elementCountAsInt(String tensorName) {
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalStateException(
                    "tensor "
                            + tensorName
                            + " has "
                            + elementCount
                            + " elements, more than an int-indexed array can hold");
        }
        return (int) elementCount;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Shape shape && Arrays.equals(dimensions, shape.dimensions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < dimensions.length; i++) {
            text.append(i == 0 ? "" : " × ").append(dimensions[i]);
        }
        return text.toString();
    }
}
